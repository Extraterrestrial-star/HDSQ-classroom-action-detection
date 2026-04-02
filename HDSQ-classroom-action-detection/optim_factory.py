import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    """Recognizes ViT keys (blocks, patch_embed) and VideoMamba keys (layers) for layer-wise decay."""
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed") or var_name.startswith("encoder.patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks") or var_name.startswith("encoder.blocks"):
        if var_name.startswith("encoder.blocks"):
            var_name = var_name[8:]
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith("layers"):
        # VideoMamba / Mamba backbone: layers.0, layers.1, ...
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None,
                         lr_scale=1.0,
                         slow_assigner=None, fast_assigner=None,
                         lr_scale_slow_backbone=1.0, lr_scale_fast_backbone=0.1, lr_scale_heads=5.0):
    """
    v3: When slow_assigner and fast_assigner are provided (dual_stream), use separate layer decay per tower:
    - slow backbone: lr_scale_slow_backbone (default 1.0x) with slow_assigner (12 layers)
    - fast backbone: lr_scale_fast_backbone (0.1x) with fast_assigner (32 layers)
    - fusion_adapter + head_*: lr_scale_heads (5x~10x)
    """
    parameter_group_names = {}
    parameter_group_vars = {}
    is_twostream = hasattr(model, 'slow') and hasattr(model, 'fast')
    use_dual_assigner = is_twostream and slow_assigner is not None and fast_assigner is not None

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if is_twostream:
            no_decay = len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or any(name.endswith(s) for s in skip_list)
            this_weight_decay = 0.0 if no_decay else weight_decay
            # TwoStreamInternVideoMamba uses slow./fast.; TwoStreamCoDAQ wraps in _twostream, so params are _twostream.slow.* / _twostream.fast.*
            if name.startswith('slow.'):
                inner = name[5:]
                is_slow = True
            elif name.startswith('_twostream.slow.'):
                inner = name[len('_twostream.slow.'):]
                is_slow = True
            else:
                is_slow = False
            if is_slow:
                if use_dual_assigner:
                    layer_id = slow_assigner.get_layer_id(inner)
                    scale = slow_assigner.get_scale(layer_id) * lr_scale_slow_backbone
                else:
                    layer_id = get_num_layer(inner) if get_num_layer is not None else None
                    scale = get_layer_scale(layer_id) if (get_layer_scale is not None and layer_id is not None) else 1.0
                group_name = "twostream_slow_layer%d_%s" % (layer_id if layer_id is not None else -1, "no_decay" if no_decay else "decay")
            elif name.startswith('fast.'):
                inner = name[5:]
                if use_dual_assigner:
                    layer_id = fast_assigner.get_layer_id(inner)
                    scale = fast_assigner.get_scale(layer_id) * lr_scale_fast_backbone
                else:
                    layer_id = get_num_layer(inner) if get_num_layer is not None else None
                    scale = get_layer_scale(layer_id) if (get_layer_scale is not None and layer_id is not None) else 1.0
                group_name = "twostream_fast_layer%d_%s" % (layer_id if layer_id is not None else -1, "no_decay" if no_decay else "decay")
            elif name.startswith('_twostream.fast.'):
                inner = name[len('_twostream.fast.'):]
                if use_dual_assigner:
                    layer_id = fast_assigner.get_layer_id(inner)
                    scale = fast_assigner.get_scale(layer_id) * lr_scale_fast_backbone
                else:
                    layer_id = get_num_layer(inner) if get_num_layer is not None else None
                    scale = get_layer_scale(layer_id) if (get_layer_scale is not None and layer_id is not None) else 1.0
                group_name = "twostream_fast_layer%d_%s" % (layer_id if layer_id is not None else -1, "no_decay" if no_decay else "decay")
            else:
                # fusion_adapter, head_fused, head_fast, head_slow, fast_temporal_module
                scale = lr_scale_heads if use_dual_assigner else 1.0
                group_name = "twostream_rest_" + ("no_decay" if no_decay else "decay")
            layer_id = None
        else:
            # Single-stream / encoder
            if (len(param.shape) == 1 or name.endswith(".bias") or name in skip_list) and name.startswith('encoder.'):
                group_name = "no_decay_encoder"
                this_weight_decay = 0.
                scale = 1.0
            elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = "no_decay_others"
                this_weight_decay = 0.
                scale = lr_scale
            elif name.startswith('encoder.'):
                group_name = "decay_encoder"
                this_weight_decay = weight_decay
                scale = 1.0
            else:
                group_name = "decay_others"
                this_weight_decay = weight_decay
                scale = lr_scale

            if get_num_layer is not None:
                layer_id = get_num_layer(name)
                group_name = "layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

        if group_name not in parameter_group_names:
            if not is_twostream and get_layer_scale is not None and layer_id is not None:
                scale = get_layer_scale(layer_id) * scale
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # Diagnostic: verify CoDA-Q / fusion_adapter params are correctly excluded from weight decay
    if is_twostream:
        _nwd_params = []
        _wd_params = []
        for gname, ginfo in parameter_group_names.items():
            for pname in ginfo["params"]:
                if ginfo["weight_decay"] == 0.0:
                    _nwd_params.append(pname)
                else:
                    _wd_params.append(pname)
        _codaq_keys = [p for p in _nwd_params if "head_fused" in p or "gamma" in p]
        _codaq_wd_leak = [p for p in _wd_params if "head_fused" in p and ("embed" in p or "pos" in p)]
        print("[OptimFactory] CoDA-Q / fusion params with NO weight decay (%d): %s" % (len(_codaq_keys), _codaq_keys))
        if _codaq_wd_leak:
            print("[OptimFactory] WARNING: CoDA-Q embed/pos params STILL have weight decay: %s" % _codaq_wd_leak)
        else:
            print("[OptimFactory] OK: All CoDA-Q embed/pos params correctly skip weight decay.")

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None,
                     lr_scale=1.0,
                     slow_assigner=None, fast_assigner=None,
                     lr_scale_slow_backbone=1.0, lr_scale_fast_backbone=0.1, lr_scale_heads=5.0):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale, lr_scale=lr_scale,
            slow_assigner=slow_assigner, fast_assigner=fast_assigner,
            lr_scale_slow_backbone=lr_scale_slow_backbone, lr_scale_fast_backbone=lr_scale_fast_backbone, lr_scale_heads=lr_scale_heads)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
