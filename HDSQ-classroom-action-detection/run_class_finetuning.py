import argparse
import datetime
from operator import is_
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets import build_ava_dataset, BatchCollator, build_kinetics_dataset, build_ak_dataset
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate
import utils
import modeling_finetune
from data.transforms import build_transforms
import torch.nn as nn
from losses.asl import AsymmetricLossMultiLabel  # 保留 ASL


def get_args():
  parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification',
                                   add_help=False)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=30, type=int)
  parser.add_argument('--update_freq', default=1, type=int)
  parser.add_argument('--save_ckpt_freq', default=100, type=int)
  parser.add_argument('--val_freq', default=2, type=int)
  parser.add_argument('--enable_randaug', action='store_true', default=False)
  parser.add_argument('--aug_type', default='strong', type=str,
                      choices=['strong', 'default', 'default_w_cutout', 'default_wo_affine'])

  # Model parameters
  parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                      help='Name of model to train')
  parser.add_argument('--tubelet_size', type=int, default=2)
  parser.add_argument('--input_size', default=224, type=int,
                      help='videos input size')

  parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                      help='Dropout rate (default: 0.)')
  parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                      help='Attention dropout rate (default: 0.)')
  parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                      help='Drop path rate (default: 0.1)')

  parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
  parser.add_argument('--model_ema', action='store_true', default=False)
  parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
  parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
  parser.add_argument('--head_type', type=str, default='linear', choices=['linear', 'acar', 'codaq'])

  # CoDA-Q and two-stream+CoDA-Q
  parser.add_argument('--codaq_num_corr_queries', default=4, type=int)
  parser.add_argument('--codaq_num_disc_queries', default=4, type=int)
  parser.add_argument('--codaq_decoder_depth', default=1, type=int)
  parser.add_argument('--codaq_decoder_num_heads', default=4, type=int)
  parser.add_argument('--codaq_dim_feedforward', default=1024, type=int)
  parser.add_argument('--codaq_dropout', default=0.1, type=float)
  parser.add_argument('--codaq_ctx_ext_x', default=0.2, type=float)
  parser.add_argument('--codaq_ctx_ext_y', default=0.1, type=float)
  parser.add_argument('--codaq_lambda_cd', default=0.05, type=float)
  parser.add_argument('--codaq_lambda_div', default=0.05, type=float)
  parser.add_argument('--codaq_temporal_samples', default=4, type=int)
  parser.add_argument('--codaq_use_motion_tokens', action='store_true')
  parser.add_argument('--codaq_no_gate_disc', action='store_true')
  parser.add_argument('--codaq_reg_warmup_steps', default=2000, type=int)
  parser.add_argument('--codaq_reg_on_all_tokens', action='store_true')
  parser.add_argument('--codaq_vis_freq', default=0, type=int)
  parser.add_argument('--codaq_vis_max_rois', default=4, type=int)
  parser.add_argument('--codaq_vis_roi_index', default=0, type=int)

  # Optimizer parameters
  parser.add_argument('--accumulation_step', type=int, default=1)
  parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                      help='Optimizer (default: "adamw"')
  parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                      help='Optimizer Epsilon (default: 1e-8)')
  parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                      help='Optimizer Betas (default: None, use opt default)')
  parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                      help='Clip gradient norm (default: None, no clipping)')
  parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                      help='SGD momentum (default: 0.9)')
  parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='weight decay (default: 0.05)')
  parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

  parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                      help='learning rate (default: 1e-3)')
  parser.add_argument('--layer_decay', type=float, default=0.75)

  parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                      help='warmup learning rate (default: 1e-6)')
  parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                      help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

  parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                      help='epochs to warmup LR, if scheduler supports')
  parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                      help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

  # Augmentation parameters
  parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                      help='Color jitter factor (default: 0.4)')
  parser.add_argument('--num_sample', type=int, default=2,
                      help='Repeated_aug (default: 2)')
  parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                      help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
  parser.add_argument('--smoothing', type=float, default=0.1,
                      help='Label smoothing (default: 0.1)')
  parser.add_argument('--train_interpolation', type=str, default='bicubic',
                      help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

  # Evaluation parameters
  parser.add_argument('--crop_pct', type=float, default=None)
  parser.add_argument('--short_side_size', type=int, default=224)
  parser.add_argument('--test_num_segment', type=int, default=5)
  parser.add_argument('--test_num_crop', type=int, default=3)

  # Random Erase params
  parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                      help='Random erase prob (default: 0.25)')
  parser.add_argument('--remode', type=str, default='pixel',
                      help='Random erase mode (default: "pixel")')
  parser.add_argument('--recount', type=int, default=1,
                      help='Random erase count (default: 1)')
  parser.add_argument('--resplit', action='store_true', default=False,
                      help='Do not random erase first (clean) augmentation split')

  # Mixup params
  parser.add_argument('--mixup', type=float, default=0.8,
                      help='mixup alpha, mixup enabled if > 0.')
  parser.add_argument('--cutmix', type=float, default=1.0,
                      help='cutmix alpha, cutmix enabled if > 0.')
  parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                      help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
  parser.add_argument('--mixup_prob', type=float, default=1.0,
                      help='Probability of performing mixup or cutmix when either/both is enabled')
  parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                      help='Probability of switching to cutmix when both mixup and cutmix enabled')
  parser.add_argument('--mixup_mode', type=str, default='batch',
                      help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

  # Finetuning params
  parser.add_argument('--finetune', default='', help='finetune from checkpoint')
  parser.add_argument('--model_key', default='model|module', type=str)
  parser.add_argument('--model_prefix', default='', type=str)
  parser.add_argument('--init_scale', default=0.001, type=float)
  parser.add_argument('--use_mean_pooling', action='store_true')
  parser.set_defaults(use_mean_pooling=True)
  parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

  # Dataset parameters
  parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                      help='dataset path')
  parser.add_argument('--eval_data_path', default=None, type=str,
                      help='dataset path for evaluation')
  parser.add_argument('--nb_classes', default=80, type=int,
                      help='number of the classification types')
  parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
  parser.add_argument('--num_segments', type=int, default=1)
  parser.add_argument('--num_frames', type=int, default=16)
  parser.add_argument('--sampling_rate', type=int, default=4)

  # --- Two-stream (InternVideo slow + VideoMamba fast) ---
  parser.add_argument('--dual_stream', action='store_true', default=False,
                      help='Enable two-stream model (requires a twostream_* model).')
  parser.add_argument('--slow_frames', type=int, default=16,
                      help='Slow pathway sampled frames (default: 16).')
  parser.add_argument('--slow_stride', type=int, default=4,
                      help='Slow pathway temporal stride inside the aligned window (default: 4).')
  parser.add_argument('--fast_frames', type=int, default=32,
                      help='Fast pathway sampled frames (default: 32).')
  parser.add_argument('--fast_stride', type=int, default=2,
                      help='Fast pathway temporal stride inside the aligned window (default: 2).')
  parser.add_argument('--slow_jitter', action='store_true', default=False,
                      help='Enable random temporal jitter for slow pathway sampling.')
  parser.add_argument('--videomamba_ckpt', type=str, default='',
                      help='Path to VideoMamba pretrained checkpoint (*.pth).')
  parser.add_argument('--lambda_slow', type=float, default=0.3,
                      help='Loss weight for slow-only auxiliary head.')
  parser.add_argument('--lambda_fast', type=float, default=0.3,
                      help='Loss weight for fast-only auxiliary head.')
  parser.add_argument('--lambda_kd', type=float, default=0.1,
                      help='Loss weight for Bernoulli-KL distillation (slow/fast -> fused).')
  parser.add_argument('--kd_temperature', type=float, default=1.0,
                      help='Temperature for distillation sigmoid(logits/T).')
  parser.add_argument('--p_drop_fast', type=float, default=0.2,
                      help='Stream dropout prob for fast stream (fusion path only).')
  parser.add_argument('--p_drop_slow', type=float, default=0.2,
                      help='Stream dropout prob for slow stream (fusion path only).')
  parser.add_argument('--freeze_slow_epochs', type=int, default=0,
                      help='Two-stream v3: freeze slow backbone for first N epochs (Stage-1), then unfreeze (Stage-2). 0 = no freeze.')
  parser.add_argument('--p_drop_fast_stage2', type=float, default=None,
                      help='If set, use this p_drop_fast when epoch >= freeze_slow_epochs (Stage-2); else use p_drop_fast. E.g. 0.2.')
  parser.add_argument('--lr_scale_slow_backbone', type=float, default=1.0,
                      help='v3: LR scale for slow backbone (default 1.0x).')
  parser.add_argument('--lr_scale_fast_backbone', type=float, default=0.1,
                      help='v3: LR scale for fast backbone (default 0.1x).')
  parser.add_argument('--lr_scale_heads', type=float, default=5.0,
                      help='v3: LR scale for fusion_adapter + heads (default 5x).')
  parser.add_argument('--lambda_cons', type=float, default=0.0,
                      help='v3: Consistency loss weight (stopgrad z_fused -> fast). Use warmup 0.02~0.05; 0 = off.')
  parser.add_argument('--data_set', default='ava',
                      choices=['ava', 'ava-kinetics'],
                      type=str, help='dataset')
  parser.add_argument('--output_dir', default='',
                      help='path where to save, empty for no saving')
  parser.add_argument('--log_dir', default=None,
                      help='path where to tensorboard log')
  parser.add_argument('--device', default='cuda',
                      help='device to use for training / testing')
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--resume', default='',
                      help='resume from checkpoint')
  parser.add_argument('--auto_resume', action='store_true')
  parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
  parser.set_defaults(auto_resume=True)

  parser.add_argument('--save_ckpt', action='store_true')
  parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
  parser.set_defaults(save_ckpt=True)

  parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                      help='start epoch')
  parser.add_argument('--eval', action='store_true',
                      help='Perform evaluation only')
  parser.add_argument('--dist_eval', action='store_true', default=False,
                      help='Enabling distributed evaluation')
  parser.add_argument('--num_workers', default=10, type=int)
  parser.add_argument('--pin_mem', action='store_true',
                      help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
  parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
  parser.set_defaults(pin_mem=True)

  # distributed training parameters
  parser.add_argument('--world_size', default=1, type=int,
                      help='number of distributed processes')
  parser.add_argument('--local_rank', default=-1, type=int)
  parser.add_argument('--dist_on_itp', action='store_true')
  parser.add_argument('--dist_url', default='env://',
                      help='url used to set up distributed training')

  parser.add_argument('--eval_kinetics', action='store_true', default=False)
  parser.add_argument('--enable_deepspeed', action='store_true', default=False)

  # --- Loss Choice: Keep ASL ---
  parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'asl'],
                      help='Loss type: BCE (baseline) or ASL (recommended)')
  parser.add_argument('--asl_gamma_pos', type=float, default=0.0)
  parser.add_argument('--asl_gamma_neg', type=float, default=4.0)
  parser.add_argument('--asl_clip', type=float, default=0.05)

  known_args, _ = parser.parse_known_args()

  if known_args.enable_deepspeed:
    try:
      import deepspeed
      from deepspeed import DeepSpeedConfig
      parser = deepspeed.add_config_arguments(parser)
      ds_init = deepspeed.initialize
    except:
      print("Please 'pip install deepspeed'")
      exit(0)
  else:
    ds_init = None

  return parser.parse_args(), ds_init


def main(args, ds_init):
  utils.init_distributed_mode(args)

  if ds_init is not None:
    utils.create_ds_config(args)

  if args.model == 'twostream_codaq_vit_base_patch16_224':
    args.num_frames = 64

  print(args)

  device = torch.device(args.device)

  # fix the seed for reproducibility
  seed = args.seed + utils.get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)

  cudnn.benchmark = True

  transform_train = build_transforms(is_train=True, args=args)
  transform_val = build_transforms(is_train=False, args=args)

  if args.data_set == 'ava':
    dataset_train = build_ava_dataset(is_train=True, transforms=transform_train, args=args)
  else:
    dataset_train = build_ak_dataset(is_train=True, transforms=transform_train)

  if args.eval_kinetics:
    dataset_val = build_kinetics_dataset(is_train=False, transforms=transform_val)
  else:
    dataset_val = build_ava_dataset(is_train=False, transforms=transform_val, args=args)

  num_tasks = utils.get_world_size()
  global_rank = utils.get_rank()
  sampler_train = torch.utils.data.DistributedSampler(
    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
  )
  print("Sampler_train = %s" % str(sampler_train))
  if args.dist_eval:
    if len(dataset_val) % num_tasks != 0:
      print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
            'This will slightly alter validation results as extra duplicate entries are added to achieve '
            'equal num of samples per-process.')
    sampler_val = torch.utils.data.DistributedSampler(
      dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
  else:
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

  if global_rank == 0 and args.log_dir is not None:
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
  else:
    log_writer = None

  collate_func = BatchCollator(size_divisible=16)
  data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
    collate_fn=collate_func,
  )
  data_loader_train.num_samples = len(dataset_train)

  if dataset_val is not None:
    data_loader_val = torch.utils.data.DataLoader(
      dataset_val, sampler=sampler_val,
      batch_size=4,
      num_workers=args.num_workers,
      pin_memory=args.pin_mem,
      drop_last=False,
      collate_fn=collate_func,
    )
    data_loader_val.num_samples = len(dataset_val)
  else:
    data_loader_val = None

  mixup_fn = None

  # --- Two-stream model guard: auto-switch model name if needed ---
  if getattr(args, "dual_stream", False) and (not str(args.model).startswith("twostream_")):
    print(f"[WARN] --dual_stream enabled but --model={args.model} is not a twostream_* model. "
          f"Auto switching to twostream_vit_base_patch16_224")
    args.model = "twostream_vit_base_patch16_224"

  # Extra kwargs for twostream / twostream_codaq / codaq
  twostream_kwargs = {}
  codaq_kwargs = {}
  if getattr(args, "dual_stream", False) or args.model == 'twostream_codaq_vit_base_patch16_224':
    twostream_kwargs = dict(
      slow_frames=getattr(args, 'slow_frames', 16),
      slow_stride=getattr(args, 'slow_stride', 4),
      fast_frames=getattr(args, 'fast_frames', 32),
      fast_stride=getattr(args, 'fast_stride', 2),
      slow_jitter=getattr(args, 'slow_jitter', False),
      videomamba_ckpt=getattr(args, 'videomamba_ckpt', ''),
      lambda_slow=getattr(args, 'lambda_slow', 0.3),
      lambda_fast=getattr(args, 'lambda_fast', 0.3),
      lambda_kd=getattr(args, 'lambda_kd', 0.1),
      lambda_cons=getattr(args, 'lambda_cons', 0.0),
      kd_temperature=getattr(args, 'kd_temperature', 1.0),
      p_drop_fast=getattr(args, 'p_drop_fast', 0.2),
      p_drop_slow=getattr(args, 'p_drop_slow', 0.2),
    )
  if args.head_type == 'codaq' or args.model == 'twostream_codaq_vit_base_patch16_224':
    codaq_kwargs = dict(
      codaq_num_corr_queries=args.codaq_num_corr_queries,
      codaq_num_disc_queries=args.codaq_num_disc_queries,
      codaq_decoder_depth=args.codaq_decoder_depth,
      codaq_decoder_num_heads=args.codaq_decoder_num_heads,
      codaq_dim_feedforward=args.codaq_dim_feedforward,
      codaq_dropout=args.codaq_dropout,
      codaq_ctx_ext=(args.codaq_ctx_ext_x, args.codaq_ctx_ext_y),
      codaq_lambda_cd=args.codaq_lambda_cd,
      codaq_lambda_div=args.codaq_lambda_div,
      codaq_temporal_samples=args.codaq_temporal_samples,
      codaq_use_motion_tokens=args.codaq_use_motion_tokens,
      codaq_gate_disc=(not args.codaq_no_gate_disc),
      codaq_reg_warmup_steps=args.codaq_reg_warmup_steps,
      codaq_reg_on_local_only=(not args.codaq_reg_on_all_tokens),
    )

  all_frames = 64 if args.model == 'twostream_codaq_vit_base_patch16_224' else (args.num_frames * args.num_segments)

  # 创建模型
  model = create_model(
    args.model,
    pretrained=False,
    num_classes=args.nb_classes,
    all_frames=all_frames,
    tubelet_size=args.tubelet_size,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    attn_drop_rate=args.attn_drop_rate,
    drop_block_rate=None,
    use_mean_pooling=args.use_mean_pooling,
    init_scale=args.init_scale,
    head_type=args.head_type,
    **twostream_kwargs,
    **codaq_kwargs,
  )

  # --- 修改开始: 兼容双流模型获取 patch_size ---
  if hasattr(model, 'patch_embed'):
    patch_size = model.patch_embed.patch_size
  elif hasattr(model, 'slow') and hasattr(model.slow, 'patch_embed'):
    # 对于 TwoStreamInternVideoMamba，patch_embed 在 slow 分支中
    patch_size = model.slow.patch_embed.patch_size
  print("Patch size = %s" % str(patch_size))

  args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
  args.patch_size = patch_size

  if args.finetune:
    if args.finetune.startswith('https'):
      checkpoint = torch.hub.load_state_dict_from_url(
        args.finetune, map_location='cpu', check_hash=True)
    else:
      checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load ckpt from %s" % args.finetune)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
      if model_key in checkpoint:
        checkpoint_model = checkpoint[model_key]
        print("Load state_dict by model_key = %s" % model_key)
        break
    if checkpoint_model is None:
      checkpoint_model = checkpoint
    state_dict = model.state_dict()
    # Remove head from checkpoint if shape mismatch (single-stream has head.*; twostream has head_fused/head_slow/head_fast)
    for k in ['head.weight', 'head.bias']:
      if k in checkpoint_model:
        if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
          print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
          del checkpoint_model[k]
        elif k not in state_dict:
          del checkpoint_model[k]
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
      if key.startswith('backbone.'):
        new_dict[key[9:]] = checkpoint_model[key]
      elif key.startswith('encoder.'):
        new_dict[key[8:]] = checkpoint_model[key]
      else:
        new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    is_twostream = hasattr(model, 'slow') and (getattr(args, 'dual_stream', False) or 'twostream' in str(args.model))
    if is_twostream:
      # Map ViT checkpoint keys to model.slow. Infer prefix from model so both TwoStreamCoDAQ (_twostream.slow) and TwoStreamInternVideoMamba (slow) work.
      vit = model.slow
      slow_prefix = 'slow.'
      for k in model.state_dict():
        if '.slow.' in k or (k.startswith('slow.') and 'blocks.' in k):
          idx = k.find('slow.') + 5
          slow_prefix = k[:idx]
          break
      slow_dict = OrderedDict()
      for key, value in checkpoint_model.items():
        if key.startswith('patch_embed.') or key.startswith('blocks.') or key == 'pos_embed' or key.startswith('norm.'):
          slow_dict[slow_prefix + key] = value
        elif key.startswith('fc_norm.'):
          slow_dict[slow_prefix + 'norm.' + key[8:]] = value
      checkpoint_model = slow_dict
      # Position embedding interpolation for slow stream
      pos_embed_key = slow_prefix + 'pos_embed'
      if pos_embed_key in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[pos_embed_key]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = vit.patch_embed.num_patches
        num_extra_tokens = vit.pos_embed.shape[-2] - num_patches
        t_frames = args.num_frames // vit.patch_embed.tubelet_size
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // t_frames) ** 0.5)
        new_size = int((num_patches // t_frames) ** 0.5)
        if orig_size != new_size:
          print("Position interpolate (slow stream) from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
          extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
          pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
          pos_tokens = pos_tokens.reshape(-1, t_frames, orig_size, orig_size, embedding_size)
          pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
          pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
          pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, t_frames, new_size, new_size, embedding_size)
          pos_tokens = pos_tokens.flatten(1, 3)
          new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
          checkpoint_model[pos_embed_key] = new_pos_embed
      # TwoStreamCoDAQ also exposes model.slow and model.patch_embed (= model.slow.patch_embed), so state_dict has both _twostream.slow.* and slow.* / patch_embed.*. Add aliases so loader does not report them as missing.
      if slow_prefix == '_twostream.slow.':
        for k, v in list(checkpoint_model.items()):
          if k.startswith('_twostream.slow.'):
            suffix = k[len('_twostream.slow.'):]
            checkpoint_model['slow.' + suffix] = v
            if suffix.startswith('patch_embed.'):
              checkpoint_model[suffix] = v  # model.patch_embed = model.slow.patch_embed
      print("[TwoStream] Loaded InternVideo checkpoint into slow stream (%d keys, prefix=%r)." % (
      len(checkpoint_model), slow_prefix))
    else:
      # Single-stream: interpolate position embedding using top-level model
      if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        if orig_size != new_size:
          print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
          extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
          pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
          pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size,
                                          orig_size, embedding_size)
          pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
          pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
          pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1,
                                                              args.num_frames // model.patch_embed.tubelet_size,
                                                              new_size, new_size, embedding_size)
          pos_tokens = pos_tokens.flatten(1, 3)
          new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
          checkpoint_model['pos_embed'] = new_pos_embed

    # Snapshot a few slow-stream tensors from checkpoint for verification (twostream only)
    if is_twostream:
      verify_keys = [k for k in list(checkpoint_model.keys()) if 'blocks.0.' in k or 'patch_embed.' in k][:4]
      ckpt_snapshot = {k: checkpoint_model[k].clone() for k in verify_keys}

    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    # Verify slow-stream pretrained load (twostream only)
    if is_twostream and ckpt_snapshot:
      model_sd = model.state_dict()
      n_ok, n_miss, n_diff = 0, 0, 0
      for k, v_ckpt in ckpt_snapshot.items():
        if k not in model_sd:
          n_miss += 1
          if n_miss <= 2:
            print("[TwoStream] Verify PRETRAIN: %s -> MISSING in model" % k)
        else:
          v_mod = model_sd[k]
          if v_mod.shape != v_ckpt.shape:
            n_diff += 1
            if n_diff <= 1:
              print("[TwoStream] Verify PRETRAIN: %s -> shape mismatch model %s vs ckpt %s" % (
              k, tuple(v_mod.shape), tuple(v_ckpt.shape)))
          elif torch.allclose(v_mod.float(), v_ckpt.float(), atol=1e-5, rtol=1e-3):
            n_ok += 1
          else:
            n_diff += 1
            if n_diff <= 1:
              print("[TwoStream] Verify PRETRAIN: %s -> value MISMATCH (max_diff=%.6f)" % (
              k, (v_mod.float() - v_ckpt.float()).abs().max().item()))
      print("[TwoStream] Verify PRETRAIN: %d keys OK, %d missing, %d mismatch (slow stream %s)" % (
        n_ok, n_miss, n_diff, "loaded successfully" if n_ok > 0 and n_miss == 0 and n_diff == 0 else "CHECK ABOVE"))

  model.to(device)

  model_ema = None

  model_without_ddp = model
  n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

  # --- Loss Setup: 已移除 CB (pos_weight from file) 和 LA (Logit Adjust) ---
  if args.loss == 'asl':
    print(f"Use ASL: gamma_pos={args.asl_gamma_pos}, gamma_neg={args.asl_gamma_neg}, clip={args.asl_clip}")
    # 注意：这里不再加载外部 pos_weight
    criterion = AsymmetricLossMultiLabel(
      gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg,
      clip=args.asl_clip)
  else:
    print("Use BCEWithLogitsLoss")
    criterion = nn.BCEWithLogitsLoss()

  print("Model = %s" % str(model_without_ddp))
  print('number of params:', n_parameters)

  total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
  num_training_steps_per_epoch = len(dataset_train) // total_batch_size
  args.min_lr = args.min_lr * total_batch_size / 256
  args.warmup_lr = args.warmup_lr * total_batch_size / 256
  print("LR = %.8f" % args.lr)
  print("Batch size = %d" % total_batch_size)
  print("Update frequent = %d" % args.update_freq)
  print("Number of training examples = %d" % len(dataset_train))
  print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

  num_layers = model_without_ddp.get_num_layers()
  assigner = None
  slow_assigner = None
  fast_assigner = None
  is_dual = getattr(args, 'dual_stream', False) and hasattr(model_without_ddp, 'slow') and hasattr(model_without_ddp,
                                                                                                   'fast')

  if is_dual and hasattr(model_without_ddp.slow, 'get_num_layers') and hasattr(model_without_ddp.fast,
                                                                               'get_num_layers') and args.layer_decay < 1.0:
    # v3 A1: separate assigners for slow (12) and fast (32) for correct layer-wise decay
    num_layers_slow = model_without_ddp.slow.get_num_layers()
    num_layers_fast = model_without_ddp.fast.get_num_layers()
    slow_assigner = LayerDecayValueAssigner(
      list(args.layer_decay ** (num_layers_slow + 1 - i) for i in range(num_layers_slow + 2)))
    fast_assigner = LayerDecayValueAssigner(
      list(args.layer_decay ** (num_layers_fast + 1 - i) for i in range(num_layers_fast + 2)))
    print("Two-stream v3 layer decay: slow_assigner (%d layers), fast_assigner (%d layers)" % (
    num_layers_slow, num_layers_fast))
    print("Slow assigned values = %s" % str(slow_assigner.values))
    print("Fast assigned values = %s" % str(fast_assigner.values))
  elif args.layer_decay < 1.0:
    if is_dual and hasattr(model_without_ddp.fast, 'get_num_layers'):
      num_layers = max(num_layers, model_without_ddp.fast.get_num_layers())
      print("Two-stream layer decay: num_layers = max(slow, fast) = %d" % num_layers)
    assigner = LayerDecayValueAssigner(
      list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    print("Assigned values = %s" % str(assigner.values))

  skip_weight_decay_list = model.no_weight_decay()
  print("Skip weight decay list: ", skip_weight_decay_list)

  if args.enable_deepspeed:
    loss_scaler = None
    if slow_assigner is not None and fast_assigner is not None:
      optimizer_params = get_parameter_groups(
        model, args.weight_decay, skip_weight_decay_list,
        get_num_layer=None, get_layer_scale=None,
        slow_assigner=slow_assigner, fast_assigner=fast_assigner,
        lr_scale_slow_backbone=getattr(args, 'lr_scale_slow_backbone', 1.0),
        lr_scale_fast_backbone=getattr(args, 'lr_scale_fast_backbone', 0.1),
        lr_scale_heads=getattr(args, 'lr_scale_heads', 5.0))
    else:
      optimizer_params = get_parameter_groups(
        model, args.weight_decay, skip_weight_decay_list,
        assigner.get_layer_id if assigner is not None else None,
        assigner.get_scale if assigner is not None else None)
    model, optimizer, _, _ = ds_init(
      args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
    )

    print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
    assert model.gradient_accumulation_steps() == args.update_freq
  else:
    if args.distributed:
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
      model_without_ddp = model.module

    if slow_assigner is not None and fast_assigner is not None:
      optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=None, get_layer_scale=None,
        slow_assigner=slow_assigner, fast_assigner=fast_assigner,
        lr_scale_slow_backbone=getattr(args, 'lr_scale_slow_backbone', 1.0),
        lr_scale_fast_backbone=getattr(args, 'lr_scale_fast_backbone', 0.1),
        lr_scale_heads=getattr(args, 'lr_scale_heads', 5.0))
    else:
      optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = None

  print("Use step level LR scheduler!")
  lr_schedule_values = utils.cosine_scheduler(
    args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
    warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
  )
  if args.weight_decay_end is None:
    args.weight_decay_end = args.weight_decay
  wd_schedule_values = utils.cosine_scheduler(
    args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
  print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

  utils.auto_load_model(
    args=args, model=model, model_without_ddp=model_without_ddp,
    optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

  if args.eval:
    validation_one_epoch(data_loader_val, model, device, args.output_dir, args.start_epoch, log_writer)
    exit(0)

  print(f"Start training for {args.epochs} epochs")
  start_time = time.time()
  for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
      data_loader_train.sampler.set_epoch(epoch)
    if log_writer is not None:
      log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

    # Two-stream v3: Stage-1 freeze slow + low p_drop_fast; Stage-2 unfreeze + optional higher p_drop_fast
    if getattr(args, 'dual_stream', False):
      m = model_without_ddp
      if hasattr(m, 'slow') and getattr(args, 'freeze_slow_epochs', 0) > 0:
        freeze_slow = epoch < args.freeze_slow_epochs
        for p in m.slow.parameters():
          p.requires_grad = not freeze_slow
        if utils.is_main_process():
          print("[TwoStream v3] Epoch %d: slow backbone %s" % (epoch, "frozen" if freeze_slow else "unfrozen"))
      if hasattr(m, 'p_drop_fast') and getattr(args, 'p_drop_fast_stage2', None) is not None:
        m.p_drop_fast = float(
          args.p_drop_fast_stage2 if epoch >= getattr(args, 'freeze_slow_epochs', 0) else args.p_drop_fast)
        if utils.is_main_process() and epoch == 0:
          print("[TwoStream v3] p_drop_fast: Stage-1=%.2f, Stage-2 (from epoch %d)=%.2f" % (
          args.p_drop_fast, getattr(args, 'freeze_slow_epochs', 0), args.p_drop_fast_stage2))

    train_stats = train_one_epoch(
      model, data_loader_train, optimizer,
      device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
      log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
      lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
      num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, criterion=criterion,
      args=args
    )
    # 1. save ckpt
    if args.output_dir and args.save_ckpt:
      if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
        utils.save_model(
          args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
          loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
    # 2. log
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters}

    if args.output_dir and utils.is_main_process():
      if log_writer is not None:
        log_writer.flush()
      with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")
    # 3. eval
    if data_loader_val is not None and (
            (epoch + 1) % args.val_freq == 0 or epoch + 1 == args.epochs):
      validation_one_epoch(data_loader_val, model, device, args.output_dir, epoch, log_writer)
  torch.distributed.barrier()
  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
  opts, ds_init = get_args()
  if opts.output_dir:
    Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
  main(opts, ds_init)