
# -*- coding: utf-8 -*-
"""
VideoMamba backbone adapter for InternVideo Downstream Spatial-Temporal Action Localization.

Key goals:
- Load VideoMamba pretrained checkpoint (e.g. videomamba_m16_k400_mask_ft_f32_res224.pth).
- Provide spatiotemporal feature map output: [B, C, T, H, W] for ROI-level action localization.
- Keep the original VideoMamba architecture (Mamba blocks + spatial/temporal positional embeddings),
  but remove the classification head to use it as a backbone.

This file is adapted from the user's provided VideoMamba implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Any

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception as e:  # pragma: no cover
    Mamba = None
    _MAMBA_IMPORT_ERR = e

try:
    # Optional fused norm kernels (may be unavailable depending on installation)
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except Exception:  # pragma: no cover
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
import torch.nn.functional as F
import math

_logger = logging.getLogger(__name__)


class Block(nn.Module):
    """
    Mamba block wrapper with (Add -> Norm -> Mixer) structure (prenorm-like with optional fused add+norm).
    """
    def __init__(
        self,
        dim: int,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails; please install mamba_ssm with triton kernels."
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), "Only LayerNorm and RMSNorm are supported."

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        use_checkpoint: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns (hidden_states, residual) to support fused add+norm.
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model: int,
    ssm_cfg: Optional[Dict[str, Any]] = None,
    norm_epsilon: float = 1e-5,
    drop_path: float = 0.0,
    rms_norm: bool = True,
    residual_in_fp32: bool = True,
    fused_add_norm: bool = True,
    layer_idx: Optional[int] = None,
    bimamba: bool = True,
    device=None,
    dtype=None,
) -> Block:
    if Mamba is None:  # pragma: no cover
        raise ImportError(f"Failed to import Mamba from mamba_ssm: {_MAMBA_IMPORT_ERR}")

    factory_kwargs = {"device": device, "dtype": dtype}
    ssm_cfg = {} if ssm_cfg is None else ssm_cfg
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module: nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
):
    """
    Weight init used by VideoMamba.
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """Image/Video to Patch Embedding via Conv3D."""
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = kernel_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)  # [B, C, T, H, W]


@dataclass
class VideoMambaCfg:
    img_size: int = 224
    patch_size: int = 16
    depth: int = 32
    embed_dim: int = 576
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    ssm_cfg: Optional[Dict[str, Any]] = None
    norm_epsilon: float = 1e-5
    fused_add_norm: bool = True
    rms_norm: bool = True
    residual_in_fp32: bool = True
    bimamba: bool = True
    kernel_size: int = 1
    num_frames: int = 32
    use_checkpoint: bool = False
    checkpoint_num: int = 0


class VideoMambaBackbone(nn.Module):
    """
    VideoMamba backbone that outputs spatiotemporal feature map [B, C, T, H, W].
    """
    def __init__(self, cfg: VideoMambaCfg):
        super().__init__()
        self.cfg = cfg

        self.residual_in_fp32 = cfg.residual_in_fp32
        self.fused_add_norm = cfg.fused_add_norm
        self.use_checkpoint = cfg.use_checkpoint
        self.checkpoint_num = cfg.checkpoint_num

        self.embed_dim = cfg.embed_dim
        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            kernel_size=cfg.kernel_size,
            in_chans=3,
            embed_dim=cfg.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Spatial and temporal positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(
            torch.zeros(1, cfg.num_frames // cfg.kernel_size, cfg.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=cfg.drop_rate)

        # Mamba layers
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.depth)]
        inter_dpr = [0.0] + dpr

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0.0 else nn.Identity()
        self.layers = nn.ModuleList(
            [
                create_block(
                    cfg.embed_dim,
                    ssm_cfg=cfg.ssm_cfg,
                    norm_epsilon=cfg.norm_epsilon,
                    rms_norm=cfg.rms_norm,
                    residual_in_fp32=cfg.residual_in_fp32,
                    fused_add_norm=cfg.fused_add_norm,
                    layer_idx=i,
                    bimamba=cfg.bimamba,
                    drop_path=inter_dpr[i],
                )
                for i in range(cfg.depth)
            ]
        )
        self.norm_f = (nn.LayerNorm if not cfg.rms_norm else RMSNorm)(cfg.embed_dim, eps=cfg.norm_epsilon)

        # init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(_init_weights, n_layer=cfg.depth))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    def get_num_layers(self):
        return len(self.layers)

    def forward_features_tokens(
        self, x: Tensor, inference_params=None
    ) -> Tuple[Tensor, Tuple[int, int, int]]:
        """
        Returns:
            hidden_states: [B, 1 + T*H*W, C]
            thw: (T, H, W) after patch embedding
        """
        x = self.patch_embed(x)  # [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # per-frame spatial tokens: (B*T, HW, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # add cls token per-frame then add spatial pos embedding
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # --- FIX START: 动态插值 Positional Embedding ---
        if x.shape[1] != self.pos_embed.shape[1]:
            # 1. 分离 CLS token 和 Patch tokens
            cls_pos = self.pos_embed[:, :1, :]
            patch_pos = self.pos_embed[:, 1:, :]

            # 2. 计算原始预训练的网格大小 (假设是正方形, e.g., 14x14)
            orig_num_patches = patch_pos.shape[1]
            orig_size = int(math.sqrt(orig_num_patches))

            # 3. 调整形状以进行插值: [1, N, C] -> [1, C, H, W]
            patch_pos = patch_pos.transpose(1, 2).reshape(1, self.embed_dim, orig_size, orig_size)

            # 4. 插值到当前输入的网格大小 (H, W 来自函数开头的 x.shape)
            patch_pos = F.interpolate(patch_pos, size=(H, W), mode='bicubic', align_corners=False)

            # 5. 恢复形状: [1, C, H_new, W_new] -> [1, N_new, C]
            patch_pos = patch_pos.flatten(2).transpose(1, 2)

            # 6. 拼接回 CLS token 并相加
            pos_embed = torch.cat((cls_pos, patch_pos), dim=1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        # --- FIX END ---

        # Build a *single* video-level cls token ...

        # Build a *single* video-level cls token (take cls from first temporal slice),
        # then add temporal embeddings on patch tokens.
        video_cls = x[0:B, :1, :]  # [B, 1, C]  (first time slice)
        x = x[:, 1:]               # [B*T, HW, C]

        # (B*T, HW, C) -> (B*HW, T, C)
        x = rearrange(x, '(b t) n c -> (b n) t c', b=B, t=T)
        x = x + self.temporal_pos_embedding[:, :T, :]
        # (B*HW, T, C) -> (B, T*HW, C)
        x = rearrange(x, '(b n) t c -> b (t n) c', b=B, t=T)

        # concat back video cls token -> [B, 1 + T*HW, C]
        x = torch.cat((video_cls, x), dim=1)
        x = self.pos_drop(x)

        # Mamba blocks
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params, use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)

        # final norm
        if not self.fused_add_norm:
            residual = hidden_states if residual is None else residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states, (T, H, W)

    def forward_feature_map(self, x: Tensor, inference_params=None) -> Tensor:
        """
        Returns spatiotemporal feature map:
            feat: [B, C, T, H, W]
        """
        hidden_states, (T, H, W) = self.forward_features_tokens(x, inference_params=inference_params)
        # remove cls token
        patch_tokens = hidden_states[:, 1:, :]  # [B, T*H*W, C]
        B, L, C = patch_tokens.shape
        assert L == T * H * W, f"Token length mismatch: L={L}, T*H*W={T*H*W}"
        feat = patch_tokens.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return feat

    def forward(self, x: Tensor, inference_params=None) -> Tensor:
        # Backbone only: return feature map.
        return self.forward_feature_map(x, inference_params=inference_params)

    def load_pretrained(self, ckpt_path: str, prefix: str = "", strict: bool = False) -> Dict[str, Any]:
        """
        Load a VideoMamba pretrained checkpoint. This function is intentionally robust:
        - supports checkpoints with keys: 'model', 'state_dict', or a plain state dict.
        - strips common prefixes (e.g., 'module.').
        - ignores classifier head weights if present.

        Returns:
            msg: the return dict from load_state_dict.
        """
        _logger.info(f"[VideoMambaBackbone] Loading pretrained weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt if isinstance(ckpt, dict) else {}

        new_state = {}
        for k, v in state_dict.items():
            if prefix and k.startswith(prefix):
                k2 = k[len(prefix):]
            else:
                k2 = k
            if k2.startswith("module."):
                k2 = k2[len("module."):]
            # drop classification head if exists
            if k2.startswith("head.") or k2 in {"head.weight", "head.bias"}:
                continue
            new_state[k2] = v

        msg = self.load_state_dict(new_state, strict=strict)
        _logger.info(f"[VideoMambaBackbone] load_state_dict: {msg}")
        return {"missing_keys": msg.missing_keys, "unexpected_keys": msg.unexpected_keys}


def build_videomamba_middle_backbone(
    num_frames: int = 32,
    img_size: int = 224,
    patch_size: int = 16,
    drop_path_rate: float = 0.1,
    use_checkpoint: bool = False,
    checkpoint_num: int = 0,
) -> VideoMambaBackbone:
    """
    Build VideoMamba-M (m16) backbone: embed_dim=576, depth=32.
    """
    cfg = VideoMambaCfg(
        img_size=img_size,
        patch_size=patch_size,
        depth=32,
        embed_dim=576,
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        ssm_cfg=None,
        fused_add_norm=True,
        rms_norm=True,
        residual_in_fp32=True,
        bimamba=True,
        kernel_size=1,
        num_frames=num_frames,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
    )
    return VideoMambaBackbone(cfg)
