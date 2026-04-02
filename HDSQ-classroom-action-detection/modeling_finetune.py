from functools import partial
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from dataclasses import dataclass
from alphaction.modeling.poolers import make_3d_pooler
import math
import pdb

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class HR2O_NL(nn.Module):
    def __init__(self, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(HR2O_NL, self).__init__()

        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        self.conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            1 if mlp_1x1 else kernel_size,
            padding=0 if mlp_1x1 else padding,
            bias=False
        )
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)

        x = x + virt_feats
        return x


class ACARHead(nn.Module):
    def __init__(self, num_classes=60, dropout=0., bias=False,
                 reduce_dim=1024, hidden_dim=512, downsample='max2x2', depth=2,
                 kernel_size=3, mlp_1x1=False):
        super(ACARHead, self).__init__()

        # actor-context feature encoder
        self.conv1 = nn.Conv2d(reduce_dim * 2, hidden_dim, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, bias=False)

        # down-sampling before HR2O
        assert downsample in ['none', 'max2x2']
        if downsample == 'none':
            self.downsample = nn.Identity()
        elif downsample == 'max2x2':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # high-order relation reasoning operator (HR2O_NL)
        layers = []
        for _ in range(depth):
            layers.append(HR2O_NL(hidden_dim, kernel_size, mlp_1x1))
        self.hr2o = nn.Sequential(*layers)

        # classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(reduce_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None

    # data: features, rois, num_rois, roi_ids, sizes_before_padding
    # returns: outputs
    def forward(self, roi_feats, roi_ids, img_features):
        """
        roi_feats: [num_rois, emb_dim]
        roi_ids: [num_rois]
        img_features: [bs,emb_dim,t,h,w]
        """
        high_order_feats = []
        cur_roi_id = 0
        # pdb.set_trace()
        for idx in range(img_features.shape[0]):  # iterate over mini-batch
            n_rois = roi_ids[idx]
            if n_rois == 0:
                continue

            # eff_h, eff_w = math.ceil(h * sizes_before_padding[idx][1]), math.ceil(w * sizes_before_padding[idx][0])
            bg_feats = img_features[idx]
            bg_feats = bg_feats.unsqueeze(0).repeat((n_rois, 1, 1, 1))
            actor_feats = roi_feats[cur_roi_id:cur_roi_id+roi_ids[idx]]
            cur_roi_id += n_rois
            tiled_actor_feats = actor_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats)
            interact_feats = torch.cat([bg_feats, tiled_actor_feats], dim=1)

            interact_feats = self.conv1(interact_feats)
            interact_feats = nn.functional.relu(interact_feats)
            interact_feats = self.conv2(interact_feats)
            interact_feats = nn.functional.relu(interact_feats)

            interact_feats = self.downsample(interact_feats)

            interact_feats = self.hr2o(interact_feats)
            interact_feats = self.gap(interact_feats)
            high_order_feats.append(interact_feats)

        high_order_feats = torch.cat(high_order_feats, dim=0).view(np.sum(np.array(roi_ids)), -1)

        outputs = self.fc1(roi_feats)
        outputs = nn.functional.relu(outputs)
        outputs = torch.cat([outputs, high_order_feats], dim=1)

        if self.dp is not None:
            outputs = self.dp(outputs)
        outputs = self.fc2(outputs)

        return outputs


# ============================================================
# CoDA-Q Head: Correlation-Discriminative Actor Query Head
# ============================================================
class CoDATransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, need_weights: bool = False):
        q2, _ = self.self_attn(query, query, query, need_weights=False)
        query = self.norm1(query + self.dropout(q2))
        q2, attn = self.cross_attn(query, memory, memory, need_weights=need_weights, average_attn_weights=True)
        query = self.norm2(query + self.dropout(q2))
        query = self.norm3(query + self.dropout(self.linear2(self.dropout(self.act(self.linear1(query))))))
        return query, attn


class CoDAQueryHead(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, num_corr_queries: int = 4, num_disc_queries: int = 4,
                 decoder_depth: int = 1, decoder_num_heads: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1,
                 pooler_resolution: int = 7, temporal_samples: int = 4, use_motion_tokens: bool = False, gate_disc: bool = True,
                 lambda_cd: float = 0.05, lambda_div: float = 0.05, reg_warmup_steps: int = 0, reg_on_local_only: bool = True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.num_corr_queries = int(num_corr_queries)
        self.num_disc_queries = int(num_disc_queries)
        self.num_queries = self.num_corr_queries + self.num_disc_queries
        self.pooler_resolution = int(pooler_resolution)
        self.num_spatial_tokens = self.pooler_resolution * self.pooler_resolution
        self.temporal_samples = int(max(1, temporal_samples))
        self.use_motion_tokens = bool(use_motion_tokens)
        self.gate_disc = bool(gate_disc)
        self.lambda_cd = float(lambda_cd)
        self.lambda_div = float(lambda_div)
        self.reg_warmup_steps = int(max(0, reg_warmup_steps))
        self.reg_on_local_only = bool(reg_on_local_only)
        self._global_step: int = 0
        self.corr_query_embed = nn.Parameter(torch.zeros(self.num_corr_queries, embed_dim))
        self.disc_query_embed = nn.Parameter(torch.zeros(self.num_disc_queries, embed_dim))
        self.actor_proj = nn.Linear(embed_dim, embed_dim)
        self.local_spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_spatial_tokens, embed_dim))
        self.ctx_spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_spatial_tokens, embed_dim))
        self.local_time_pos = nn.Parameter(torch.zeros(1, self.temporal_samples, 1, embed_dim))
        self.ctx_time_pos = nn.Parameter(torch.zeros(1, self.temporal_samples, 1, embed_dim))
        if self.use_motion_tokens:
            motion_len = max(1, self.temporal_samples - 1)
            self.motion_spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_spatial_tokens, embed_dim))
            self.motion_time_pos = nn.Parameter(torch.zeros(1, motion_len, 1, embed_dim))
        else:
            self.motion_spatial_pos = None
            self.motion_time_pos = None
        self.layers = nn.ModuleList([
            CoDATransformerDecoderLayer(embed_dim=embed_dim, num_heads=decoder_num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu")
            for _ in range(int(max(1, decoder_depth)))
        ])
        self.gate_proj = nn.Linear(embed_dim, embed_dim) if (self.gate_disc and self.num_disc_queries > 0) else None
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._debug_enabled = False
        self._debug_max_rois = 4
        self._debug_state: Optional[Dict[str, Any]] = None
        trunc_normal_(self.corr_query_embed, std=0.02)
        if self.num_disc_queries > 0:
            trunc_normal_(self.disc_query_embed, std=0.02)
        trunc_normal_(self.local_spatial_pos, std=0.02)
        trunc_normal_(self.ctx_spatial_pos, std=0.02)
        trunc_normal_(self.local_time_pos, std=0.02)
        trunc_normal_(self.ctx_time_pos, std=0.02)
        if self.motion_spatial_pos is not None:
            trunc_normal_(self.motion_spatial_pos, std=0.02)
        if self.motion_time_pos is not None:
            trunc_normal_(self.motion_time_pos, std=0.02)
        if self.gate_proj is not None:
            trunc_normal_(self.gate_proj.weight, std=0.02)
            nn.init.constant_(self.gate_proj.bias, 0.0)
        trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.constant_(self.classifier.bias, 0.0)

    def set_debug(self, enable: bool, max_rois: int = 4) -> None:
        self._debug_enabled = bool(enable)
        self._debug_max_rois = int(max_rois)

    def pop_debug_state(self) -> Optional[Dict[str, Any]]:
        state = self._debug_state
        self._debug_state = None
        return state

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    @staticmethod
    def _offdiag_mean_square(mat: torch.Tensor) -> torch.Tensor:
        R, K, _ = mat.shape
        if K <= 1:
            return mat.new_tensor(0.0)
        eye = torch.eye(K, device=mat.device, dtype=torch.bool).unsqueeze(0)
        off = mat.masked_select(~eye).view(R, -1)
        return (off ** 2).mean()

    def _match_time_pos(self, time_pos: torch.Tensor, S: int) -> torch.Tensor:
        S0 = time_pos.shape[1]
        if S == S0:
            return time_pos
        if S < S0:
            return time_pos[:, :S, :, :]
        pad = time_pos[:, -1:, :, :].expand(1, S - S0, 1, self.embed_dim)
        return torch.cat([time_pos, pad], dim=1)

    def _reg_warmup_scale(self) -> float:
        if self.reg_warmup_steps <= 0:
            return 1.0
        return float(min(1.0, max(0.0, self._global_step / float(self.reg_warmup_steps))))

    def forward(self, roi_feats: torch.Tensor, roi_maps: torch.Tensor, roi_ctx_maps: Optional[torch.Tensor] = None,
                roi_ids: Optional[List[int]] = None, img_features: Optional[torch.Tensor] = None):
        R = roi_feats.shape[0]
        if R == 0:
            logits = roi_feats.new_zeros((0, self.num_classes))
            return logits, {"loss_codaq_cd": roi_feats.new_tensor(0.0), "loss_codaq_div": roi_feats.new_tensor(0.0),
                           "loss_codaq_total": roi_feats.new_tensor(0.0), "lambda_cd_eff": roi_feats.new_tensor(0.0), "lambda_div_eff": roi_feats.new_tensor(0.0)}
        _, C, S, P, P2 = roi_maps.shape
        assert C == self.embed_dim and P == self.pooler_resolution and P2 == self.pooler_resolution
        local = roi_maps.permute(0, 2, 3, 4, 1).reshape(R, S, self.num_spatial_tokens, C)
        local = local + self.local_spatial_pos + self._match_time_pos(self.local_time_pos, S)
        if self.use_motion_tokens and S >= 2:
            motion = roi_maps[:, :, 1:, :, :] - roi_maps[:, :, :-1, :, :]
            Sm = motion.shape[2]
            motion = motion.permute(0, 2, 3, 4, 1).reshape(R, Sm, self.num_spatial_tokens, C)
            motion = motion + self.motion_spatial_pos + self._match_time_pos(self.motion_time_pos, Sm)
            motion_tokens = motion.reshape(R, Sm * self.num_spatial_tokens, C)
        else:
            Sm = 0
            motion_tokens = None
        local_tokens = local.reshape(R, S * self.num_spatial_tokens, C)
        num_local_tokens = local_tokens.shape[1]
        if roi_ctx_maps is not None:
            ctx = roi_ctx_maps.permute(0, 2, 3, 4, 1).reshape(R, S, self.num_spatial_tokens, C)
            ctx = ctx + self.ctx_spatial_pos + self._match_time_pos(self.ctx_time_pos, S)
            ctx_tokens = ctx.reshape(R, S * self.num_spatial_tokens, C)
            num_ctx_tokens = ctx_tokens.shape[1]
        else:
            ctx_tokens = None
            num_ctx_tokens = 0
        memory = torch.cat([local_tokens, motion_tokens, ctx_tokens], dim=1) if (motion_tokens is not None and ctx_tokens is not None) else (
            torch.cat([local_tokens, motion_tokens], dim=1) if motion_tokens is not None else (
            torch.cat([local_tokens, ctx_tokens], dim=1) if ctx_tokens is not None else local_tokens))
        num_motion_tokens = Sm * self.num_spatial_tokens if Sm > 0 else 0
        corr_q = self.corr_query_embed.unsqueeze(0).expand(R, -1, -1)
        query = torch.cat([corr_q, self.disc_query_embed.unsqueeze(0).expand(R, -1, -1)], dim=1) if self.num_disc_queries > 0 else corr_q
        query = query + self.actor_proj(roi_feats).unsqueeze(1)
        attn_last = None
        for li, layer in enumerate(self.layers):
            need_weights = (li == len(self.layers) - 1)
            query, attn = layer(query, memory, need_weights=need_weights)
            if need_weights:
                attn_last = attn
        corr_vec = query[:, :self.num_corr_queries, :].mean(dim=1)
        disc_vec = query[:, self.num_corr_queries:, :].mean(dim=1) if self.num_disc_queries > 0 else None
        fused = corr_vec
        if disc_vec is not None:
            fused = fused + (torch.sigmoid(self.gate_proj(corr_vec)) * disc_vec) if self.gate_proj is not None else fused + disc_vec
        logits = self.classifier(fused)
        if attn_last is None or self.num_disc_queries <= 0:
            loss_cd = roi_feats.new_tensor(0.0)
            loss_div = roi_feats.new_tensor(0.0)
        else:
            A_corr = attn_last[:, :self.num_corr_queries, :]
            A_disc = attn_last[:, self.num_corr_queries:, :]
            N_reg = int(num_local_tokens + num_motion_tokens) if self.reg_on_local_only else attn_last.shape[2]
            A_corr_reg = A_corr[:, :, :N_reg]
            A_disc_reg = A_disc[:, :, :N_reg]
            loss_cd = torch.bmm(A_disc_reg, A_corr_reg.transpose(1, 2)).pow(2).mean()
            A_disc_norm = F.normalize(A_disc_reg, p=2, dim=-1)
            loss_div = self._offdiag_mean_square(torch.bmm(A_disc_norm, A_disc_norm.transpose(1, 2)))
            if self._debug_enabled:
                max_rois = min(self._debug_max_rois, R)
                self._debug_state = {"attn_corr": A_corr[:max_rois].detach().cpu(), "attn_disc": A_disc[:max_rois].detach().cpu(),
                    "pooler_resolution": self.pooler_resolution, "num_spatial_tokens": self.num_spatial_tokens, "num_time": S,
                    "num_local_tokens": num_local_tokens, "num_motion_tokens": num_motion_tokens, "num_ctx_tokens": num_ctx_tokens,
                    "use_motion_tokens": self.use_motion_tokens, "reg_on_local_only": self.reg_on_local_only, "loss_cd": float(loss_cd.detach().cpu().item()),
                    "loss_div": float(loss_div.detach().cpu().item()), "global_step": self._global_step}
        warm_scale = self._reg_warmup_scale()
        lambda_cd_eff = self.lambda_cd * warm_scale
        lambda_div_eff = self.lambda_div * warm_scale
        loss_total = lambda_cd_eff * loss_cd + lambda_div_eff * loss_div
        aux_losses = {"loss_codaq_cd": loss_cd, "loss_codaq_div": loss_div, "loss_codaq_total": loss_total,
                     "lambda_cd_eff": roi_feats.new_tensor(lambda_cd_eff), "lambda_div_eff": roi_feats.new_tensor(lambda_div_eff)}
        return logits, aux_losses


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
                    num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        # B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


@dataclass
class ROIPoolingCfg:
    POOLER_RESOLUTION: int = 7
    POOLER_SCALE: float = 0.0625
    POOLER_SAMPLING_RATIO: int = 0
    POOLER_TYPE: str = 'align3d'
    MEAN_BEFORE_POOLER: bool = True


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=80,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 head_type='linear',
                 use_mean_pooling=True,
                 codaq_num_corr_queries: int = 4,
                 codaq_num_disc_queries: int = 4,
                 codaq_decoder_depth: int = 1,
                 codaq_decoder_num_heads: int = 4,
                 codaq_dim_feedforward: int = 1024,
                 codaq_dropout: float = 0.1,
                 codaq_ctx_ext: Tuple[float, float] = (0.2, 0.1),
                 codaq_lambda_cd: float = 0.05,
                 codaq_lambda_div: float = 0.05,
                 codaq_temporal_samples: int = 4,
                 codaq_use_motion_tokens: bool = False,
                 codaq_gate_disc: bool = True,
                 codaq_reg_warmup_steps: int = 0,
                 codaq_reg_on_local_only: bool = True,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.grid_size = [img_size//patch_size, img_size//patch_size]
        self.head_type = head_type
        self.codaq_ctx_ext = codaq_ctx_ext if head_type == 'codaq' else (0.2, 0.1)
        self.codaq_temporal_samples = int(codaq_temporal_samples) if head_type == 'codaq' else 4
        self._codaq_debug_enabled: bool = False
        self._codaq_debug_meta: Optional[Dict[str, Any]] = None
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        # self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.norm = norm_layer(embed_dim)  # 这一项是预训练权重中没有的
        self.fc_norm = None
        if head_type == 'acar':
            self.head = ACARHead(num_classes=num_classes, hidden_dim=embed_dim, reduce_dim=embed_dim) if num_classes > 0 else nn.Identity()
            trunc_normal_(self.head.fc2.weight, std=.02)
            self.head.fc2.weight.data.mul_(init_scale)
        elif head_type == 'codaq':
            self.head = CoDAQueryHead(
                num_classes=num_classes, embed_dim=embed_dim,
                num_corr_queries=codaq_num_corr_queries, num_disc_queries=codaq_num_disc_queries,
                decoder_depth=codaq_decoder_depth, decoder_num_heads=codaq_decoder_num_heads,
                dim_feedforward=codaq_dim_feedforward, dropout=codaq_dropout,
                pooler_resolution=ROIPoolingCfg().POOLER_RESOLUTION, temporal_samples=codaq_temporal_samples,
                use_motion_tokens=codaq_use_motion_tokens, gate_disc=codaq_gate_disc,
                lambda_cd=codaq_lambda_cd, lambda_div=codaq_lambda_div,
                reg_warmup_steps=codaq_reg_warmup_steps, reg_on_local_only=codaq_reg_on_local_only,
            ) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # rois setting
        self.head_cfg = ROIPoolingCfg()
        self.pooler = make_3d_pooler(self.head_cfg)
        resolution = self.head_cfg.POOLER_RESOLUTION
        self.max_pooler = nn.MaxPool2d((resolution, resolution))

        self.test_ext = (0.1, 0.05)
        self.proposal_per_clip = 100

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        if self.head_type == 'codaq' and hasattr(self.head, 'corr_query_embed'):
            nwd |= {'head.corr_query_embed', 'head.disc_query_embed', 'head.local_spatial_pos', 'head.ctx_spatial_pos', 'head.local_time_pos', 'head.ctx_time_pos', 'head.motion_spatial_pos', 'head.motion_time_pos'}
        return nwd

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, proposals):
        x = self.patch_embed(x)
        B, width, t, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)

        if self.pos_embed is not None:
            pos_embed = self.pos_embed.reshape(t, -1, width)
            pos_embed = interpolate_pos_embed_online(
                pos_embed, self.grid_size, [h, w], 0).reshape(1, -1, width)
            x = x + pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.reshape(B, t, h, w, -1).permute(0, 4, 1, 2, 3)
        x_mean = x.mean(dim=2, keepdim=False)
        roi_maps_mean = self.pooler(x_mean, proposals)
        roi_feats = self.max_pooler(roi_maps_mean).view(roi_maps_mean.size(0), -1)
        roi_maps = roi_maps_mean
        roi_ctx_maps = None
        if self.head_type == 'codaq':
            S = self.codaq_temporal_samples
            T = x.shape[2]
            indices = torch.linspace(0, T - 1, S).long().to(x.device) if T > S else torch.arange(T, device=x.device)
            maps_list = [self.pooler(x[:, :, int(i), :, :], proposals) for i in indices]
            roi_maps = torch.stack(maps_list, dim=2)
            proposals_ctx = [p.extend(self.codaq_ctx_ext) for p in proposals]
            ctx_list = [self.pooler(x[:, :, int(i), :, :], proposals_ctx) for i in indices]
            roi_ctx_maps = torch.stack(ctx_list, dim=2)
        return roi_feats, x, roi_maps, roi_ctx_maps

    def sample_box(self, boxes):
        proposals = []
        num_proposals = self.proposal_per_clip
        for boxes_per_image in boxes:
            num_boxes = len(boxes_per_image)

            if num_boxes > num_proposals:
                choice_inds = torch.randperm(num_boxes)[:num_proposals]
                proposals_per_image = boxes_per_image[choice_inds]
            else:
                proposals_per_image = boxes_per_image
            proposals_per_image = proposals_per_image.random_aug(0.2, 0.1, 0.1, 0.05)
            proposals.append(proposals_per_image)
        return proposals


    def set_codaq_debug(self, enable: bool, max_rois: int = 4) -> None:
        self._codaq_debug_enabled = bool(enable)
        if self.head_type == 'codaq' and hasattr(self.head, 'set_debug'):
            self.head.set_debug(enable, max_rois=max_rois)

    def set_codaq_step(self, step: int) -> None:
        if self.head_type == 'codaq' and hasattr(self.head, 'set_global_step'):
            self.head.set_global_step(int(step))

    def pop_codaq_debug_state(self) -> Optional[Dict[str, Any]]:
        if self.head_type == 'codaq' and hasattr(self.head, 'pop_debug_state'):
            return self.head.pop_debug_state()
        return None

    def pop_codaq_debug_meta(self) -> Optional[Dict[str, Any]]:
        meta = self._codaq_debug_meta
        self._codaq_debug_meta = None
        return meta

    def _cache_codaq_debug_meta(self, video: torch.Tensor, proposals: List[Any]) -> None:
        if not self._codaq_debug_enabled or self.head_type != 'codaq' or video.ndim != 5:
            return
        max_rois = int(getattr(self.head, '_debug_max_rois', 4))
        boxes_all = [p.bbox for p in proposals if len(p) > 0]
        if not boxes_all:
            return
        boxes_xyxy = torch.cat(boxes_all, dim=0)
        if boxes_xyxy.numel() == 0:
            return
        m = min(max_rois, boxes_xyxy.shape[0])
        tmid = video.shape[2] // 2
        frames = video[:, :, tmid, :, :].detach().cpu()
        img_ids = []
        for bi, p in enumerate(proposals):
            if len(p) == 0:
                continue
            img_ids.append(torch.full((len(p),), bi, device=video.device, dtype=torch.long))
        img_ids = torch.cat(img_ids, dim=0).detach().cpu()
        self._codaq_debug_meta = {"frames": frames[img_ids[:m]], "boxes_xyxy": boxes_xyxy[:m].detach().cpu(), "img_ids": img_ids[:m], "t_index": tmid}

    def forward(self, x, boxes):
        if self.training:
            proposals = self.sample_box(boxes)
        else:
            proposals = [box.extend(self.test_ext) for box in boxes]
        roi_feats, img_features, roi_maps, roi_ctx_maps = self.forward_features(x, proposals)
        if self.head_type == 'acar':
            roi_ids = [len(i) for i in proposals]
            return self.head(roi_feats, roi_ids, img_features)
        if self.head_type == 'codaq':
            if self._codaq_debug_enabled:
                self._cache_codaq_debug_meta(x, proposals)
            roi_ids = [len(i) for i in proposals]
            logits, aux_losses = self.head(roi_feats, roi_maps, roi_ctx_maps, roi_ids=roi_ids, img_features=img_features)
            return (logits, aux_losses) if self.training else logits
        return self.head(roi_feats)




# =============================================================================
# Two-stream model: InternVideo(ViT) slow stream + VideoMamba fast stream
# - Backbone-level gated lateral fusion (GLF): fast -> slow
# - ROI-level Cross-Attention fusion: query=slow ROI tokens, key/value=fast ROI tokens
# - Training only: outputs 3 heads (fused/slow/fast) for multi-head supervision & distillation
# =============================================================================
import logging
from typing import Dict, Any

try:
    # Local adapter that exposes VideoMamba feature map output: [B, C, T, H, W]
    from videomamba import build_videomamba_middle_backbone
except Exception as e:  # pragma: no cover
    build_videomamba_middle_backbone = None
    _VIDEOMAMBA_IMPORT_ERR = e

_logger = logging.getLogger(__name__)


def _get_model_attr(model: nn.Module, name: str, default=None):
    """Safely get attributes from (potential) DDP-wrapped model."""
    if hasattr(model, name):
        return getattr(model, name)
    if hasattr(model, "module") and hasattr(model.module, name):
        return getattr(model.module, name)
    return default


class ResidualAdapterFusion(nn.Module):
    """
    v3 ROI-level fusion (true residual from zero, no output LN to preserve baseline):
    - fast_proj = LN(W(roi_fast_motion))
    - g = sigmoid(MLP([LN(roi_slow), fast_proj])), g in [N,1]
    - roi_fused = roi_slow + gamma * g * fast_proj, with per-channel LayerScale gamma init 1e-5 (CaiT-style).
    Initial: gamma~0 => roi_fused ~ roi_slow; training can grow gamma if fast helps.
    For CoDA-Q token-level fusion: W_map projects fast ROI token maps [R,Cf,S,P,P] -> [R,Cs,S,P,P].
    """
    def __init__(self, c_slow: int, c_fast: int, hidden_ratio: float = 0.25, layer_scale_init: float = 1e-5):
        super().__init__()
        self.W = nn.Linear(c_fast, c_slow, bias=False)
        self.ln_fast = nn.LayerNorm(c_slow)
        self.ln_slow = nn.LayerNorm(c_slow)
        # Token map projection: fast ROI maps (Cf) -> slow dim (Cs) for token-level residual injection
        self.W_map = nn.Linear(c_fast, c_slow, bias=False)
        self.ln_fast_map = nn.LayerNorm(c_slow)  # v2: separate LN for token map projection
        hidden = max(8, int(c_slow * 2 * hidden_ratio))
        self.gate_mlp = nn.Sequential(
            nn.Linear(c_slow * 2, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1, bias=True),
        )
        # per-channel LayerScale (CaiT), init small so fused ≈ slow at start
        self.gamma = nn.Parameter(torch.ones(c_slow) * layer_scale_init)
        # v2: separate gamma for token-level fusion (decoupled from vector gamma)
        self.gamma_map = nn.Parameter(torch.ones(c_slow) * layer_scale_init)

    def forward(
        self,
        roi_slow: torch.Tensor,
        roi_fast_dyn: torch.Tensor,
        drop_fast: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        roi_slow: [N, Cs], roi_fast_dyn: [N, Cf]
        Returns: roi_fused [N, Cs], g [N, 1] for logging
        """
        fast_proj = self.ln_fast(self.W(roi_fast_dyn))
        roi_slow_norm = self.ln_slow(roi_slow)
        g = torch.sigmoid(self.gate_mlp(torch.cat([roi_slow_norm, fast_proj], dim=1)))
        if drop_fast:
            g = torch.zeros_like(g, device=g.device)
        # roi_fused = roi_slow + gamma * g * fast_proj (no LN on output to keep baseline unchanged)
        roi_fused = roi_slow + (self.gamma * (g * fast_proj))
        return roi_fused, g

    def project_token_maps(self, roi_maps_fast: torch.Tensor) -> torch.Tensor:
        """
        Project fast ROI token maps to slow dimension for token-level residual injection.
        roi_maps_fast: [R, Cf, S, P, P] -> out: [R, Cs, S, P, P]
        Uses dedicated ln_fast_map (v2) to avoid sharing LN statistics with vector path.
        """
        R, Cf, S, P, _ = roi_maps_fast.shape
        x = roi_maps_fast.permute(0, 2, 3, 4, 1).reshape(R, S * P * P, Cf)
        x = self.ln_fast_map(self.W_map(x))
        return x.reshape(R, S, P, P, -1).permute(0, 4, 1, 2, 3)


class ROITemporalModule(nn.Module):
    """
    v3 C1: ROI-level temporal aggregation. Input [N, C, T, h, w] -> flatten spatial -> [N, C, T] -> 1D depthwise conv + pool -> [N, C].
    """
    def __init__(self, c: int, temporal_kernel: int = 3):
        super().__init__()
        self.temporal_conv = nn.Conv1d(c, c, temporal_kernel, padding=temporal_kernel // 2, groups=c)
        self.norm = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, T, h, w] (e.g. from 3D ROI pool)
        out: [N, C]
        """
        N, C, T, h, w = x.shape
        x = x.reshape(N, C, T, -1).mean(dim=3)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x


class TwoStreamInternVideoMamba(nn.Module):
    """
    Dual-stream v3 (ROI-level residual + ROI temporal):
    - Slow: baseline path unchanged — forward_backbone_5d -> mean(T) -> ROIAlign -> roi_slow [N,Cs]
    - Fast: VideoMamba -> 3D ROI pool -> roi_fast_seq [N,Cf,T] -> ROITemporalModule -> roi_fast_motion [N,Cf]
    - Fusion: roi_fused = roi_slow + gamma*g*fast_proj (LayerScale gamma, no output LN)
    - 3 heads: fused (main) + slow (eval mAP) + fast (aux + eval mAP)

    Input expectation:
        samples: [B, 3, L, H, W], where L == slow_frames * slow_stride (default 64)
    """
    def __init__(
        self,
        num_classes: int = 80,
        # Slow (InternVideo / ViT) config
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: float = 0.0,
        all_frames: int = 16,
        tubelet_size: int = 2,
        head_type: str = "linear",
        use_mean_pooling: bool = True,
        # Dual-stream sampling
        slow_frames: int = 16,
        slow_stride: int = 4,
        fast_frames: int = 32,
        fast_stride: int = 2,
        slow_jitter: bool = False,
        # VideoMamba
        videomamba_ckpt: str = "",
        videomamba_drop_path: float = 0.1,
        videomamba_use_checkpoint: bool = False,
        videomamba_checkpoint_num: int = 0,
        # Fusion (v2: residual adapter only)
        fusion_hidden_ratio: float = 0.25,
        # Loss weights (used by engine_for_finetuning)
        lambda_slow: float = 0.3,
        lambda_fast: float = 0.3,
        lambda_kd: float = 0.1,
        lambda_cons: float = 0.0,
        kd_temperature: float = 1.0,
        # Stream dropout: v2 只 drop fast adapter 贡献，不 drop slow（保证 fused 至少等于 slow）
        p_drop_fast: float = 0.2,
        p_drop_slow: float = 0.2,  # v2 未使用，保留兼容
        **kwargs,
    ):
        super().__init__()

        if build_videomamba_middle_backbone is None:  # pragma: no cover
            raise ImportError(
                f"Failed to import videomamba.py. Error: {_VIDEOMAMBA_IMPORT_ERR}"
            )

        # ---- store loss hyper-params (read by training engine) ----
        self.lambda_slow = float(lambda_slow)
        self.lambda_fast = float(lambda_fast)
        self.lambda_kd = float(lambda_kd)
        self.lambda_cons = float(lambda_cons)
        self.kd_temperature = float(kd_temperature)
        self.p_drop_fast = float(p_drop_fast)
        self.p_drop_slow = float(p_drop_slow)

        # ---- sampling params ----
        self.slow_frames = int(slow_frames)
        self.slow_stride = int(slow_stride)
        self.fast_frames = int(fast_frames)
        self.fast_stride = int(fast_stride)
        self.slow_jitter = bool(slow_jitter)

        # Sanity checks for the L=64 aligned window rule
        assert self.slow_frames * self.slow_stride == self.fast_frames * self.fast_stride, \
            "Slow/Fast must cover the same temporal window: slow_frames*slow_stride == fast_frames*fast_stride"
        self.frame_span = self.slow_frames * self.slow_stride  # e.g. 64

        # ---- slow stream (InternVideo ViT) ----
        self.slow = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,  # placeholder, we'll override heads below
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            all_frames=self.slow_frames,  # must match input slow_clip length (e.g. 16), not full clip 64
            tubelet_size=tubelet_size,
            head_type="linear",         # keep backbone head simple; we create separate heads
            use_mean_pooling=use_mean_pooling,
        )

        # ---- fast stream (VideoMamba) ----
        self.fast = build_videomamba_middle_backbone(
            num_frames=self.fast_frames,
            img_size=img_size,
            patch_size=patch_size,
            drop_path_rate=videomamba_drop_path,
            use_checkpoint=videomamba_use_checkpoint,
            checkpoint_num=videomamba_checkpoint_num,
        )

        if videomamba_ckpt:
            result = self.fast.load_pretrained(videomamba_ckpt, strict=False)
            missing = result.get("missing_keys", [])
            unexpected = result.get("unexpected_keys", [])
            n_missing, n_unexpected = len(missing), len(unexpected)
            # Backbone load success: missing_keys should be few (e.g. only head/norm_f); if hundreds, ckpt key mismatch
            if n_missing <= 20:
                print(
                    "[TwoStream] VideoMamba pretrained weights loaded **successfully** from: %s "
                    "(missing_keys=%d, unexpected_keys=%d)" % (videomamba_ckpt, n_missing, n_unexpected)
                )
            else:
                print(
                    "[TwoStream] VideoMamba pretrained weights **FAILED** to load from: %s "
                    "(missing_keys=%d, too many - check ckpt path and key names). "
                    "First 5 missing: %s" % (videomamba_ckpt, n_missing, missing[:5])
                )

        c_slow = embed_dim
        c_fast = self.fast.embed_dim

        # ---- ROI-level residual adapter (v3: LayerScale gamma, no output LN) ----
        self.fusion_adapter = ResidualAdapterFusion(
            c_slow=c_slow, c_fast=c_fast, hidden_ratio=fusion_hidden_ratio, layer_scale_init=1e-5
        )

        # ---- v3 C1: ROI-level temporal module for fast stream ----
        self.fast_temporal_module = ROITemporalModule(c_fast, temporal_kernel=3)

        # ---- heads: fused (main) + slow (for eval mAP) + fast (aux + eval mAP) ----
        self.head_type = head_type
        if head_type == "acar":
            self.head_fused = ACARHead(num_classes=num_classes, hidden_dim=c_slow, reduce_dim=c_slow)
            self.head_slow = ACARHead(num_classes=num_classes, hidden_dim=c_slow, reduce_dim=c_slow)
            self.head_fast = ACARHead(num_classes=num_classes, hidden_dim=c_fast, reduce_dim=c_fast)
        else:
            self.head_fused = nn.Linear(c_slow, num_classes)
            self.head_slow = nn.Linear(c_slow, num_classes)
            self.head_fast = nn.Linear(c_fast, num_classes)

        self._init_head()

        # ---- logging summary ----
        _logger.info(
            "[TwoStreamInternVideoMamba v3] frame_span=%d | slow=%d@stride%d | fast=%d@stride%d | "
            "lambda_fast=%.3f | lambda_cons=%.4f | p_drop_fast=%.2f",
            self.frame_span, self.slow_frames, self.slow_stride, self.fast_frames, self.fast_stride,
            self.lambda_fast, self.lambda_cons, self.p_drop_fast,
        )

    def _init_head(self):
        if isinstance(self.head_fused, nn.Linear):
            trunc_normal_(self.head_fused.weight, std=.02)
            nn.init.constant_(self.head_fused.bias, 0)
        if isinstance(self.head_slow, nn.Linear):
            trunc_normal_(self.head_slow.weight, std=.02)
            nn.init.constant_(self.head_slow.bias, 0)
        if isinstance(self.head_fast, nn.Linear):
            trunc_normal_(self.head_fast.weight, std=.02)
            nn.init.constant_(self.head_fast.bias, 0)

    def get_num_layers(self):
        # Keep optimizer layer-decay behavior compatible with existing code.
        return self.slow.get_num_layers()

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        if hasattr(self.slow, "no_weight_decay"):
            nwd |= set(self.slow.no_weight_decay())
        if hasattr(self.fast, "no_weight_decay"):
            nwd |= set(self.fast.no_weight_decay())
        # also include fast/slow fusion params if needed
        return nwd

    def _sample_slow_fast(self, clip64: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        clip64: [B, 3, L, H, W]  L==frame_span
        returns:
            slow_clip: [B, 3, slow_frames, H, W]
            fast_clip: [B, 3, fast_frames, H, W]
        """
        assert clip64.shape[2] == self.frame_span, f"Expect {self.frame_span} frames, got {clip64.shape[2]}"

        tau = self.slow_stride
        alpha = tau // self.fast_stride
        assert tau % alpha == 0, "Need integer fast stride = tau/alpha"

        if self.training and self.slow_jitter:
            slow_start = torch.randint(low=0, high=tau, size=(1,), device=clip64.device).item()
        else:
            slow_start = (tau - 1) // 2
        slow_clip = clip64[:, :, slow_start::tau, :, :]

        fast_stride = self.fast_stride
        fast_start = (fast_stride - 1) // 2
        fast_clip = clip64[:, :, fast_start::fast_stride, :, :]

        # Safety: enforce exact lengths (in case of off-by-one due to start)
        slow_clip = slow_clip[:, :, : self.slow_frames, :, :]
        fast_clip = fast_clip[:, :, : self.fast_frames, :, :]

        return slow_clip, fast_clip

    def _roi_pool_from_2d(self, feat2d: torch.Tensor, proposals) -> torch.Tensor:
        """
        feat2d: [B, C, H, W] -> ROIAlign + maxpool -> [N, C]
        Uses slow.pooler / slow.max_pooler for consistent ROI grid.
        """
        rois = self.slow.pooler(feat2d, proposals)
        return self.slow.max_pooler(rois).view(rois.size(0), -1)

    def _roi_pool_5d(self, feat5d: torch.Tensor, proposals) -> torch.Tensor:
        """
        feat5d: [B, C, T, H, W]. Pooler is 2D (per-frame); loop over T and stack -> [N, C, T, res, res].
        """
        B, C, T, H, W = feat5d.shape
        out_list = []
        for t in range(T):
            feat2d_t = feat5d[:, :, t, :, :]
            rois_t = self.slow.pooler(feat2d_t, proposals)
            out_list.append(rois_t)
        rois_5d = torch.stack(out_list, dim=2)
        return rois_5d

    @staticmethod
    def _fast_motion_map(fast_feat: torch.Tensor) -> torch.Tensor:
        """
        Legacy: fast_feat [B, Cf, T, H, W] -> F_dyn [B, Cf, H, W] = mean_t(|F[t]-F[t-1]|).
        Kept for reference; v3 uses ROI temporal instead.
        """
        diff = fast_feat[:, :, 1:, :, :] - fast_feat[:, :, :-1, :, :]
        return diff.abs().mean(dim=2)

    def forward(self, x: torch.Tensor, boxes):
        """
        v2: Training return dict(logits_fused, logits_fast, aux). Eval return same (no logits_slow).
        """
        if self.training:
            proposals = self.slow.sample_box(boxes)
        else:
            proposals = [box.extend(self.slow.test_ext) for box in boxes]

        slow_clip, fast_clip = self._sample_slow_fast(x)

        # Slow: baseline path — mean(T) then ROIAlign (unchanged)
        slow_feat = self.slow.forward_backbone_5d(slow_clip)  # [B, C_s, T_s, H, W]
        slow_2d = slow_feat.mean(dim=2)
        roi_slow = self._roi_pool_from_2d(slow_2d, proposals)  # [N, Cs]

        # Fast: v3 C1 ROI-level temporal — 3D ROI pool -> [N, Cf, T, h, w] -> ROITemporalModule -> roi_fast_motion [N, Cf]
        fast_feat = self.fast.forward_feature_map(fast_clip)   # [B, C_f, T_f, H, W]
        roi_fast_5d = self._roi_pool_5d(fast_feat, proposals)   # [N, C_f, T, res, res]
        roi_fast_motion = self.fast_temporal_module(roi_fast_5d)  # [N, Cf]

        # Drop only fast adapter contribution (same position)
        drop_fast = False
        if self.training and self.p_drop_fast > 0.0:
            if torch.rand(1, device=x.device) < self.p_drop_fast:
                drop_fast = True

        roi_fused, g = self.fusion_adapter(roi_slow, roi_fast_motion, drop_fast=drop_fast)

        # Heads: fused (main) + slow (eval mAP) + fast (aux + eval mAP)
        if self.head_type == "acar":
            roi_ids = [len(i) for i in proposals]
            img_fused = slow_2d
            img_fast = fast_feat.mean(dim=2)
            z_fused = self.head_fused(roi_fused, roi_ids, img_fused)
            z_slow = self.head_slow(roi_slow, roi_ids, img_fused)
            z_fast = self.head_fast(roi_fast_motion, roi_ids, img_fast)
        else:
            z_fused = self.head_fused(roi_fused)
            z_slow = self.head_slow(roi_slow)
            z_fast = self.head_fast(roi_fast_motion)

        gamma_mean = self.fusion_adapter.gamma.detach().mean()
        if self.training:
            aux = {
                "gate_mean": g.mean().detach(),
                "gamma": gamma_mean.expand(g.size(0), 1).mean() if g.dim() > 0 else gamma_mean,
                "drop_fast": torch.tensor(float(drop_fast), device=x.device),
            }
            return {"logits_fused": z_fused, "logits_slow": z_slow, "logits_fast": z_fast, "aux": aux}
        return {"logits_fused": z_fused, "logits_slow": z_slow, "logits_fast": z_fast}


class TwoStreamCoDAQ(nn.Module):
    """Dual-stream (InternVideo + VideoMamba) + CoDA-Q head with token-level fusion."""
    def __init__(self, num_classes: int = 80, codaq_ctx_ext: Tuple[float, float] = (0.2, 0.1), codaq_temporal_samples: int = 4,
                 codaq_num_corr_queries: int = 4, codaq_num_disc_queries: int = 4, codaq_decoder_depth: int = 1,
                 codaq_decoder_num_heads: int = 4, codaq_dim_feedforward: int = 1024, codaq_dropout: float = 0.1,
                 codaq_lambda_cd: float = 0.05, codaq_lambda_div: float = 0.05, codaq_use_motion_tokens: bool = False,
                 codaq_gate_disc: bool = True, codaq_reg_warmup_steps: int = 0, codaq_reg_on_local_only: bool = True,
                 **kwargs):
        super().__init__()
        # Exclude codaq_* and head_type so inner two-stream uses linear head; we add CoDA-Q head on top
        twostream_kw = {k: v for k, v in kwargs.items() if not k.startswith("codaq_") and k != "head_type"}
        self._twostream = TwoStreamInternVideoMamba(num_classes=num_classes, head_type="linear", **twostream_kw)
        c_slow = self._twostream.slow.embed_dim
        self.codaq_ctx_ext = codaq_ctx_ext
        self.codaq_temporal_samples = int(codaq_temporal_samples)
        self.head_fused = CoDAQueryHead(
            num_classes=num_classes, embed_dim=c_slow,
            num_corr_queries=codaq_num_corr_queries, num_disc_queries=codaq_num_disc_queries,
            decoder_depth=codaq_decoder_depth, decoder_num_heads=codaq_decoder_num_heads,
            dim_feedforward=codaq_dim_feedforward, dropout=codaq_dropout,
            pooler_resolution=ROIPoolingCfg().POOLER_RESOLUTION, temporal_samples=codaq_temporal_samples,
            use_motion_tokens=codaq_use_motion_tokens, gate_disc=codaq_gate_disc,
            lambda_cd=codaq_lambda_cd, lambda_div=codaq_lambda_div,
            reg_warmup_steps=codaq_reg_warmup_steps, reg_on_local_only=codaq_reg_on_local_only,
        )
        self.head_slow = self._twostream.head_slow
        self.head_fast = self._twostream.head_fast
        self._twostream.head_fused = self.head_fused
        self.slow = self._twostream.slow
        self.fast = self._twostream.fast
        self.fusion_adapter = self._twostream.fusion_adapter
        self.fast_temporal_module = self._twostream.fast_temporal_module
        self.patch_embed = self.slow.patch_embed
        self._codaq_debug_enabled = False
        self._codaq_debug_meta: Optional[Dict[str, Any]] = None
        for _k, _v in list(self._twostream.__dict__.items()):
            if _k not in self.__dict__:
                setattr(self, _k, _v)

    def get_num_layers(self):
        return self._twostream.get_num_layers()

    def no_weight_decay(self):
        nwd = self._twostream.no_weight_decay() if hasattr(self._twostream, "no_weight_decay") else set()
        if not isinstance(nwd, set):
            nwd = set(nwd)
        # v2 fix: CoDA-Q embedding/pos params must NOT have weight decay.
        # PyTorch named_parameters() traverses _twostream.* first, so the actual
        # parameter names seen by the optimizer have the _twostream. prefix.
        # We add BOTH prefixed and unprefixed to be safe.
        _codaq_nwd_names = (
            "head_fused.corr_query_embed", "head_fused.disc_query_embed",
            "head_fused.local_spatial_pos", "head_fused.ctx_spatial_pos",
            "head_fused.local_time_pos", "head_fused.ctx_time_pos",
        )
        for name in _codaq_nwd_names:
            nwd.add(name)
            nwd.add("_twostream." + name)
        # fusion adapter learnable scales should also skip weight decay
        for gname in ("fusion_adapter.gamma", "fusion_adapter.gamma_map"):
            nwd.add(gname)
            nwd.add("_twostream." + gname)
        return nwd

    def _roi_pool_from_2d(self, feat2d, proposals):
        return self._twostream._roi_pool_from_2d(feat2d, proposals)

    def _roi_pool_5d(self, feat5d, proposals):
        return self._twostream._roi_pool_5d(feat5d, proposals)

    def _sample_slow_fast(self, clip64):
        return self._twostream._sample_slow_fast(clip64)

    def _build_roi_maps_temporal(self, feat5d, proposals):
        B, C, T, H, W = feat5d.shape
        S = self.codaq_temporal_samples
        indices = list(range(T)) if T <= S else [int(round(i)) for i in torch.linspace(0, T - 1, S).tolist()]
        maps_list = [self.slow.pooler(feat5d[:, :, t, :, :], proposals) for t in indices]
        return torch.stack(maps_list, dim=2)

    def _cache_codaq_debug_meta(self, video: torch.Tensor, proposals: List[Any]) -> None:
        if not self._codaq_debug_enabled or video.ndim != 5:
            return
        max_rois = int(getattr(self.head_fused, "_debug_max_rois", 4))
        boxes_all = [p.bbox for p in proposals if len(p) > 0]
        if not boxes_all:
            return
        boxes_xyxy = torch.cat(boxes_all, dim=0)
        if boxes_xyxy.numel() == 0:
            return
        m = min(max_rois, boxes_xyxy.shape[0])
        tmid = video.shape[2] // 2
        frames = video[:, :, tmid, :, :].detach().cpu()
        img_ids = []
        for bi, p in enumerate(proposals):
            if len(p) == 0:
                continue
            img_ids.append(torch.full((len(p),), bi, dtype=torch.long))
        img_ids = torch.cat(img_ids, dim=0).detach().cpu()
        self._codaq_debug_meta = {"frames": frames[img_ids[:m]], "boxes_xyxy": boxes_xyxy[:m].detach().cpu(), "img_ids": img_ids[:m], "t_index": tmid}

    def set_codaq_debug(self, enable: bool, max_rois: int = 4) -> None:
        self._codaq_debug_enabled = bool(enable)
        if hasattr(self.head_fused, "set_debug"):
            self.head_fused.set_debug(enable, max_rois=max_rois)

    def set_codaq_step(self, step: int) -> None:
        if hasattr(self.head_fused, "set_global_step"):
            self.head_fused.set_global_step(int(step))

    def pop_codaq_debug_state(self) -> Optional[Dict[str, Any]]:
        return self.head_fused.pop_debug_state() if hasattr(self.head_fused, "pop_debug_state") else None

    def pop_codaq_debug_meta(self) -> Optional[Dict[str, Any]]:
        meta = self._codaq_debug_meta
        self._codaq_debug_meta = None
        return meta

    def forward(self, x: torch.Tensor, boxes):
        if self.training:
            proposals = self.slow.sample_box(boxes)
        else:
            proposals = [box.extend(self.slow.test_ext) for box in boxes]
        slow_clip, fast_clip = self._sample_slow_fast(x)
        slow_feat = self.slow.forward_backbone_5d(slow_clip)
        Ts = slow_feat.shape[2]
        fast_feat = self.fast.forward_feature_map(fast_clip)
        Tf = fast_feat.shape[2]
        slow_2d = slow_feat.mean(dim=2)
        roi_slow = self._roi_pool_from_2d(slow_2d, proposals)
        roi_fast_5d = self._roi_pool_5d(fast_feat, proposals)
        roi_fast_motion = self.fast_temporal_module(roi_fast_5d)
        drop_fast = False
        if self.training and getattr(self, "p_drop_fast", 0.0) > 0.0 and torch.rand(1, device=x.device) < self.p_drop_fast:
            drop_fast = True
        roi_fused, g = self.fusion_adapter(roi_slow, roi_fast_motion, drop_fast=drop_fast)
        S = self.codaq_temporal_samples
        idx_slow = list(range(Ts)) if Ts <= S else [int(round(i)) for i in torch.linspace(0, Ts - 1, S, device=x.device).tolist()]
        idx_fast = [min(int(round(i / max(1, Ts - 1) * (Tf - 1))), Tf - 1) for i in idx_slow]
        roi_maps_slow = self._build_roi_maps_temporal(slow_feat, proposals)
        maps_fast_list = [self.slow.pooler(fast_feat[:, :, t, :, :], proposals) for t in idx_fast]
        roi_maps_fast = torch.stack(maps_fast_list, dim=2)
        proposals_ctx = [p.extend(self.codaq_ctx_ext) for p in proposals]
        roi_ctx_maps_slow = self._build_roi_maps_temporal(slow_feat, proposals_ctx)
        roi_maps_fast_proj = self.fusion_adapter.project_token_maps(roi_maps_fast)

        # v2 Fix B: Only inject fast into LOCAL tokens; context stays pure slow
        #   to preserve relational/spatial info for interaction-type classes.
        # v2 Fix C: Use dedicated gamma_map (decoupled from vector gamma)
        #   with 1/sqrt(S*P*P) normalization to stabilize token-level gradients.
        P = roi_maps_slow.shape[-1]  # pooler_resolution
        token_norm_scale = 1.0 / math.sqrt(S * P * P)
        gamma_map = self.fusion_adapter.gamma_map.view(1, -1, 1, 1, 1)
        g_bc = g.view(-1, 1, 1, 1, 1)
        roi_maps_fused = roi_maps_slow + token_norm_scale * gamma_map * g_bc * roi_maps_fast_proj
        # Context tokens: pure slow (no fast injection to avoid motion noise pollution)
        roi_ctx_maps_fused = roi_ctx_maps_slow

        if self._codaq_debug_enabled:
            self._cache_codaq_debug_meta(x, proposals)
        logits_fused, aux_losses = self.head_fused(roi_feats=roi_fused, roi_maps=roi_maps_fused, roi_ctx_maps=roi_ctx_maps_fused)
        logits_slow = self.head_slow(roi_slow)
        logits_fast = self.head_fast(roi_fast_motion)
        aux = dict(aux_losses)
        aux["gate_mean"] = g.mean().detach()
        aux["gamma"] = self.fusion_adapter.gamma.detach().mean()
        aux["gamma_map"] = self.fusion_adapter.gamma_map.detach().mean()
        aux["token_norm_scale"] = torch.tensor(token_norm_scale, device=x.device)
        aux["drop_fast"] = torch.tensor(float(drop_fast), device=x.device)
        if self.training:
            return {"logits_fused": logits_fused, "logits_slow": logits_slow, "logits_fast": logits_fast, "aux": aux}
        return {"logits_fused": logits_fused, "logits_slow": logits_slow, "logits_fast": logits_fast}


# Monkey-patch a minimal backbone feature extractor into VisionTransformer without breaking existing forward().
def _vit_forward_backbone_5d(self: "VisionTransformer", x: torch.Tensor) -> torch.Tensor:
    x = self.patch_embed(x)
    B, width, t, h, w = x.size()
    x = x.flatten(2).transpose(1, 2)

    if self.pos_embed is not None:
        pos_embed = self.pos_embed.reshape(t, -1, width)
        pos_embed = interpolate_pos_embed_online(pos_embed, self.grid_size, [h, w], 0).reshape(1, -1, width)
        x = x + pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)
    feat5d = x.reshape(B, t, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
    return feat5d


# Attach only if not already present (safe for repeated imports)
if not hasattr(VisionTransformer, "forward_backbone_5d"):
    VisionTransformer.forward_backbone_5d = _vit_forward_backbone_5d


@register_model
def twostream_vit_base_patch16_224(pretrained=False, **kwargs):
    """
    Two-stream default: InternVideo ViT-Base slow + VideoMamba-M fast.
    """
    model = TwoStreamInternVideoMamba(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def twostream_codaq_vit_base_patch16_224(pretrained=False, **kwargs):
    """Two-stream (InternVideo + VideoMamba) + CoDA-Q head and token-level fusion."""
    model = TwoStreamCoDAQ(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=14,
                              embed_dim=1408,
                              depth=40,
                              num_heads=16,
                              mlp_ratio=48 / 11,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              **kwargs)
    model.default_cfg = _cfg()
    return model


def interpolate_pos_embed_online(
    pos_embed, orig_size: Tuple[int], new_size: Tuple[int], num_extra_tokens: int
):
    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]
    embedding_size = pos_tokens.shape[-1]
    pos_tokens = pos_tokens.reshape(
        -1, orig_size[0], orig_size[1], embedding_size
    ).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=new_size, mode="bicubic", align_corners=False,
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    return new_pos_embed


if __name__ == '__main__':
    # test forward
    # create proposal
    from alphaction.structures.bounding_box import BoxList

    im_w, im_h = 464, 256
    n = 2
    xy = torch.zeros([n, 2])
    w = torch.rand([n, 1]) * 464
    h = torch.rand([n, 1]) * 256
    boxes = torch.cat([xy, w, h], dim=1)
    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # guard against no boxes

    boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xywh").convert("xyxy")
    # print(boxes.bbox)

    bs = 2
    t = 16
    proposals = [boxes, boxes]  # bs=2
    x = torch.rand([bs, 3, t, im_h, im_w])

    visual_transformer = vit_base_patch16_224(head_type='acar')
    print(visual_transformer)

    rois = visual_transformer(x, proposals)
    print(rois.shape)  # [4,num_classes]