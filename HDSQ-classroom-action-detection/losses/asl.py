# losses/asl.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss for Multi-Label (Ben-Baruch et al., ICCV'20)
    - 支持 per-class pos_weight（用于长尾正例增强）
    - gamma_pos / gamma_neg：正负样本的不同聚焦因子
    - clip：对负类的 (1-p) 做上调裁剪，抑制易负样本梯度
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, pos_weight=None):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip) if clip is not None else None
        self.eps = float(eps)
        if pos_weight is not None:
            # 注册为 buffer，确保自动 to(device/dtype)
            self.register_buffer("pos_weight", pos_weight.reshape(1, -1).float())
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        """
        logits: [N, C] (未经 Sigmoid)
        targets: [N, C] in {0,1}
        """
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # asymmetric clipping for negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # log-prob
        # 避免 log(0)：加 eps
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        # 基础 CE（正负项分别计算）
        loss = targets * log_pos + (1.0 - targets) * log_neg  # [N, C]

        # asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = targets * xs_pos + (1.0 - targets) * xs_neg
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            modulating = (1.0 - pt) ** one_sided_gamma
            loss = loss * modulating

        # 正例权重（EN）：仅放大正例项
        if self.pos_weight is not None:
            # 令 W = 1 + (pos_weight - 1) * targets
            W = 1.0 + (self.pos_weight - 1.0) * targets
            loss = loss * W

        return -loss.mean()
