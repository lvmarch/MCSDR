# src/engine/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFocusingL1Loss(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, static_weight_bins=None, static_weights=None):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.register_buffer('static_weight_bins', torch.tensor(static_weight_bins) if static_weight_bins else None)
        self.register_buffer('static_weights', torch.tensor(static_weights) if static_weights else None)

    def get_static_weights(self, y_true):
        if self.static_weight_bins is None or self.static_weights is None:
            return torch.ones_like(y_true)

        weights = torch.ones_like(y_true)
        # 遍历每个分箱来分配权重
        for i in range(len(self.static_weight_bins) - 1):
            lower_bound = self.static_weight_bins[i]
            upper_bound = self.static_weight_bins[i + 1]
            # 找到落在当前分箱内的样本
            mask = (y_true >= lower_bound) & (y_true < upper_bound)
            weights[mask] = self.static_weights[i]
        return weights

    def forward(self, y_pred, y_true):
        l1_loss = torch.abs(y_true - y_pred)

        focusing_factor = (l1_loss.detach() / self.beta) ** self.gamma

        static_weights = self.get_static_weights(y_true)

        # loss = static_weights * focusing_factor * l1_loss
        loss = static_weights * focusing_factor * l1_loss  # 您的 diffusion loss 是预测噪声，因此这里的y_pred和y_true应为predicted_noise和noise

        return loss.mean()

def wasserstein_loss_1d(pred, target):
    pred_sorted, _ = torch.sort(pred, dim=1)
    target_sorted, _ = torch.sort(target, dim=1)

    return torch.mean(torch.abs(pred_sorted - target_sorted))