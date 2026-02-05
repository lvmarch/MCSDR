# src/utils/metrics.py
import torch
import numpy as np
from scipy.stats import norm


def calculate_crps(preds_k_samples, labels):
    if preds_k_samples.shape[1] == 0:
        return 0.0

    labels_reshaped = labels.unsqueeze(1)  # [B, 1]

    term1 = torch.abs(preds_k_samples - labels_reshaped).mean(dim=1)

    term2 = torch.abs(
        preds_k_samples.unsqueeze(2) - preds_k_samples.unsqueeze(1)
    ).mean(dim=[1, 2])

    crps_per_sample = term1 - 0.5 * term2

    return crps_per_sample.mean().item()
