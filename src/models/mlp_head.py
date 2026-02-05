import torch
import torch.nn as nn


def get_shallow_mlp_head(dim_in, dim_out=1, dropout=0.4):  # Added dropout as parameter
    regressor = torch.nn.Sequential(
        torch.nn.BatchNorm1d(dim_in),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(dim_in, dim_in),
        torch.nn.BatchNorm1d(dim_in),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(dim_in, dim_out),
    )

    return regressor
