import torch
import torch.nn as nn


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim] or [bs] if label_dim is 1
        # output: [bs, bs]
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)  # Convert [bs] to [bs, 1]

        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # features: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim] (two views)
        # labels: [bs, label_dim] or [bs]

        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)  # Ensure labels are [bs, 1]

        # Process features from two views
        f1 = features[:, 0]  # [bs, feat_dim]
        f2 = features[:, 1]  # [bs, feat_dim]

        # Concatenate features for similarity calculation: [2*bs, feat_dim]
        all_features = torch.cat([f1, f2], dim=0)
        # Repeat labels for concatenated features: [2*bs, label_dim]
        all_labels = labels.repeat(2, 1)

        label_diffs = self.label_diff_fn(all_labels)  # [2bs, 2bs]
        logits = self.feature_sim_fn(all_features).div(self.t)  # [2bs, 2bs]

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # For numerical stability
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2*bs

        # Create a mask to remove diagonal elements (self-similarity)
        mask = (1 - torch.eye(n, device=logits.device)).bool()

        # Select non-diagonal elements
        logits_flat = logits.masked_select(mask).view(n, n - 1)
        exp_logits_flat = exp_logits.masked_select(mask).view(n, n - 1)
        label_diffs_flat = label_diffs.masked_select(mask).view(n, n - 1)

        loss = 0.
        # Iterate over each anchor sample
        for k in range(n - 1):  # For each potential positive pair (excluding self)
            # Current positive pair's logit and label difference
            pos_logits_k = logits_flat[:, k]  # Shape: (n,)
            pos_label_diffs_k = label_diffs_flat[:, k]  # Shape: (n,)

            # Mask for negative samples: label_diffs >= current positive's label_diff
            # neg_mask shape: (n, n-1)
            neg_mask = (label_diffs_flat >= pos_label_diffs_k.unsqueeze(1)).float()

            # Sum of exp_logits for negative samples
            # (neg_mask * exp_logits_flat) filters for negatives
            # .sum(dim=-1) sums over all negatives for each anchor
            log_denominator = torch.log((neg_mask * exp_logits_flat).sum(dim=-1) + 1e-8)  # Add epsilon for stability

            # Log probability for the current positive pair
            pos_log_probs = pos_logits_k - log_denominator  # Shape: (n,)

            # Accumulate loss, normalized as per original Rank-N-Contrast paper
            loss += -pos_log_probs.sum()

        return loss / (n * (n - 1))  # Normalize by total number of pairs considered

