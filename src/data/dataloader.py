# src/data/dataloader.py
import torch
import numpy as np


def test_collate_fn(batch):
    first_item = batch[0]
    has_tabular = len(first_item) == 3

    v1_list, v2_list, v3_list = [], [], []
    labels_list = []
    tabular_list = [] if has_tabular else None

    for item in batch:
        clips, label = item[0], item[1]
        v1_list.append(clips[0])
        v2_list.append(clips[1])
        v3_list.append(clips[2])
        labels_list.append(label)
        if has_tabular:
            tabular_list.append(item[2])

    images_v1 = torch.stack(v1_list, dim=0)
    images_v2 = torch.stack(v2_list, dim=0)
    images_v3 = torch.stack(v3_list, dim=0)
    labels = torch.stack(labels_list, dim=0)

    images = (images_v1, images_v2, images_v3)

    if has_tabular:
        tabular_data = torch.stack(tabular_list, dim=0)
        return images, labels, tabular_data
    else:
        return images, labels
