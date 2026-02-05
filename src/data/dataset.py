# src/data/dataset.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_video
from .augmentation import get_augmentation, get_augmentation_for_test
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import logging  # 添加日志记录
import random  # 为 RNC 添加
import torch.nn.functional as F  # <-- 导入 F.interpolate 所需


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    C, T, H, W = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        cropped_length = int(np.floor(valid_size * p))
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias: begin + bias + cropped_length, :, :]
        cropped_length = data.shape[1]  # 获取实际裁剪长度
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        min_frames = min(16, valid_size)
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), min_frames),
                                    valid_size)

        bias_range = valid_size - cropped_length + 1
        bias = np.random.randint(0, bias_range) if bias_range > 0 else 0

        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            logging.error(
                f"valid_crop_resize 裁剪后长度为 0。cropped_length={cropped_length}, bias={bias}, valid_size={valid_size}")
            data = data_numpy
            cropped_length = data.shape[1]

    # resize
    data = torch.tensor(data, dtype=torch.float)
    # (C, T_crop, H, W) -> (C*H*W, T_crop)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * H * W, cropped_length)

    if data.shape[-1] == 0:
        logging.error(f"valid_crop_resize 插值前 T_crop=0。回退...")
        data = torch.zeros((C, H, W, window)).permute(0, 3, 1, 2).contiguous().numpy()
        return data

    data = data[None, None, :, :]  # (1, 1, C*H*W, T_crop)
    data = F.interpolate(data, size=(C * H * W, window), mode='bilinear',
                         align_corners=False).squeeze()  # (C*H*W, window)
    data = data.contiguous().view(C, H, W, window).permute(0, 3, 1, 2).contiguous().numpy()  # (C, window, H, W)

    return data



class EchoNet(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.root = self.cfg['data']['data_folder']
        self.split = split

        self.file_list_path = self.cfg['data'].get('file_list_path')
        if not self.file_list_path:
            raise ValueError("[DATASET ERROR] 'data.file_list_path' 未在 YAML 配置文件中指定。")
        if not os.path.exists(self.file_list_path):
            raise FileNotFoundError(f"[DATASET ERROR] 找不到 'data.file_list_path' 指定的文件: {self.file_list_path}")

        self.file_list = pd.read_csv(self.file_list_path)

        if self.split != 'all' and 'Split' in self.file_list.columns:
            original_count = len(self.file_list)
            self.file_list = self.file_list[self.file_list['Split'].str.lower() == self.split.lower()].reset_index(
                drop=True)
            print(
                f"INFO: Filtered dataset for split '{self.split}'. Kept {len(self.file_list)} of {original_count} records.")

        if len(self.file_list) == 0:
            raise ValueError(
                f"\n\n[DATASET ERROR] After filtering for split='{self.split}', no data was found.\n"
                f"Please check your CSV file: '{self.cfg['data']['file_list_path']}'\n"
                f"Ensure that the 'Split' column exists and contains entries for '{self.split}' (case-insensitive).\n"
            )

        self.clip_length = self.cfg['data'].get('frames', 32)

        if self.split == 'train' and self.cfg['data'].get('aug', False):
            self.transform = get_augmentation(n_frames=self.clip_length, num_views=1)
        else:
            self.transform = get_augmentation_for_test(n_frames=self.clip_length, num_views=1)

        if self.split == 'train':
            self.p_interval = self.cfg['data'].get('p_interval_train', [0.8, 1.0])
        else:
            self.p_interval = self.cfg['data'].get('p_interval_test', [0.95])
        logging.info(f"EchoNet (split={split}): p_interval set to {self.p_interval}")

        self.use_tabular_data = self.cfg['data'].get('use_tabular_data', False)
        if self.use_tabular_data:
            print("INFO: Loading and processing tabular data with imputation.")
            self.tabular_cols = self.cfg['data']['tabular_cols']

            missing_cols = [col for col in self.tabular_cols if col not in self.file_list.columns]
            if missing_cols:
                raise ValueError(
                    f"[DATASET ERROR] 'tabular_cols' {missing_cols} in config not found in CSV file columns: {list(self.file_list.columns)}")

            tabular_df = self.file_list[self.tabular_cols].copy()

            if 'Sex' in self.tabular_cols:
                tabular_df.loc[:, 'Sex'], _ = pd.factorize(tabular_df['Sex'])
                print("INFO: 'Sex' column factorized. NaNs are mapped to -1.")

            numerical_cols = [col for col in self.tabular_cols if col != 'Sex']
            if numerical_cols:
                imputer = SimpleImputer(strategy='mean')
                tabular_df[numerical_cols] = imputer.fit_transform(tabular_df[numerical_cols])
                print(f"INFO: NaN values in {numerical_cols} imputed with their respective column means.")

            scaler = MinMaxScaler()
            tabular_values = tabular_df.values.astype(float)
            normalized_data = scaler.fit_transform(tabular_values)
            self.tabular_data = torch.from_numpy(normalized_data).float()
            print(f"INFO: Successfully processed and scaled tabular columns: {self.tabular_cols}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if 'FileName' not in self.file_list.columns:
            raise ValueError(f"[DATASET ERROR] EchoNet 加载器期望在 CSV 中找到 'FileName' 列。")

        record = self.file_list.iloc[index]
        video_name = record['FileName']

        if not video_name.endswith(".avi"):
            video_name_with_ext = video_name + ".avi"
        else:
            video_name_with_ext = video_name

        video_path = os.path.join(self.root, "Videos", video_name_with_ext)

        try:
            video, _, info = read_video(video_path, pts_unit='sec', output_format="TCHW")
            video = video.float() / 255.0  # (T_full, C, H, W), [0, 1]
        except Exception as e:
            print(f"ERROR: Could not read video {video_path}. Error: {e}")
            dummy_ef = torch.zeros(1)
            if self.use_tabular_data:
                dummy_tabular_dim = self.cfg['data'].get('tabular_config', {}).get('dim', 0)
                return torch.zeros((3, self.clip_length, 112, 112)), dummy_ef, torch.zeros(dummy_tabular_dim)
            else:
                return torch.zeros((3, self.clip_length, 112, 112)), dummy_ef

        total_frames = video.shape[0]

        video = video.permute(1, 0, 2, 3)

        video_resized_np = valid_crop_resize(video.numpy(), total_frames, self.p_interval, self.clip_length)

        video_tensor = torch.from_numpy(video_resized_np).permute(1, 0, 2, 3)

        video_tensor_T_HWC = video_tensor.permute(0, 2, 3, 1)  # (T_window, H, W, C)
        video_np_list = [
            (torch.clip(frame, 0, 1) * 255).numpy().astype(np.uint8)
            for frame in video_tensor_T_HWC
        ]

        if self.transform:
            transformed_video_np = self.transform(video_np_list)
            video = torch.from_numpy(transformed_video_np).permute(0, 3, 1, 2).contiguous() / 255.0
        else:
            video = video_tensor.contiguous()  # (T_window, C, H, W)

        video_tensor = video.permute(1, 0, 2, 3)  # (C, T_window, H, W)

        ef = record['EF']
        ef = torch.tensor([ef], dtype=torch.float32)

        if self.use_tabular_data:
            tabular_vec = self.tabular_data[index]
            return video_tensor, ef, tabular_vec
        else:
            return video_tensor, ef


class EchoNetRNC(EchoNet):
    def __init__(self, cfg, split='train'):
        super().__init__(cfg, split)
        self.n_views = self.cfg['data'].get('n_views', 2)
        if self.transform is None:
            raise ValueError(
                "RNC requires augmentations, but none were created. Check config `data.aug` is true for training.")

        self.p_interval = self.cfg['data'].get('p_interval_train', [0.8, 1.0])
        logging.info(f"EchoNetRNC (split={split}): p_interval hardcoded to train: {self.p_interval}")

    def __getitem__(self, index):
        if 'FileName' not in self.file_list.columns:
            raise ValueError(f"[DATASET ERROR] EchoNetRNC 加载器期望在 CSV 中找到 'FileName' 列。")

        record = self.file_list.iloc[index]
        video_name = record['FileName']

        if not video_name.endswith(".avi"):
            video_name_with_ext = video_name + ".avi"
        else:
            video_name_with_ext = video_name

        video_path = os.path.join(self.root, "Videos", video_name_with_ext)

        try:
            video, _, _ = read_video(video_path, pts_unit='sec', output_format="TCHW")
            video = video.float() / 255.0  # (T, C, H, W)
        except Exception as e:
            logging.error(f"ERROR: RNC 无法读取 {video_path}: {e}")
            dummy_video = torch.zeros((3, self.clip_length, 112, 112))
            dummy_views = [dummy_video, dummy_video]
            ef = record['EF']
            ef = torch.tensor([ef], dtype=torch.float32)
            if self.use_tabular_data:
                tabular_vec = self.tabular_data[index]
                return dummy_views, ef, tabular_vec
            else:
                return dummy_views, ef

        total_frames = video.shape[0]
        video = video.permute(1, 0, 2, 3)  # (C, T_full, H, W)
        video_np = video.numpy()

        views = []
        for _ in range(self.n_views):
            video_resized_np = valid_crop_resize(video_np, total_frames, self.p_interval, self.clip_length)

            video_tensor = torch.from_numpy(video_resized_np).permute(1, 0, 2, 3)  # (T_window, C, H, W)
            video_tensor_T_HWC = video_tensor.permute(0, 2, 3, 1)  # (T_window, H, W, C)
            video_np_list = [
                (torch.clip(frame, 0, 1) * 255).numpy().astype(np.uint8)
                for frame in video_tensor_T_HWC
            ]

            transformed_np = self.transform(video_np_list)
            view_tensor = torch.from_numpy(transformed_np).permute(0, 3, 1, 2).contiguous() / 255.0
            view_tensor = view_tensor.permute(1, 0, 2, 3)  # (C, T_window, H, W)
            views.append(view_tensor)

        ef = record['EF']
        ef = torch.tensor([ef], dtype=torch.float32)

        if self.use_tabular_data:
            tabular_vec = self.tabular_data[index]
            return views, ef, tabular_vec
        else:
            return views, ef

