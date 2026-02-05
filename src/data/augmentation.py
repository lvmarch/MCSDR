# src/data/augmentation.py
import albumentations as A
import cv2
import numpy as np
import random
from albumentations.core.transforms_interface import ImageOnlyTransform


class SpeckleNoise(ImageOnlyTransform):
    def __init__(self, mean=0, std=0.15, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, img, **params):
        img_float = img.astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, img.shape).astype(np.float32)
        noisy_img = img_float * (1 + noise)

        return (np.clip(noisy_img, 0, 1) * 255).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ('mean', 'std')


class AcousticShadow(ImageOnlyTransform):
    def __init__(self, n_shadows=(1, 2), alpha_max=0.85, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.n_shadows = n_shadows
        self.alpha_max = alpha_max

    def apply(self, img, **params):
        h, w = img.shape[:2]
        img_with_shadows = img.copy().astype(np.float32)

        num_shadows = random.randint(self.n_shadows[0], self.n_shadows[1])

        for _ in range(num_shadows):
            edge = random.randint(0, 3)

            if edge == 0:  # 从顶部
                vertex1 = (random.randint(0, w - 1), 0)
                base_point1 = (0, h - 1)
                base_point2 = (w - 1, h - 1)
            elif edge == 1:  # 从底部
                vertex1 = (random.randint(0, w - 1), h - 1)
                base_point1 = (0, 0)
                base_point2 = (w - 1, 0)
            elif edge == 2:  # 从左侧
                vertex1 = (0, random.randint(0, h - 1))
                base_point1 = (w - 1, 0)
                base_point2 = (w - 1, h - 1)
            else:  # 从右侧
                vertex1 = (w - 1, random.randint(0, h - 1))
                base_point1 = (0, 0)
                base_point2 = (0, h - 1)

            base_point1 = (base_point1[0] + random.randint(-w // 4, w // 4), base_point1[1])
            base_point2 = (base_point2[0] + random.randint(-w // 4, w // 4), base_point2[1])

            triangle = np.array([vertex1, base_point1, base_point2], dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, triangle, 1)  # 用1填充掩码
            mask = mask.astype(bool)

            alpha = random.uniform(0.4, self.alpha_max)

            img_with_shadows[mask] *= (1 - alpha)

        return np.clip(img_with_shadows, 0, 255).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ('n_shadows', 'alpha_max')


class Ensure3Channels(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        if img.ndim == 2:  # (H, W) -> (H, W, 3)
            return np.stack([img, img, img], axis=-1)
        if img.shape[2] == 1:  # (H, W, 1) -> (H, W, 3)
            return np.concatenate([img, img, img], axis=-1)
        if img.shape[2] == 4:  # (H, W, 4) -> (H, W, 3)
            return img[:, :, :3]
        return img

    def get_transform_init_args_names(self):
        return ()


def get_augmentation(n_frames=32, num_views=1):
    qpart_transforms = A.Compose([
        # # --- 1. 几何变换 ---
        # A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(
        #     shift_limit=0.06,
        #     scale_limit=0.1,
        #     rotate_limit=15,
        #     p=0.7,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=0
        # ),
        #
        # # --- 2. 颜色和强度变换 ---
        # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        #
        # # --- 3. 超声特定伪影模拟 ---
        # SpeckleNoise(std=0.1, p=0.5),
        # AcousticShadow(p=0.4),

        # --- 4. 必要的尺寸预处理 ---
        A.PadIfNeeded(124, 124, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        A.RandomCrop(112, 112, always_apply=True),
    ])

    # --- 传递 num_views ---
    video_augmentation = VideoAlbumentations(n_frames, qpart_transforms, num_views=num_views)

    return video_augmentation


def get_augmentation_for_test(n_frames=32, num_views=1):
    qpart_transforms = A.Compose([
        A.PadIfNeeded(124, 124, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        A.CenterCrop(112, 112, always_apply=True),
    ])

    # --- 传递 num_views ---
    video_augmentation = VideoAlbumentations(n_frames, qpart_transforms, num_views=num_views)

    return video_augmentation


class VideoAlbumentations:

    def __init__(self, n_frames, transform, num_views=1):
        self.n_frames = n_frames
        self.transform = transform
        self.num_views = num_views
        self.total_frames = self.n_frames * self.num_views

        self.transform.add_targets({f'image{i}': 'image' for i in range(self.total_frames)})


    def __call__(self, video_frames_list):
        if len(video_frames_list) != self.total_frames:
            raise ValueError(f"VideoAlbumentations 期望 {self.total_frames} 帧 "
                             f"(n_frames={self.n_frames} * num_views={self.num_views})，"
                             f"但收到了 {len(video_frames_list)} 帧。")

        inputs = {'image': video_frames_list[0]}
        inputs.update({f'image{i}': frame for i, frame in enumerate(video_frames_list)})

        transformed = self.transform(**inputs)

        out_video_list = [transformed[f'image{i}'] for i in range(self.total_frames)]
        return np.array(out_video_list)
