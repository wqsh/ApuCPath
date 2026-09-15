import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Normalize

def get_preprocessing_transform():
    """与训练时验证集的预处理保持一致（无随机裁剪/翻转）"""
    return Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class SegmentationDataset(Dataset):
    def __init__(self, data_root, split='train', patch_size=1024):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.images = sorted((self.data_root / split / 'images').glob('*.png'))
        self.masks = sorted((self.data_root / split / 'masks').glob('*.npy'))
        assert len(self.images) == len(self.masks), "Mismatch between images and masks"

        # 空间变换（裁剪、翻转）—— 同时对 image 和 mask 进行
        if split == 'train':
            self.spatial_transform = A.Compose([
                A.RandomCrop(patch_size, patch_size),
                A.HorizontalFlip(p=0.5),
            ])
        else:
            self.spatial_transform = None  # 验证集不做随机变换，但可能需要中心裁剪？这里保持原尺寸

        # 图像的数值变换（归一化 + ToTensor）
        self.image_transform = get_preprocessing_transform()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取原始数据
        image = np.array(Image.open(self.images[idx]).convert('RGB'))  # (H, W, 3)
        inst_by_class = np.load(self.masks[idx])  # (num_classes, H, W) int32

        # 将 mask 转换为 (H, W, num_classes) 以便与 albumentations 空间变换兼容
        mask_hwc = np.transpose(inst_by_class, (1, 2, 0))  # (H, W, num_classes)

        # 应用空间变换（仅裁剪和翻转，不影响数值）
        if self.spatial_transform is not None:
            transformed = self.spatial_transform(image=image, mask=mask_hwc)
            image = transformed['image']
            mask_hwc = transformed['mask']

        # 分别转换图像和 mask
        # 图像：归一化 + ToTensor
        image_tensor = self.image_transform(image=image)['image']  # (3, H, W)

        # mask：直接转为 torch.LongTensor，形状 (num_classes, H, W)
        mask_tensor = torch.from_numpy(np.transpose(mask_hwc, (2, 0, 1))).long()  # (num_classes, H, W)

        # 提取实例 masks 和 labels
        masks_list = []
        labels_list = []
        num_classes = mask_tensor.shape[0]
        for c in range(num_classes):
            channel = mask_tensor[c]  # (H, W)
            unique_ids = torch.unique(channel)
            unique_ids = unique_ids[unique_ids != 0]
            for inst_id in unique_ids:
                binary_mask = (channel == inst_id).float()
                masks_list.append(binary_mask)
                labels_list.append(c + 1)  # 类别 ID 从 1 开始

        if len(masks_list) == 0:
            # 无实例（理论上很少），返回空
            masks_tensor = torch.zeros((0, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
        else:
            masks_tensor = torch.stack(masks_list, dim=0)
            labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

        return {
            'image': image_tensor,
            'masks': masks_tensor,
            'labels': labels_tensor
        }