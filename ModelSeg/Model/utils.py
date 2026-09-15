import torch
import torchvision.ops as ops
from torch.utils.data import Sampler

import random

class BalancedBatchSampler(Sampler):
    """每个 batch 中前景和背景图像数量相等（1:1），剩余不足时单独成 batch

    Args:
        dataset: 数据集，每个样本应包含 'boxes' 用于判断前景/背景
        batch_size: 每个 batch 的总图像数（必须为偶数）
        drop_last: 是否丢弃最后一个不足 batch_size 的 batch
    """

    def __init__(self, dataset, batch_size, drop_last=False):
        if batch_size % 2 != 0:
            raise (f"batch_size must be even number, current is {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.fg_per_batch = batch_size // 2
        self.bg_per_batch = batch_size // 2
        self.drop_last = drop_last

        # 分离前景和背景索引
        self.fg_indices = []
        self.bg_indices = []
        for idx in range(len(dataset)):
            # 假设 dataset[idx] 返回 (image, target)
            if  dataset.image_info[idx]["mask"] is None:
                self.bg_indices.append(idx)
            else:
                self.fg_indices.append(idx)

        # 检查前景数量是否足够（至少能凑出一个 batch 的一半）
        if len(self.fg_indices) < self.fg_per_batch:
            raise (f"fg image must large than {self.fg_per_batch}，only have {len(self.fg_indices)}")

    def __iter__(self):
        # 每个 epoch 重新打乱
        fg = self.fg_indices.copy()
        bg = self.bg_indices.copy()
        random.shuffle(fg)
        random.shuffle(bg)

        batches = []
        # 构造完整 batch（前景和背景各取固定数量）
        while len(fg) >= self.fg_per_batch and len(bg) >= self.bg_per_batch:
            batch = []
            batch.extend(fg[:self.fg_per_batch])
            fg = fg[self.fg_per_batch:]
            batch.extend(bg[:self.bg_per_batch])
            bg = bg[self.bg_per_batch:]
            random.shuffle(batch)  # 可选：打乱批次内顺序
            batches.append(batch)

        # 处理剩余样本（不足一个完整 batch）
        if not self.drop_last:
            remaining = fg + bg
            if remaining:
                random.shuffle(remaining)
                batches.append(remaining)

        for batch in batches:
            yield batch

    def __len__(self):
        total = len(self.fg_indices) + len(self.bg_indices)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size

