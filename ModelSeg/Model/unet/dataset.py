import random
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms as T

class SyntheticShapesDataset(Dataset):
    """
    生成合成形状数据集（圆形、矩形），每个图像包含固定数量的目标（默认4个）。
    返回：
        image: Tensor, shape [3, H, W], 值范围 [0,1]
        mask:  Tensor, shape [H, W], dtype=torch.long, 像素值 0=背景,1=圆形,2=矩形
    """
    def __init__(self, num_samples=100, img_size=256, num_objects=4, class_names=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_objects = num_objects
        if class_names is None:
            self.class_names = ['BG', 'circle', 'rectangle']
        else:
            self.class_names = class_names

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 创建空白 RGB 图像（黑色背景）
        img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 初始化语义掩码：0 表示背景
        semantic_mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)

        for _ in range(self.num_objects):
            shape_type = random.choice(self.class_names[1:])  # 随机选择圆形或矩形
            color = tuple(random.randint(100, 255) for _ in range(3))

            # 为该目标创建临时单通道掩码，用于填充语义掩码
            temp_mask = Image.new('L', (self.img_size, self.img_size), 0)
            mask_draw = ImageDraw.Draw(temp_mask)

            if shape_type == 'circle':
                r = random.randint(15, 40)
                x_center = random.randint(r, self.img_size - r - 1)
                y_center = random.randint(r, self.img_size - r - 1)
                x1, y1 = x_center - r, y_center - r
                x2, y2 = x_center + r, y_center + r
                draw.ellipse([x1, y1, x2, y2], fill=color)
                mask_draw.ellipse([x1, y1, x2, y2], fill=1)      # 临时掩码中目标区域置1
            elif shape_type == 'rectangle':
                w = random.randint(20, 50)
                h = random.randint(20, 50)
                x1 = random.randint(0, self.img_size - w)
                y1 = random.randint(0, self.img_size - h)
                x2, y2 = x1 + w, y1 + h
                draw.rectangle([x1, y1, x2, y2], fill=color)
                mask_draw.rectangle([x1, y1, x2, y2], fill=1)

            # 将临时掩码转换为 numpy 数组，并更新语义掩码（后绘制的目标会覆盖先绘制的）
            temp_mask_np = np.array(temp_mask, dtype=np.bool_)
            class_id = self.class_names.index(shape_type)   # 1:circle, 2:rectangle
            semantic_mask[temp_mask_np] = class_id

        # 转换为 tensor
        img_tensor = F.to_tensor(img)                     # [3, H, W], 归一化到 [0,1]
        mask_tensor = torch.as_tensor(semantic_mask, dtype=torch.long)

        return img_tensor, mask_tensor

def img_to_tensor_predict(img):
    # 图像转为 tensor 并归一化到 [0,1]
    return T.functional.to_tensor(img)
