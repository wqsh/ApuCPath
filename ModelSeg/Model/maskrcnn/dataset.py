import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np

import random
from torch.utils.data import Sampler

class SyntheticShapesDataset(Dataset):
    """
    生成包含随机形状（圆形、矩形）的图像，每个图像有1~4个目标。
    类别：1（形状），背景为0。
    """
    def __init__(self, num_samples=100, img_size=256, class_names = ['BG', 'circle', 'rectangle']):
        self.num_samples = num_samples
        self.img_size = img_size
        self.class_names = class_names

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 创建空白图像（RGB）和配套的绘制对象
        img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 随机决定目标数量
        # num_objects = random.randint(1, 4)
        num_objects = 4
        boxes = []
        masks = []
        labels = []

        for _ in range(num_objects):
            shape_type = random.choice(self.class_names[1:])
            color = tuple(random.randint(100, 255) for _ in range(3))  # 图像颜色，避免全黑

            # 为当前目标创建一个单独的掩膜画布（灰度，0为背景，1为前景）
            mask = Image.new('L', (self.img_size, self.img_size), 0)
            mask_draw = ImageDraw.Draw(mask)

            # 随机位置和大小，并同时在 RGB 图和 mask 图上绘制
            if shape_type == 'circle':
                r = random.randint(15, 40)
                x_center = random.randint(r, self.img_size - r - 1)
                y_center = random.randint(r, self.img_size - r - 1)
                x1, y1 = x_center - r, y_center - r
                x2, y2 = x_center + r, y_center + r
                draw.ellipse([x1, y1, x2, y2], fill=color)  # 彩色图像
                mask_draw.ellipse([x1, y1, x2, y2], fill=1)  # 二值掩膜
            elif shape_type == 'rectangle':
                w = random.randint(20, 50)
                h = random.randint(20, 50)
                x1 = random.randint(0, self.img_size - w)
                y1 = random.randint(0, self.img_size - h)
                x2, y2 = x1 + w, y1 + h
                draw.rectangle([x1, y1, x2, y2], fill=color)
                mask_draw.rectangle([x1, y1, x2, y2], fill=1)

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_names.index(shape_type)) 
            # 将 PIL mask 转为 numpy 数组，值自然为 0 或 1（dtype=np.uint8）
            masks.append(np.array(mask, dtype=np.uint8))

        # 转换为 tensor
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        masks_tensor = torch.as_tensor(np.array(masks), dtype=torch.uint8)  # [N, H, W]
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])

        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'masks': masks_tensor,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }

        # 图像转为 tensor 并归一化到 [0,1]
        img_tensor = T.functional.to_tensor(img)
        return img_tensor, target

def img_to_tensor_predict(img):
    # 图像转为 tensor 并归一化到 [0,1]
    return T.functional.to_tensor(img)

def tensor_to_img_predict(img_tensor):
    return (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)