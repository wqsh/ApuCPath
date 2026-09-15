import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from Model.utils import BalancedBatchSampler
from Model.UNIMask2Former.dataset import get_preprocessing_transform

import timm
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
import os
import shutil
from pathlib import Path
from PIL import Image

import sys

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

# ============================================
# 1. UNI Encoder (ViT-Large)
# ============================================
class UNIEncoder(nn.Module):
    def __init__(self, checkpoint_path: str = None, frozen=True, out_indices=(6,12,18,24)):
        super().__init__()
        self.model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                                        init_values=1e-5, num_classes=0, dynamic_img_size=True)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
        self.frozen = frozen
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval()
        self.num_features = self.model.num_features  # 1024
        self.out_indices = out_indices

    def forward(self, x):
        B, _, H, W = x.shape
        # 获取 patch size（ViT-L/16 中为 16）
        patch_size = self.model.patch_embed.patch_size
        if isinstance(patch_size, int):
            ph, pw = patch_size, patch_size
        else:
            ph, pw = patch_size[0], patch_size[1]
        H_patch = H // ph
        W_patch = W // pw
        D = self.num_features

        # Patch embedding
        x = self.model.patch_embed(x)          # (B, N, D), N = 1 + H_patch*W_patch (包含 CLS)
        # 添加位置编码
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)

        outputs = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i+1 in self.out_indices:
                # 去掉 CLS token
                x_patch = x[:, 1:, :]          # (B, H_patch*W_patch, D)
                feat = x_patch.permute(0, 2, 1).reshape(B, D, H_patch, W_patch)
                outputs.append(feat)

        # 最后一层加 norm
        x = self.model.norm(x)
        x_patch = x[:, 1:, :]
        final = x_patch.permute(0, 2, 1).reshape(B, D, H_patch, W_patch)
        outputs.append(final)   # 共 5 个特征图
        return outputs

# ============================================
# 2. Pixel Decoder (Transformer based)
# ============================================

class PixelDecoderFPN(nn.Module):
    """多尺度特征融合，输出单一高分辨率特征图"""
    def __init__(self, in_channels_list, hidden_dim=256):
        super().__init__()
        # in_channels_list: [c1, c2, c3, c4, c5] 对应 UNIEncoder 返回的5个尺度
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for i, in_c in enumerate(in_channels_list):
            lateral = nn.Conv2d(in_c, hidden_dim, kernel_size=1)
            output = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.lateral_convs.append(lateral)
            self.output_convs.append(output)

        # 最顶层额外处理
        self.top_down = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features: list of 5 tensors, 从浅到深
        laterals = [lateral(feat) for lateral, feat in zip(self.lateral_convs, features)]
        # 自顶向下融合
        out = laterals[-1]
        for i in range(len(laterals)-2, -1, -1):
            out = F.interpolate(out, size=laterals[i].shape[-2:], mode='bilinear', align_corners=False)
            out = out + laterals[i]
            out = self.output_convs[i](out)
        out = self.final_conv(out)
        return out  # (B, hidden_dim, H, W)

class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, in_channels=768, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=6
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.input_proj(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        x_encoded = self.encoder(x_flat)
        x_encoded = x_encoded.permute(0, 2, 1).reshape(b, c, h, w)
        return self.output_conv(x_encoded)

# ============================================
# 3. Transformer Decoder (Mask2Former Decoder)
# ============================================
class MaskedAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, memory_mask=None):
        # tgt: (B, Q, D)
        # memory: (B, S, D)
        # memory_mask: (B, Q, S)  (加法偏置, 应取极大负值)
        # self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.norm1(tgt2)
        # cross-attention with mask
        if memory_mask is not None:
            # 将 mask 扩展为 (B*num_heads, Q, S) 以适配 MultiheadAttention
            B, Q, S = memory_mask.shape
            num_heads = self.cross_attn.num_heads
            memory_mask = memory_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(B*num_heads, Q, S)
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.norm2(tgt2)
        # FFN
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.norm3(tgt2)
        return tgt

class Mask2FormerDecoder(nn.Module):
    def __init__(self, d_model=256, num_queries=100, num_layers=9):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.mask_embed = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList([MaskedAttentionLayer(d_model) for _ in range(num_layers)])
        self.mask_update_proj = nn.Linear(d_model, d_model)

    def forward(self, memory, pixel_features):
        """
        memory: (B, S, D)  (S = H*W)
        pixel_features: (B, D, H, W)  用于生成掩码
        """
        B, S, D = memory.shape
        H, W = pixel_features.shape[-2:]
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        mask_emb = self.mask_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # 存储每一层的 mask logits（用于辅助损失，可选）
        all_mask_logits = []

        for layer in self.layers:
            # 生成当前 mask logits
            mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_emb, pixel_features)  # (B, Q, H, W)
            all_mask_logits.append(mask_logits)

            # 构造注意力偏置（官方方式：-sigmoid * 极大值）
            mask_bias = -torch.sigmoid(mask_logits).flatten(2) * 100.0   # 或 1e4，确保 < 65504

            # 通过掩码注意力层
            tgt = layer(tgt, memory, memory_mask=mask_bias)

            # 更新 mask embedding
            mask_emb = self.mask_update_proj(tgt)

        # 返回最后一层的 mask 和所有 mask logits（可选）
        return tgt, all_mask_logits[-1], all_mask_logits

# ============================================
# 4. Combined Model: UNI + Mask2Former
# ============================================
class UNIMask2Former(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 256, num_queries: int = 100,
                 uni_checkpoint: str = None, backbone_frozen: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = UNIEncoder(uni_checkpoint, frozen=backbone_frozen,
                                   out_indices=(6,12,18,24))  # 返回5个尺度
        # 获取每个尺度的通道数（ViT-L 均为 1024）
        in_channels_list = [self.backbone.num_features] * 5
        self.pixel_decoder = PixelDecoderFPN(in_channels_list, hidden_dim=d_model)
        self.transformer_decoder = Mask2FormerDecoder(d_model=d_model, num_queries=num_queries)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.mask_embed = nn.Linear(d_model, d_model)

    def forward(self, x, return_aux_loss=False):
        multi_feats = self.backbone(x)          # list of 5 tensors
        pixel_features = self.pixel_decoder(multi_feats)   # (B, d_model, H, W)
        b, c, h, w = pixel_features.shape
        memory = pixel_features.flatten(2).permute(0, 2, 1)   # (B, H*W, d_model)

        queries, mask_logits, all_masks = self.transformer_decoder(memory, pixel_features)

        class_logits = self.class_embed(queries)   # (B, Q, num_classes+1)

        if return_aux_loss:
            # 返回所有解码器层的 mask logits 用于深监督（可选）
            aux_outputs = [{'mask_logits': m, 'class_logits': class_logits} for m in all_masks[:-1]]
            print(f"aux_outputs1: {str(aux_outputs)}")

            return {
                'class_logits': class_logits,
                'mask_logits': mask_logits,
                'aux_outputs': aux_outputs
            }
        else:
            return {'class_logits': class_logits, 'mask_logits': mask_logits}

# ============================================
# 5. Loss Functions (匈牙利匹配)
# ============================================
class HungarianMatcher(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['class_logits'].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs['class_logits'][b].softmax(-1)  # [Q, C+1]
            tgt_ids = targets[b]['labels']  # [M]

            # 如果没有目标，返回空匹配
            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            # 分类代价：添加 eps 避免 log(0)
            cost_class = -torch.log(out_prob[:, tgt_ids] + 1e-8)  # [Q, M]

            # Mask 代价
            out_mask = outputs['mask_logits'][b]  # [Q, H, W]
            tgt_mask = targets[b]['masks']  # [M, H_raw, W_raw]
            H_out, W_out = out_mask.shape[-2:]
            tgt_mask_resized = F.interpolate(
                tgt_mask.unsqueeze(1), size=(H_out, W_out),
                mode='bilinear', align_corners=False
            ).squeeze(1)  # [M, H, W]

            cost_mask = self._mask_cost(out_mask, tgt_mask_resized, bce_weight=1.0, dice_weight=1.0)

            C = cost_class + cost_mask
            # 替换 NaN/inf 为大数值
            if torch.isnan(C).any() or torch.isinf(C).any():
                C = torch.nan_to_num(C, nan=1e10, posinf=1e10, neginf=1e10)
            C_np = C.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_np)
            indices.append((torch.as_tensor(row_ind), torch.as_tensor(col_ind)))
        return indices

    @staticmethod
    def _mask_cost(pred_masks, true_masks, bce_weight=1.0, dice_weight=1.0):
        """
        pred_masks: [Q, H, W] logits
        true_masks: [M, H, W] binary float (0/1)
        return: [Q, M] cost = bce_weight * BCE + dice_weight * (1 - dice)
        """
        # Dice part
        pred = pred_masks.sigmoid().flatten(1)  # [Q, HW]
        true = true_masks.flatten(1)  # [M, HW]
        numerator = 2.0 * torch.matmul(pred, true.t())
        denominator = pred.sum(1, keepdim=True) + true.sum(1, keepdim=True).t()
        dice = (numerator + 1e-8) / (denominator + 1e-8)
        dice_cost = 1 - dice  # [Q, M]

        # BCE part
        # Expand pred_masks to [Q, M, H, W] and true_masks to [Q, M, H, W]
        Q, H, W = pred_masks.shape
        M = true_masks.shape[0]
        pred_expanded = pred_masks.unsqueeze(1).expand(-1, M, -1, -1)  # [Q, M, H, W]
        true_expanded = true_masks.unsqueeze(0).expand(Q, -1, -1, -1)  # [Q, M, H, W]
        bce = F.binary_cross_entropy_with_logits(pred_expanded, true_expanded, reduction='none')
        bce_cost = bce.mean(dim=(2, 3))  # [Q, M]

        return bce_weight * bce_cost + dice_weight * dice_cost

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1, aux_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.aux_weight = aux_weight
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['class_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_masks(self, outputs, targets, indices):
        src_masks = outputs['mask_logits']
        idx = self._get_src_permutation_idx(indices)

        # 如果没有匹配到任何目标，返回零损失（两个键都要有）
        if idx[0].numel() == 0:
            device = src_masks.device
            return {
                'loss_mask': torch.tensor(0.0, device=device),
                'loss_dice': torch.tensor(0.0, device=device)
            }

        src_masks = src_masks[idx]
        target_masks_flat = []
        for b, (_, J) in enumerate(indices):
            tgt = targets[b]['masks'][J]
            tgt_resized = F.interpolate(tgt.unsqueeze(1), size=src_masks.shape[-2:],
                                        mode='bilinear', align_corners=False).squeeze(1)
            target_masks_flat.append(tgt_resized)
        target_masks = torch.cat(target_masks_flat, dim=0)

        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks)
        loss_dice = self.dice_loss(src_masks.sigmoid(), target_masks)
        return {'loss_mask': loss_mask, 'loss_dice': loss_dice}

    def dice_loss(self, inputs, targets):
        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        losses = {}
        indices = self.matcher(outputs, targets)
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))

        if 'aux_outputs' in outputs:
            for i, aux_out in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_out, targets)
                losses.update({f'{k}_{i}': v for k, v in self.loss_labels(aux_out, targets, aux_indices).items()})
                losses.update({f'{k}_{i}': v for k, v in self.loss_masks(aux_out, targets, aux_indices).items()})
        return losses

def build_criterion(num_classes, matcher):
    weight_dict = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}
    criterion = SetCriterion(num_classes, matcher, weight_dict)
    return criterion, weight_dict  # 返回权重字典

# ============================================
# 6. Data Collate Function
# ============================================
def collate_fn(batch):
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'masks': [item['masks'] for item in batch],
        'labels': [item['labels'] for item in batch],
    }

# ============================================
# 7. 辅助评估函数
# ============================================
def calculate_iou(pred_mask, true_mask, num_classes):
    ious = []
    for cls in range(1, num_classes + 1):
        pred = (pred_mask == cls)
        true = (true_mask == cls)
        intersection = np.logical_and(pred, true).sum()
        union = np.logical_or(pred, true).sum()
        iou = intersection / union if union > 0 else np.nan
        ious.append(iou)
    return np.nanmean(ious)

def evaluate(model, dataloader, device, num_classes, criterion=None):
    """
    criterion: SetCriterion 实例，用于计算验证损失（可选）
    返回: dict with keys 'loss' (平均损失) 和 'mIoU'
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            # 构造 targets 列表，每个元素是 {'masks': Tensor, 'labels': Tensor}
            targets = [
                {'masks': m.to(device), 'labels': l.to(device)}
                for m, l in zip(batch['masks'], batch['labels'])
            ]
            outputs = model(images)

            if criterion is not None:
                loss_dict = criterion(outputs, targets)
                total_loss += sum(loss_dict.values()).item()

            # # 计算每个样本的语义预测图用于 mIoU
            # B = images.shape[0]
            # for b in range(B):
            #     h, w = batch['masks'][b].shape[1], batch['masks'][b].shape[2]
            #     true_semantic = torch.zeros(h, w, dtype=torch.long, device=device)
            #     masks_b = batch['masks'][b]
            #     labels_b = batch['labels'][b]
            #     for i in range(masks_b.shape[0]):
            #         lbl = labels_b[i].item() if torch.is_tensor(labels_b[i]) else labels_b[i]
            #         mask_i = masks_b[i].squeeze()
            #         true_semantic[mask_i > 0.5] = lbl
            #
            #     # 预测语义图
            #     mask_logits = outputs['mask_logits'][b]           # [Q, H_feat, W_feat]
            #     class_logits = outputs['class_logits'][b]         # [Q, num_classes+1]
            #     mask_probs = torch.sigmoid(mask_logits)
            #     mask_probs_up = F.interpolate(
            #         mask_probs.unsqueeze(0), size=(h, w),
            #         mode='bilinear', align_corners=False
            #     ).squeeze(0)                                      # [Q, H, W]
            #     class_probs = class_logits.softmax(dim=-1)[:, :-1] # 去掉“无对象”类
            #     # 融合 mask 和 class 得到每个像素的类别得分
            #     scores = torch.einsum('qhw,qc->chw', mask_probs_up, class_probs)  # [C, H, W]
            #     pred_semantic = scores.argmax(dim=0)              # [H, W]
            #
            #     all_preds.append(pred_semantic.cpu().numpy())
            #     all_targets.append(true_semantic.cpu().numpy())

    # # 计算 mIoU
    # ious = []
    # for p, t in zip(all_preds, all_targets):
    #     for cls in range(1, num_classes + 1):
    #         pred = (p == cls)
    #         true = (t == cls)
    #         inter = np.logical_and(pred, true).sum()
    #         union = np.logical_or(pred, true).sum()
    #         iou = inter / union if union > 0 else np.nan
    #         ious.append(iou)
    # miou = np.nanmean(ious) if ious else 0.0

    avg_loss = total_loss / len(dataloader) if criterion is not None else 0.0
    # return {'loss': avg_loss, 'mIoU': miou}
    return {'loss': avg_loss}

def evaluate(model, dataloader, device, num_classes, criterion=None, epoch=0, total_epochs=1):
    """
    criterion: SetCriterion 实例，用于计算验证损失（可选）
    返回: dict with keys 'loss' (平均损失) 和 'mIoU'
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        # 进度条：与 train 风格一致，描述改为 valid
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} valid")
        for batch in progress_bar:
            images = batch['image'].to(device)
            targets = [
                {'masks': m.to(device), 'labels': l.to(device)}
                for m, l in zip(batch['masks'], batch['labels'])
            ]
            outputs = model(images)

            if criterion is not None:
                loss_dict = criterion(outputs, targets)
                # 计算当前批次的总损失（加权和，与 train 一致）
                batch_loss = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys())
                total_loss += batch_loss.item()
                # 实时更新进度条显示当前批次损失
                progress_bar.set_postfix({'loss': batch_loss.item()})

    avg_loss = total_loss / len(dataloader) if criterion is not None else 0.0

    # return {'loss': avg_loss, 'mIoU': miou}
    return {'loss': avg_loss}


# ============================================
# 8. 主封装类 ModelMask2Former (类似 ModelUnet)
# ============================================
class ModelUNIMask2Former:
    def __init__(self, mode: str, config, checkpoint_path: str = None,
                 add_time_folder: bool = True, new_train: bool = True):
        """
        config: 配置对象，应包含以下属性：
            DATA_ROOT, NUM_CLASSES, UNI_CHECKPOINT, BACKBONE_FROZEN,
            BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
            OPTIMIZER, LR_SCHEDULER, LR_STEP_SIZE, LR_GAMMA, LR_MILESTONES,
            LR_T_MAX, LR_ETA_MIN, LR_FACTOR, LR_PATIENCE, LR_PCT_START,
            USE_GPU, OUTPUT_ROOT, SHUFFLE_TRAIN, SHUFFLE_VAL
        """
        if config.USE_GPU:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')
        self.config = config
        self.mode = mode
        self.build(mode, checkpoint_path, add_time_folder, new_train)

    def build(self, mode: str, checkpoint_path: str = None,
              add_time_folder: bool = True, new_train: bool = True):
        assert mode in ['train', 'inference']
        if mode == "train":
            folder_export = self.config.OUTPUT_ROOT
            self.start_epoch = 0

            if checkpoint_path is not None:
                if new_train:
                    # 从预训练模型初始化，但重新训练
                    self._load_checkpoint(checkpoint_path, load_optim=False)
                    self.start_epoch = 0
                    if add_time_folder:
                        self.folder_export = os.path.join(folder_export, CLogging.GetLocTimeStrForName())
                    else:
                        self.folder_export = folder_export
                else:
                    # 恢复训练
                    self.config.NUM_CLASSES = self._get_checkpoint_num_classes(checkpoint_path)
                    self._load_checkpoint(checkpoint_path, load_optim=True)
                    if os.path.splitext(checkpoint_path)[0].endswith("_best") or os.path.splitext(checkpoint_path)[0].endswith("_latest"):
                        self.folder_export = os.path.dirname(checkpoint_path)
                    else:
                        self.folder_export = os.path.dirname(os.path.dirname(checkpoint_path))
            else:
                self._init_model()
                if add_time_folder:
                    self.folder_export = os.path.join(folder_export, CLogging.GetLocTimeStrForName())
                else:
                    self.folder_export = folder_export

            os.makedirs(self.folder_export, exist_ok=True)
            # 保存配置（假设config有save_to_file方法）
            if hasattr(self.config, 'save_to_file'):
                self.config.save_to_file(os.path.join(self.folder_export, "parameter.txt"))
            # TensorBoard
            log_dir = os.path.join(self.folder_export, "tensorboard_logs")
            self.writer = SummaryWriter(log_dir=log_dir)
            self.model_export_folder = os.path.join(self.folder_export, "model")
            os.makedirs(self.model_export_folder, exist_ok=True)
            self.best_loss = float('inf')
            self.class_ids = [str(i) for i in range(self.config.NUM_CLASSES)]

        elif mode == "inference":
            self.config.NUM_CLASSES = self._get_checkpoint_num_classes(checkpoint_path)
            self._load_checkpoint(checkpoint_path, load_optim=False)

    def _init_model(self):
        self.model = UNIMask2Former(
            num_classes=self.config.NUM_CLASSES,
            d_model=256,
            num_queries=100,
            uni_checkpoint=self.config.UNI_CHECKPOINT,
            backbone_frozen=self.config.BACKBONE_FROZEN
        ).to(self.device)

    def _get_checkpoint_num_classes(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        # 优先从配置读取
        if 'num_classes' in ckpt:
            return ckpt['num_classes']
        # 否则从模型参数推断
        for key in ckpt['model_state_dict']:
            if 'class_embed.weight' in key:
                return ckpt['model_state_dict'][key].shape[0] - 1  # 去掉背景/无对象
        raise KeyError("Cannot determine num_classes from checkpoint")

    def _load_checkpoint(self, checkpoint_path: str, load_optim: bool = True):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self._init_model()
        self.model.load_state_dict(ckpt['model_state_dict'])
        if load_optim and 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if load_optim and 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt.get('epoch', 0)
        self.best_loss = ckpt.get('best_loss', float('inf'))
        if 'class_names' in ckpt:
            self.class_ids = ckpt['class_names']
        CLogging.LogToUser(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {self.start_epoch}")

    def _build_optimizer(self):
        cfg = self.config
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = cfg.OPTIMIZER.upper()
        if opt_name == "SGD":
            return optim.SGD(params, lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        if opt_name == "ADAM":
            return optim.Adam(params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        if opt_name == "ADAMW":
            return optim.AdamW(params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        if opt_name == "RMSPROP":
            return optim.RMSprop(params, lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER}")

    def _build_scheduler(self, optimizer, steps_per_epoch):
        cfg = self.config
        sched = cfg.LR_SCHEDULER.upper()
        if sched == "STEPLR":
            return StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)
        if sched == "MULTISTEPLR":
            milestones = [int(x) for x in cfg.LR_MILESTONES.split(',') if x.strip()]
            return MultiStepLR(optimizer, milestones=milestones, gamma=cfg.LR_GAMMA)
        if sched == "COSINEANNEALINGLR":
            return CosineAnnealingLR(optimizer, T_max=cfg.LR_T_MAX, eta_min=cfg.LR_ETA_MIN)
        if sched == "REDUCELRONPLATEAU":
            return ReduceLROnPlateau(optimizer, mode='min', factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE, verbose=True)
        if sched == "ONECYCLELR":
            total_steps = cfg.NUM_EPOCHS * steps_per_epoch
            return OneCycleLR(optimizer, max_lr=cfg.LEARNING_RATE, total_steps=total_steps, pct_start=cfg.LR_PCT_START)
        if sched == "FIXEDLR" or sched == "NONE":
            return None
        raise ValueError(f"Unsupported scheduler: {cfg.LR_SCHEDULER}")

    def Train(self, train_dataset, valid_dataset, balance_sample: bool = False):
        cfg = self.config
        # 创建 DataLoader (注意自定义 collate_fn)

        if balance_sample:
            batch_sampler = BalancedBatchSampler(
                dataset=train_dataset,
                batch_size=cfg.BATCH_SIZE,  # 8 张图，4 前 + 4 背
                drop_last=False  # 保留最后一个可能不足的 batch
            )

            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                collate_fn=collate_fn  # 您的 collate 函数
            )
            val_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_VAL,
                                      collate_fn=collate_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_TRAIN,
                                      collate_fn=collate_fn)
            val_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_VAL,
                                      collate_fn=collate_fn)
        if hasattr(train_dataset, 'class_names'):
            if len(self.class_ids) != len(train_dataset.class_names):
                CLogging.LogWarning("Model class num is not match with train_dataset.class_names! ")
            self.class_ids = train_dataset.class_names
        else:
            self.class_ids = [i for i in range(cfg.NUM_CLASSES + 1)]
        # 构建损失函数组件
        matcher = HungarianMatcher()
        criterion, self.weight_dict = build_criterion(cfg.NUM_CLASSES, matcher)
        self.criterion = criterion.to(self.device)
        self.optimizer = self._build_optimizer()
        steps_per_epoch = len(train_loader)
        self.scheduler = self._build_scheduler(self.optimizer, steps_per_epoch)

        # AMP 设置 (可选)
        from torch.amp import autocast, GradScaler
        scaler = GradScaler('cuda') if self.device.type == 'cuda' else None

        loss_data = {"train_loss": [], "val_loss": []}
        best_model_idx = 1
        best_val_loss = float('inf')

        for epoch in range(self.start_epoch, cfg.NUM_EPOCHS):
            self.model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} train")
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                targets = [{'masks': m.to(self.device), 'labels': l.to(self.device)}
                           for m, l in zip(batch['masks'], batch['labels'])]
                self.optimizer.zero_grad()
                if scaler is not None:
                    with autocast('cuda'):
                        outputs = self.model(images)
                        loss_dict = criterion(outputs, targets)
                        losses = sum(loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict.keys())
                    scaler.scale(losses).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(images)
                    loss_dict = criterion(outputs, targets)
                    losses = sum(loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict.keys())
                    losses.backward()
                    self.optimizer.step()
                train_loss += losses.item()
                progress_bar.set_postfix({'loss': losses.item()})
            avg_train_loss = train_loss / len(train_loader)
            loss_data["train_loss"].append(avg_train_loss)
            self.writer.add_scalar('Loss/train_total', avg_train_loss, epoch)

            # 验证（使用统一的 evaluate 函数）
            # val_metrics = evaluate(self.model, val_loader, self.device, cfg.NUM_CLASSES, criterion)
            val_metrics = self.Evaluate(val_loader, criterion=criterion, epoch=epoch)
            avg_val_loss = val_metrics['loss']
            # miou = val_metrics['mIoU']

            loss_data["val_loss"].append(avg_val_loss)
            self.writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
            # self.writer.add_scalar('Metrics/mIoU', miou, epoch)

            # miou = val_metrics['mIoU']
            # CLogging.LogToUser(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, mIoU = {miou:.4f}")
            CLogging.LogToUser(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            # 保存 checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_loss': best_val_loss,
                'num_classes': cfg.NUM_CLASSES,
                'class_names': self.class_ids,
            }
            torch.save(checkpoint, os.path.join(self.model_export_folder, f"mask2former_epoch_{epoch+1}.pth"))

            # 更新最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_idx = epoch + 1

            # 调度器步进
            if cfg.LR_SCHEDULER.upper() == "REDUCELRONPLATEAU":
                self.scheduler.step(avg_val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

        # 训练结束，导出结果
        self._export_train_result(loss_data, best_model_idx)

    def _export_train_result(self, loss_data, best_model_idx):
        # 复制最佳模型和最新模型
        model_prefix = "mask2former_epoch_"
        best_src = os.path.join(self.model_export_folder, f"{model_prefix}{best_model_idx}.pth")
        best_dst = os.path.join(self.folder_export, f"{model_prefix}{best_model_idx}_best.pth")
        shutil.copy(best_src, best_dst)

        last_epoch = len(loss_data["train_loss"])
        last_src = os.path.join(self.model_export_folder, f"{model_prefix}{last_epoch}.pth")
        last_dst = os.path.join(self.folder_export, f"{model_prefix}{last_epoch}_latest.pth")
        shutil.copy(last_src, last_dst)

        # 保存损失曲线
        loss_txt_path = os.path.join(self.folder_export, "loss.txt")
        with open(loss_txt_path, "w") as f:
            f.write("Epoch\tTrainLoss\tValLoss\n")
            for i in range(len(loss_data["train_loss"])):
                train_val = loss_data["train_loss"][i]
                val_val = loss_data["val_loss"][i] if i < len(loss_data["val_loss"]) else "N/A"
                f.write(f"{i+1}\t{train_val:.6f}\t{val_val if val_val == 'N/A' else f'{val_val:.6f}'}\n")

        plt.figure()
        plt.plot(loss_data["train_loss"], label="train loss")
        plt.plot(loss_data["val_loss"], label="valid loss")
        plt.legend()
        plt.savefig(os.path.join(self.folder_export, "loss.png"))

        CLogging.LogToUser(f"Training complete. Best model at epoch {best_model_idx} with loss {loss_data['val_loss'][best_model_idx-1]:.4f}.")
        CLogging.LogToUser('Model saved.')

    def Evaluate(self, dataloader, criterion=None, epoch=0):
        """
        criterion: SetCriterion 实例，用于计算验证损失（可选）
        返回: dict with keys 'loss' (平均损失)
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            # 进度条：与 train 风格一致，描述改为 valid
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} valid")
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                targets = [
                    {'masks': m.to(self.device), 'labels': l.to(self.device)}
                    for m, l in zip(batch['masks'], batch['labels'])
                ]
                outputs = self.model(images)

                if criterion is not None:
                    loss_dict = criterion(outputs, targets)
                    # 计算当前批次的总损失（加权和，与 train 一致）
                    batch_loss = sum(loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict.keys())
                    total_loss += batch_loss.item()
                    # 实时更新进度条显示当前批次损失
                    progress_bar.set_postfix({'loss': batch_loss.item()})

        avg_loss = total_loss / len(dataloader) if criterion is not None else 0.0
        return {'loss': avg_loss}

    def Predict(self, images, score_thresh=0.5, mask_thresh=0.5):
        """
        images: list of PIL Image or list of numpy (H,W,3)
        返回: list of list of dict, 每个 dict 包含 'mask', 'class_id', 'score'
        """
        self.model.eval()
        results = []

        transform = get_preprocessing_transform()

        for img in images:
            if isinstance(img, np.ndarray):
                img_rgb = img
            else:
                img_rgb = np.array(img.convert('RGB'))
            transformed = transform(image=img_rgb)
            img_tensor = transformed['image'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
            class_logits = outputs['class_logits'][0]
            mask_logits = outputs['mask_logits'][0]
            probs = F.softmax(class_logits, dim=-1)
            scores, pred_classes = probs.max(dim=-1)
            mask_probs = torch.sigmoid(mask_logits)
            mask_probs = F.interpolate(mask_probs.unsqueeze(1), size=img_rgb.shape[:2],
                                       mode='bilinear', align_corners=False).squeeze(1)
            num_classes = class_logits.shape[-1] - 1
            instance_results = []
            for q in range(len(scores)):
                cls_id = pred_classes[q].item()
                if cls_id == num_classes:
                    continue
                score = scores[q].item()
                if score < score_thresh:
                    continue
                mask = (mask_probs[q] > mask_thresh).cpu().numpy().astype(np.uint8)
                if mask.sum() < 100:
                    continue
                instance_results.append({'mask': mask, 'class_id': cls_id, 'score': score})
            # 简易去重（按IoU）
            instance_results.sort(key=lambda x: x['score'], reverse=True)
            keep = []
            for i, r in enumerate(instance_results):
                if not any(self._compute_iou(r['mask'], instance_results[j]['mask']) > 0.5 for j in keep):
                    keep.append(i)
            final = [instance_results[i] for i in keep]
            results.append(final)
        return results

    @staticmethod
    def _compute_iou(mask1, mask2):
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return inter / union if union > 0 else 0.0

if __name__ == "__main__":

    from Model.UNIMask2Former.dataset import SegmentationDataset
    from Model.UNIMask2Former.config import CConfigParameterUNIMask2Former

    config = CConfigParameterUNIMask2Former()
    # mode = "inference"

    checkpoint_path = r"/media/wqs/SOFT/weiqiushi/MaskRCNN/TorchvisionMaskRCNN/ModelTrain/uni-mask2former/20260608173807/mask2former_epoch_16_best.pth"
    image_path = r"/media/wqs/SOFT/weiqiushi/Project/UNI/Segmentation/sample_data/train/images/train_0000.png"

    config.UNI_CHECKPOINT = r"/media/wqs/SOFT/weiqiushi/Project/UNI/UNI2Model/UNI/pytorch_model.bin"
    config.PATCH_SIZE = 256
    config.NUM_CLASSES = 3
    config.OUTPUT_ROOT = r"/media/wqs/SOFT/weiqiushi/MaskRCNN/TorchvisionMaskRCNN/ModelTrain/uni-mask2former"
    config.NUM_EPOCHS = 100

    # mode = "train"
    mode = "inference"

    if mode == "train":
        config.DATA_ROOT = r"/media/wqs/SOFT/weiqiushi/Project/UNI/Segmentation/sample_data"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataset = SegmentationDataset(config.DATA_ROOT, split='train', patch_size=config.PATCH_SIZE)
        val_dataset = SegmentationDataset(config.DATA_ROOT, split='val', patch_size=config.PATCH_SIZE)

        model = ModelUNIMask2Former("train", config)
        model.Train(train_dataset, val_dataset)
    else:

        def visualize_results(image_path, results, output_dir):
            """可视化保存结果（可选）"""
            from PIL import ImageDraw
            import random

            org_img = Image.open(image_path).convert('RGB')
            img = Image.new('RGB', org_img.size, (0, 0, 0))
            draw = ImageDraw.Draw(img, 'RGBA')
            for res in results:
                # 随机颜色（半透明）
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)
                mask_pil = Image.fromarray((res['mask'] * 255).astype(np.uint8), mode='L')
                img.paste(color, mask=mask_pil)  # 叠加颜色
                # 可画出轮廓或标签
            output_path = Path(output_dir) / f"result_{Path(image_path).name}"
            img.save(output_path)
            print(f"Saved visualization to {output_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = ModelUNIMask2Former("inference", config, checkpoint_path=checkpoint_path)

        """读取图像并应用预处理，返回 (1, 3, H, W) 的 tensor"""
        image = np.array(Image.open(image_path).convert('RGB'))
        results = model.Predict([image])[0]

        out_path = os.path.join(os.getcwd(), "predict")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        visualize_results(image_path, results, out_path)
        a = 1


