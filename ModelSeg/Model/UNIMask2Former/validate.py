import torch
import numpy as np
import torch.nn.functional as F


def calculate_iou(pred_mask, true_mask, num_classes):
    """计算平均IoU（忽略背景类0）"""
    ious = []
    for cls in range(1, num_classes + 1):
        pred = (pred_mask == cls)
        true = (true_mask == cls)
        intersection = np.logical_and(pred, true).sum()
        union = np.logical_or(pred, true).sum()
        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union
        ious.append(iou)
    return np.nanmean(ious)


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)  # (B, 3, H, W)
            masks_batch = batch['masks']  # 可能是 (B, num_instances, H, W) 或 list
            labels_batch = batch['labels']  # 类似

            B = images.shape[0]
            for b in range(B):
                # 获取单个样本的图像
                img = images[b:b + 1]  # (1, 3, H, W)

                # 获取该样本的 masks 和 labels
                if isinstance(masks_batch, torch.Tensor):
                    masks = masks_batch[b]  # (num_instances, H, W)
                    labels = labels_batch[b]  # (num_instances,)
                else:
                    masks = masks_batch[b]
                    labels = labels_batch[b]

                h, w = masks.shape[1], masks.shape[2]
                # 初始化真实语义图（全背景）
                true_semantic = torch.zeros(h, w, dtype=torch.long, device=device)

                # 遍历该样本的每个实例
                num_instances = masks.shape[0]
                for i in range(num_instances):
                    # 获取标签值
                    lbl = labels[i]
                    if torch.is_tensor(lbl):
                        lbl_val = lbl.item() if lbl.numel() == 1 else lbl[0].item()
                    else:
                        lbl_val = int(lbl)

                    # 获取第 i 个实例的掩码，并确保是二维 (H, W)
                    mask_i = masks[i]
                    if mask_i.dim() == 3:
                        # 可能是 (1, H, W) 或 (H, W, 1)
                        mask_i = mask_i.squeeze()
                    # 现在 mask_i 应为 (H, W)
                    if mask_i.dim() != 2:
                        raise ValueError(f"Expected 2D mask, got shape {mask_i.shape}")

                    # 将掩码区域赋值为对应标签（重叠区域后覆盖）
                    true_semantic[mask_i > 0.5] = lbl_val

                # 模型预测
                outputs = model(img)
                mask_logits = outputs['mask_logits']  # (1, Q, H_feat, W_feat)
                class_logits = outputs['class_logits']  # (1, Q, num_classes+1)

                # 上采样预测 mask 到原始图像尺寸
                pred_masks = F.interpolate(mask_logits, size=(h, w), mode='bilinear', align_corners=False)
                class_probs = class_logits.softmax(dim=-1)  # (1, Q, num_classes+1)
                class_probs = class_probs[:, :, :-1]  # 去掉 no_object 类
                # 计算每个像素的类别得分: (1, num_classes, H, W)
                scores = torch.einsum('bqchw,bqc->bchw', pred_masks.unsqueeze(2), class_probs)
                pred_semantic = scores.argmax(dim=1).squeeze(0)  # (H, W)

                all_preds.append(pred_semantic.cpu().numpy())
                all_targets.append(true_semantic.cpu().numpy())

    # 计算平均 IoU
    valid_ious = [calculate_iou(p, t, num_classes) for p, t in zip(all_preds, all_targets)]
    valid_ious = [iou for iou in valid_ious if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    return {'mIoU': miou}