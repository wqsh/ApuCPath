"""
Mask Scoring R-CNN Implementation (Full Version)
Paper: https://arxiv.org/abs/1903.00241
Reference: https://github.com/zjhuang22/maskscoring_rcnn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import roi_align
from torchvision.ops import nms as torchvision_nms

class MaskIoUHead(nn.Module):
    """
    MaskIoU Head. Input: ROI pooled features (14x14x256) + predicted mask (14x14x1) (downsampled from 28x28).
    Output: mask IoU score (num_classes x num_classes). Architecture: Conv x3 + Global AvgPool + FCx2.
    """
    def __init__(self, num_classes, in_channels=256, mask_size=14):
        super().__init__()
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes * num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, roi_features, mask_logits, pos_labels):
        """Forward pass.
        Args:
            roi_features: (N, C, 14, 14)
            mask_logits: (N, num_classes, 28, 28)
            pos_labels: (N,)
        """
        N, C, H, W = roi_features.shape
        # Select and down sample mask for each proposal
        selected_masks = mask_logits[torch.arange(N, device=roi_features.device), pos_labels]
        selected_masks = selected_masks.unsqueeze(1)
        selected_masks = F.interpolate(selected_masks, size=(H, W), mode='bilinear', align_corners=False)
        # Concat and process
        concat = torch.cat([roi_features, selected_masks], dim=1)
        conv_out = self.conv_layers(concat)
        pooled = self.global_pool(conv_out).view(N, -1)
        fc_out = self.fc(pooled)
        mask_iou_pred = fc_out.view(N, self.num_classes, self.num_classes)
        # Return (pred_label, pred_label) values
        return mask_iou_pred[torch.arange(N, device=roi_features.device), pos_labels, pos_labels]

class MaskScoringRCNN(MaskRCNN):
    def __init__(self, config):
        backbone = resnet_fpn_backbone(config.BACKBONE_NAME, pretrained=config.PRETRAINED_BACKBONE)
        super().__init__(
            backbone=backbone, num_classes=config.NUM_CLASSES,
            min_size=config.MIN_SIZE, max_size=config.MAX_SIZE,
            rpn_pre_nms_top_n_train=config.RPN_PRE_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=config.RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_train=config.RPN_POST_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_test=config.RPN_POST_NMS_TOP_N_TEST,
            rpn_nms_thresh=config.RPN_NMS_THRESH,
            rpn_fg_iou_thresh=config.RPN_FG_IOU_THRESH,
            rpn_bg_iou_thresh=config.RPN_BG_IOU_THRESH,
            box_score_thresh=config.BOX_SCORE_THRESH,
            box_nms_thresh=config.BOX_NMS_THRESH,
            box_detections_per_img=config.BOX_DETECTIONS_PER_IMG,
            mask_roi_pool_size=config.MASK_ROI_POOL_SIZE,
        )
        self.config = config
        self.mask_iou_loss_weight = getattr(config, 'MASK_IOU_LOSS_WEIGHT', 1.0)
        in_channels = self.roi_heads.mask_head[-1].in_channels
        self.mask_iou_head = MaskIoUHead(
            num_classes=config.NUM_CLASSES,
            in_channels=in_channels,
            mask_size=config.MASK_ROI_POOL_SIZE
        )

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            images, targets = self.transform(images, targets)
            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor): features = [features]
            proposals, proposal_losses = self.rpn(images, features, targets)
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
            # Extract mask features and predictions
            mask_features = self.roi_heads.mask_roi_pool(features, proposals, images.image_sizes)
            x = self.roi_heads.mask_head(mask_features)
            mask_logits = self.roi_heads.mask_predictor(x)
            # Match proposals to GT
            pos_indices, pos_labels, matched_gt_indices = self._match_proposals_to_gt(proposals, targets)
            device = next(self.parameters()).device
            if len(pos_indices) > 0:
                pos_mask_features = mask_features[pos_indices]
                pos_mask_logits = mask_logits[pos_indices]
                pos_labels_tensor = torch.tensor(pos_labels, device=device)
                mask_iou_pred = self.mask_iou_head(pos_mask_features, pos_mask_logits, pos_labels_tensor)
                mask_iou_targets = self._compute_mask_iou_targets(
                    proposals, targets, images.image_sizes, mask_logits,
                    pos_indices, pos_labels, matched_gt_indices
                )
                detector_losses['loss_mask_iou'] = F.mse_loss(mask_iou_pred, mask_iou_targets) * self.mask_iou_loss_weight
            else:
                detector_losses['loss_mask_iou'] = torch.tensor(0.0, device=device)
            loss_dict = {}
            loss_dict.update(proposal_losses)
            loss_dict.update(detector_losses)
            return loss_dict
        else:
            return super().forward(images)

    def _match_proposals_to_gt(self, proposals, targets):
        pos_indices, pos_labels, matched_gt_idxs = [], [], []
        global_idx = 0
        for img_idx, (props, target) in enumerate(zip(proposals, targets)):
            if 'boxes' not in target or len(target['boxes']) == 0:
                global_idx += len(props)
                continue
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            for prop in props:
                ious = self._bbox_iou(prop[:4].unsqueeze(0), gt_boxes)
                max_iou, best_gt = ious.max(dim=1)
                if max_iou.item() > 0.5:
                    pos_indices.append(global_idx)
                    pos_labels.append(gt_labels[best_gt.item()].item())
                    matched_gt_idxs.append(best_gt.item())
                global_idx += 1
        return pos_indices, pos_labels, matched_gt_idxs

    def _compute_mask_iou_targets(self, proposals, targets, image_sizes, mask_logits, pos_indices, pos_labels, matched_gt_idxs):
        targets_list = []
        global_idx = 0
        for img_idx, (props, target) in enumerate(zip(proposals, targets)):
            if 'masks' not in target or target['masks'].numel() == 0:
                global_idx += len(props)
                continue
            gt_masks = target['masks']
            if gt_masks.dim() == 4: gt_masks = gt_masks.squeeze(1)
            for prop_idx, prop in enumerate(props):
                if global_idx in pos_indices:
                    pos_idx_in_list = pos_indices.index(global_idx)
                    gt_idx = matched_gt_idxs[pos_idx_in_list]
                    gt_mask = gt_masks[gt_idx]
                    box = prop[:4].int().tolist()
                    x1, y1, x2, y2 = [max(0, v) for v in box]
                    cropped = gt_mask[y1:y2+1, x1:x2+1]
                    if cropped.numel() == 0:
                        targets_list.append(0.0)
                    else:
                        cropped = cropped.float().unsqueeze(0).unsqueeze(0)
                        resized_gt = F.interpolate(cropped, size=(28, 28), mode='bilinear', align_corners=False).squeeze()
                        label = pos_labels[pos_idx_in_list]
                        pred_mask_logit = mask_logits[global_idx, label]
                        pred_mask = (pred_mask_logit > 0.5).float()
                        intersection = (pred_mask * resized_gt).sum()
                        union = pred_mask.sum() + resized_gt.sum() - intersection
                        targets_list.append((intersection / (union + 1e-6)).item())
                global_idx += 1
        return torch.tensor(targets_list, device=mask_logits.device)

    @staticmethod
    def _bbox_iou(boxes1, boxes2):
        boxes1 = boxes1.float(); boxes2 = boxes2.float()
        if boxes1.dim() == 1: boxes1 = boxes1.unsqueeze(0)
        if boxes2.dim() == 1: boxes2 = boxes2.unsqueeze(0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, None] + area2 - inter + 1e-6)
        return iou

    def predict_with_mask_iou(self, images, score_thresh=0.5, nms_thresh=0.5, detections_per_img=100, use_soft_nms=False):
        """Inference with optional Soft-NMS (disabled by default)."""
        self.eval()
        results = []
        with torch.no_grad():
            detections = super().forward(images)
            for i, det in enumerate(detections):
                if len(det['boxes']) == 0:
                    results.append(det)
                    continue
                boxes = det['boxes']; scores = det['scores']; labels = det['labels']; masks = det['masks']
                if use_soft_nms:
                    # Apply Soft-NMS (pseudo-code, requires proper implementation)
                    # keep = self._soft_nms(boxes, scores, nms_thresh)
                    # boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
                    raise NotImplementedError("Soft-NMS logic needs to be implemented based on your specific data.")
                else:
                    keep = torchvision_nms(boxes, scores, nms_thresh)
                    boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
                if len(scores) > detections_per_img:
                    topk = scores.argsort(descending=True)[:detections_per_img]
                    boxes, scores, labels, masks = boxes[topk], scores[topk], labels[topk], masks[topk]
                if len(boxes) == 0:
                    results.append(det)
                    continue
                # Compute mask iou scores efficiently
                mask_iou_scores = self._compute_mask_iou_from_masks(images[i], boxes, labels, masks)
                if mask_iou_scores is None: continue
                calibrated_scores = scores * mask_iou_scores
                final_mask = calibrated_scores > score_thresh
                boxes, labels, masks, calibrated_scores, mask_iou_scores = (
                    boxes[final_mask], labels[final_mask], masks[final_mask],
                    calibrated_scores[final_mask], mask_iou_scores[final_mask]
                )
                results.append({
                    'boxes': boxes, 'scores': calibrated_scores, 'labels': labels,
                    'masks': masks, 'mask_scores': mask_iou_scores
                })
        return results

    def _compute_mask_iou_from_masks(self, image, boxes, labels, masks):
        """Efficiently compute mask iou scores by reusing backbone and ROI features."""
        with torch.no_grad():
            if len(boxes) == 0:
                return torch.tensor([], device=next(self.parameters()).device)
            from torchvision.models.detection.image_list import ImageList
            image = image.to(next(self.parameters()).device)
            if image.dim() == 3: image = image.unsqueeze(0)
            image_list = ImageList(image, [image.shape[-2:]])
            features = self.backbone(image_list.tensors)
            if isinstance(features, torch.Tensor): features = [features]
            proposals_list = [boxes]
            mask_features = self.roi_heads.mask_roi_pool(features, proposals_list, image_list.image_sizes)
            # Use provided masks (already from mask head)
            # Downsample masks from 28x28 to 14x14 to match roi_features
            selected_masks = masks[torch.arange(len(boxes), device=boxes.device), labels, :, :].unsqueeze(1)
            selected_masks_resized = F.interpolate(selected_masks.float(), size=(mask_features.shape[2], mask_features.shape[3]), mode='bilinear', align_corners=False)
            concat_features = torch.cat([mask_features, selected_masks_resized], dim=1)
            conv_out = self.mask_iou_head.conv_layers(concat_features)
            pooled = self.mask_iou_head.global_pool(conv_out).view(conv_out.size(0), -1)
            fc_out = self.mask_iou_head.fc(pooled)
            mask_iou_pred = fc_out.view(-1, self.mask_iou_head.num_classes, self.mask_iou_head.num_classes)
            batch_idx = torch.arange(len(boxes), device=boxes.device)
            return mask_iou_pred[batch_idx, labels, labels]