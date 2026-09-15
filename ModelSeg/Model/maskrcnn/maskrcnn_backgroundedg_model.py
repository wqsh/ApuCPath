import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
from Model.maskrcnn.maskrcnnedge_model import MaskRCNNWithEdge, ModelMaskRCNNWithEdge

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

class MaskRCNNWithEdgeAndBGClassifier(MaskRCNNWithEdge):
    """
    继承自 MaskRCNNWithEdge，增加背景分类分支。
    - 训练时：背景图像只计算背景分类损失，不经过 RPN/ROI heads/edge head。
              背景图像的梯度可以选择是否阻断到 backbone（默认阻断）。
    - 推理时：背景图像直接返回空结果，完全跳过后续计算。
    """
    def __init__(self, config, detach_bg_gradient=True):
        """
        Args:
            config: 配置对象，需包含原有 MaskRCNNWithEdge 所需的所有参数。
            detach_bg_gradient: 是否阻断背景图像的梯度回传到 backbone。
                                设为 True 时，只有前景图像会更新 backbone；
                                设为 False 时，背景图像也会参与 backbone 更新（可能干扰）。
        """
        super().__init__(config)
        self.detach_bg_gradient = detach_bg_gradient

        # 背景分类头：使用 FPN 最高层特征（'pool' 层，通道数 backbone.out_channels，通常为 256）
        bg_in_channels = self.backbone.out_channels
        self.bg_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(bg_in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)          # 2类: 背景(0), 前景(1)
        )
        self._init_bg_classifier()

        self.bg_weight = getattr(self.config, 'BG_LOSS_WEIGHT', 0.1)

    def _init_bg_classifier(self):
        for m in self.bg_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images, targets=None):
        # ==================== 训练模式 ====================
        if self.training and targets is not None:
            # 1. 应用 transform（图像缩放、归一化，调整 targets 中的 boxes/masks）
            images, targets = self.transform(images, targets)

            # 2. Backbone 提取特征（所有图像都经过 backbone）
            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = [features]

            # 3. 背景分类预测 + 生成标签
            last_feat = features['pool']                     # (batch, C, H/32, W/32)
            # 从 targets 获取背景标签：boxes 为空则为背景(0)，否则前景(1)
            bg_targets = torch.tensor([
                0 if t['boxes'].shape[0] == 0 else 1
                for t in targets
            ], device=last_feat.device)

            # ---- 梯度隔离（可选）：背景图像不更新 backbone ----
            if self.detach_bg_gradient:
                # 获取背景和前景的索引
                bg_indices = (bg_targets == 0).nonzero(as_tuple=True)[0]
                fg_indices = (bg_targets == 1).nonzero(as_tuple=True)[0]
                # 复制特征，背景部分阻断梯度
                controlled_feat = last_feat.clone()
                for idx in bg_indices:
                    controlled_feat[idx] = last_feat[idx].detach()
                bg_logits = self.bg_classifier(controlled_feat)   # (batch, 2)
            else:
                bg_logits = self.bg_classifier(last_feat)

            bg_loss = F.cross_entropy(bg_logits, bg_targets)
            loss_dict = {'loss_bg_classifier': bg_loss * self.bg_weight}

            # 如果没有前景图像，直接返回（不进入 RPN/ROI heads）
            if bg_targets.sum() == 0:   # 全部为背景
                return loss_dict

            # 4. 分离前景图像（构造独立的 ImageList 和 targets）
            fg_indices = (bg_targets == 1).nonzero(as_tuple=True)[0]
            fg_tensors = images.tensors[fg_indices]
            fg_image_sizes = [images.image_sizes[i] for i in fg_indices]
            fg_images = ImageList(fg_tensors, fg_image_sizes)
            fg_targets = [targets[i] for i in fg_indices]

            # 提取前景对应的特征图
            fg_features = {}
            for k, v in features.items():
                fg_features[k] = v[fg_indices]

            # 5. 对前景图像执行 RPN 和 ROI heads
            proposals, proposal_losses = self.rpn(fg_images, fg_features, fg_targets)
            detections, detector_losses = self.roi_heads(
                fg_features, proposals, fg_images.image_sizes, fg_targets
            )
            loss_dict.update(proposal_losses)
            loss_dict.update(detector_losses)

            # 6. 边缘损失（只在前景图像上计算）
            mask_features = self.roi_heads.mask_roi_pool(fg_features, proposals, fg_images.image_sizes)
            x = self.roi_heads.mask_head(mask_features)
            edge_logits = self.edge_head(x)          # (N, 1, 14, 14)

            # 生成边缘 GT（复用父类的逻辑，仅对前景）
            edge_targets_list = []
            valid_indices = []
            proposal_idx = 0
            for img_idx, (proposals_img, target) in enumerate(zip(proposals, fg_targets)):
                if 'masks' not in target or target['masks'].numel() == 0:
                    proposal_idx += len(proposals_img)
                    continue
                gt_masks = target['masks']           # (num_gt, H, W)
                if gt_masks.dim() == 4:
                    gt_masks = gt_masks.squeeze(1)
                for i in range(len(proposals_img)):
                    if proposal_idx >= len(edge_logits):
                        break
                    if i < len(gt_masks):
                        mask_gt = gt_masks[i]
                    else:
                        proposal_idx += 1
                        continue
                    if mask_gt.dim() == 2:
                        mask_gt = mask_gt.unsqueeze(0).unsqueeze(0)
                    else:
                        mask_gt = mask_gt.unsqueeze(0) if mask_gt.dim() == 3 else mask_gt
                    mask_resized = F.interpolate(
                        mask_gt.float(), size=(14, 14), mode='bilinear', align_corners=False
                    ).squeeze()
                    edge_gt = self._compute_edge_ground_truth(mask_resized)
                    edge_targets_list.append(edge_gt)
                    valid_indices.append(proposal_idx)
                    proposal_idx += 1

            if len(valid_indices) > 0:
                edge_logits_valid = edge_logits[valid_indices]
                edge_targets = torch.stack(edge_targets_list, dim=0).to(edge_logits.device)
                edge_loss = F.binary_cross_entropy_with_logits(
                    edge_logits_valid.squeeze(1), edge_targets
                )
                loss_dict['loss_edge'] = edge_loss * self.edge_loss_weight
            else:
                loss_dict['loss_edge'] = torch.tensor(0.0, device=edge_logits.device)

            return loss_dict

        # ==================== 推理模式 ====================
        else:
            original_image_sizes = [img.shape[-2:] for img in images]
            # 应用 transform（无 target）
            transformed_images, _ = self.transform(images, None)
            features = self.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = [features]
            last_feat = features['pool']
            bg_logits = self.bg_classifier(last_feat)
            bg_probs = F.softmax(bg_logits, dim=1)
            is_background = bg_probs[:, 0] > 0.5   # 阈值可调

            bg_indices = [i for i, pred in enumerate(is_background) if pred]
            fg_indices = [i for i, pred in enumerate(is_background) if not pred]

            # 前景图像走完整前向
            fg_predictions = {}
            if fg_indices:
                fg_images = [transformed_images[i] for i in fg_indices]
                fg_results = super().forward(fg_images)   # 调用父类推理模式
                for orig_idx, pred in zip(fg_indices, fg_results):
                    fg_predictions[orig_idx] = pred

            # 构造最终输出
            results = []
            device = images[0].device if len(images) > 0 else torch.device('cuda')
            for i in range(len(images)):
                if i in bg_indices:
                    h, w = original_image_sizes[i]
                    results.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'labels': torch.zeros((0,), dtype=torch.int64, device=device),
                        'scores': torch.zeros((0,), device=device),
                        'masks': torch.zeros((0, h, w), dtype=torch.uint8, device=device)
                    })
                else:
                    results.append(fg_predictions[i])
            return results

class ModelMaskRCNNWithEdgeBGClassifier(ModelMaskRCNNWithEdge):
    def _init_model(self):
        self.model = MaskRCNNWithEdgeAndBGClassifier(self.parameter_maskrcnn).to(self.device)
        CLogging.LogToUser("ModelMaskRCNNWithEdgeAndBGClassifier initialized.")

    def _load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        # 同原来的逻辑，但可以自动检测是否缺少 bg_classifier 并设置 strict=False
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if not hasattr(self, 'model') or self.model is None:
            self._init_model()
        has_bg = any('bg_classifier' in k for k in ckpt['model_state_dict'].keys())
        if not has_bg:
            CLogging.LogToUser("Checkpoint has no bg_classifier, loading with strict=False")
            strict = False
        super()._load_checkpoint(checkpoint_path, strict=strict)  # 调用父类加载方法