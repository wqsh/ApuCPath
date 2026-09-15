import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from Model.maskrcnn.maskrcnn_model import ModelMaskRCNN

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

class MaskRCNNWBGClassifier(MaskRCNN):
    """
    继承自 torchvision 的 MaskRCNN，增加背景分类分支。
    - 训练时：背景图像只计算背景分类损失，不经过 RPN/ROI heads/edge head。
              背景图像的梯度可以选择是否阻断到 backbone（默认阻断）。
    - 推理时：背景图像直接返回空结果，完全跳过后续计算。
    """
    def __init__(self, config, detach_bg_gradient=True):
        """
        Args:
            config: 配置对象，需包含原有 MaskRCNN 所需的所有参数（如 backbone, rpn, roi_heads 等）。
            detach_bg_gradient: 是否阻断背景图像的梯度回传到 backbone。
        """

        # 构建 backbone
        backbone = resnet_fpn_backbone(
            config.BACKBONE_NAME,
            pretrained=config.PRETRAINED_BACKBONE
        )
        super().__init__(
            backbone=backbone,
            num_classes=config.NUM_CLASSES,
            min_size=config.MIN_SIZE,
            max_size=config.MAX_SIZE,
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
        self.detach_bg_gradient = detach_bg_gradient

        # 确定背景分类器的输入通道数
        # torchvision 的 backbone 通常是 FPN，out_channels 默认为 256
        bg_in_channels = getattr(self.backbone, 'out_channels', 256)
        self.bg_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(bg_in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)          # 2类: 背景(0), 前景(1)
        )
        self._init_bg_classifier()

        self.bg_weight = getattr(config, 'BG_LOSS_WEIGHT', 0.1)

    def _init_bg_classifier(self):
        for m in self.bg_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_background_feature(self, features):
        """
        从 backbone 输出的特征字典中获取用于背景分类的特征图。
        优先使用 'pool' 键（若有），否则取最后一个特征图（最高层）。
        """
        if isinstance(features, torch.Tensor):
            return features
        if 'pool' in features:
            return features['pool']
        # torchvision 默认返回 OrderedDict，键为 '0','1','2','3' 或 'p2'~'p5'
        # 取最后一个（最高层）特征图
        return list(features.values())[-1]

    def forward(self, images, targets=None):
        # ==================== 训练模式 ====================
        if self.training and targets is not None:
            # 1. 图像预处理（缩放、归一化，并调整 targets 中的 boxes/masks）
            images, targets = self.transform(images, targets)

            # 2. Backbone 提取特征（所有图像都经过）
            features = self.backbone(images.tensors)
            last_feat = self._get_background_feature(features)  # (batch, C, H', W')

            # 3. 生成背景标签：boxes 为空则为背景(0)，否则前景(1)
            bg_targets = torch.tensor([
                0 if t['boxes'].shape[0] == 0 else 1
                for t in targets
            ], device=last_feat.device)

            # ---- 梯度隔离：背景图像不更新 backbone ----
            if self.detach_bg_gradient:
                bg_indices = (bg_targets == 0).nonzero(as_tuple=True)[0]
                if bg_indices.numel() > 0:
                    # 原地切断背景部分的梯度，前景部分梯度正常流动
                    last_feat[bg_indices].detach_()

            # 背景分类预测与损失
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

            return loss_dict

        # ==================== 推理模式 ====================
        else:
            original_image_sizes = [img.shape[-2:] for img in images]
            # 应用 transform（无 target）
            transformed_images, _ = self.transform(images, None)
            features = self.backbone(transformed_images.tensors)
            last_feat = self._get_background_feature(features)

            # 背景分类预测
            bg_logits = self.bg_classifier(last_feat)
            bg_probs = F.softmax(bg_logits, dim=1)
            is_background = bg_probs[:, 0] > 0.5   # 阈值可调

            bg_indices = [i for i, pred in enumerate(is_background) if pred]
            fg_indices = [i for i, pred in enumerate(is_background) if not pred]

            # 前景图像走完整前向（复用已提取的特征，避免重复 transform）
            fg_predictions = {}
            if fg_indices:
                # 构造前景 ImageList
                fg_tensors = transformed_images.tensors[fg_indices]
                fg_image_sizes = [transformed_images.image_sizes[i] for i in fg_indices]
                fg_images = ImageList(fg_tensors, fg_image_sizes)

                # 提取前景特征
                fg_features = {}
                for k, v in features.items():
                    fg_features[k] = v[fg_indices]

                # RPN 推理（targets=None，只返回 proposals）
                proposals, _ = self.rpn(fg_images, fg_features, None)
                # ROI heads 推理（返回检测结果列表）
                detections, _ = self.roi_heads(fg_features, proposals, fg_images.image_sizes, None)
                detections = self.transform.postprocess(detections, fg_images.image_sizes, original_image_sizes)
                for orig_idx, pred in zip(fg_indices, detections):
                    fg_predictions[orig_idx] = pred

            # 构造最终输出（背景图像填充空结果）
            results = []
            device = transformed_images.tensors.device
            for i in range(len(images)):
                if i in bg_indices:
                    h, w = original_image_sizes[i]
                    results.append({
                        'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
                        'labels': torch.zeros((0,), dtype=torch.int64, device=device),
                        'scores': torch.zeros((0,), dtype=torch.float32, device=device),
                        'masks': torch.zeros((0, h, w), dtype=torch.float32, device=device),
                    })
                else:
                    results.append(fg_predictions[i])
            return results

class ModelMaskRCNNBGClassifier(ModelMaskRCNN):
    def _init_model(self):
        self.model = MaskRCNNWBGClassifier(self.parameter_maskrcnn).to(self.device)
        CLogging.LogToUser("MaskRCNNWBGClassifier initialized.")

    def _load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        # 同原来的逻辑，但可以自动检测是否缺少 bg_classifier 并设置 strict=False
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if not hasattr(self, 'model') or self.model is None:
            self._init_model()
        has_bg = any('bg_classifier' in k for k in ckpt['model_state_dict'].keys())
        if not has_bg:
            CLogging.LogToUser("Checkpoint has no bg_classifier, loading with strict=False")
            strict = False
        super()._load_checkpoint(checkpoint_path)  # 调用父类加载方法