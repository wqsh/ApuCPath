import torch
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

class MaskRCNNWithEdge(MaskRCNN):
    """
    带 Edge Agreement Head 的 Mask R-CNN。
    手动实现前向逻辑，避免重复计算，并添加边缘损失。
    """
    def __init__(self, config):
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
        self.config = config
        self.edge_loss_weight = getattr(config, 'EDGE_LOSS_WEIGHT', 0.1)
        self.edge_threshold = getattr(config, 'EDGE_THRESHOLD', 0.2)

        # 获取 mask head 中间特征图的通道数（兼容不同版本）
        in_channels = self.roi_heads.mask_head[0][0].in_channels

        # 添加边缘预测头
        self.edge_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 1, kernel_size=1, padding=0)
        )
        for layer in self.edge_head:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, images, targets=None):
        """
        训练时：计算所有标准损失 + 边缘损失，返回 loss_dict
        推理时：直接返回预测结果（无边缘头）
        """
        if self.training and targets is not None:
            # 1. 应用 transform，将 images list 转换为 ImageList 并标准化
            #    targets 也会相应调整 boxes 和 masks
            images, targets = self.transform(images, targets)  # images 现在是 ImageList

            # 2. Backbone 提取特征（现在 images.tensors 是 batch tensor）
            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = [features]

            # 3. RPN 生成 proposals 和损失
            proposals, proposal_losses = self.rpn(images, features, targets)

            # 4. ROI heads（包括 box 分类/回归 和 mask 预测）
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

            loss_dict = {}
            loss_dict.update(proposal_losses)
            loss_dict.update(detector_losses)

            # ---------- 边缘损失 ----------
            # 复用 roi_heads 中的 mask_roi_pool 和 mask_head 共享特征
            mask_features = self.roi_heads.mask_roi_pool(features, proposals, images.image_sizes)
            # 通过 mask_head 的前面部分得到共享特征 (与 mask 预测共享)
            x = self.roi_heads.mask_head(mask_features)

            edge_logits = self.edge_head(x)          # (N, 1, 14, 14)

            # 生成边缘 Ground Truth
            edge_targets_list = []
            valid_indices = []
            proposal_idx = 0
            for img_idx, (proposals_img, target) in enumerate(zip(proposals, targets)):
                if 'masks' not in target or target['masks'].numel() == 0:
                    proposal_idx += len(proposals_img)
                    continue
                gt_masks = target['masks']           # (num_gt, H, W)
                if gt_masks.dim() == 4:
                    gt_masks = gt_masks.squeeze(1)
                for i in range(len(proposals_img)):
                    if proposal_idx >= len(edge_logits):
                        break
                    # 简化匹配：假设 proposal 与 gt_mask 按顺序一一对应
                    if i < len(gt_masks):
                        mask_gt = gt_masks[i]
                    else:
                        proposal_idx += 1
                        continue
                    # 缩放 GT mask 到 14x14
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
        else:
            # 推理模式：直接调用父类 forward（不带边缘头）
            return super().forward(images)

    def _compute_edge_ground_truth(self, mask_2d):
        """
        从二值 mask 计算边缘 GT（Sobel 梯度幅值 + 阈值）
        mask_2d: shape (H, W), 值为 0 或 1
        """
        if not isinstance(mask_2d, torch.Tensor):
            mask_2d = torch.tensor(mask_2d, dtype=torch.float32)
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=mask_2d.dtype, device=mask_2d.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=mask_2d.dtype, device=mask_2d.device).view(1,1,3,3)
        input_ = mask_2d.unsqueeze(0).unsqueeze(0)
        grad_x = F.conv2d(input_, sobel_x, padding=1)
        grad_y = F.conv2d(input_, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        grad_mag = grad_mag / (grad_mag.max() + 1e-6)
        edge = (grad_mag > self.edge_threshold).float()
        return edge.squeeze()


# ==================== 管理类（与之前相同，只是导入上面的模型） ====================
from .maskrcnn_model import ModelMaskRCNN  # 假设父类在这个模块中

class ModelMaskRCNNWithEdge(ModelMaskRCNN):
    """
    专门用于 MaskRCNNWithEdge 模型的管理类。
    继承自 ModelMaskRCNN，复用了所有训练/预测逻辑。
    """
    def _init_model(self):
        """使用 MaskRCNNWithEdge 替代原始 MaskRCNN"""
        self.model = MaskRCNNWithEdge(self.parameter_maskrcnn).to(self.device)
        from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
        CLogging.LogToUser("ModelMaskRCNNWithEdge: Initialized with Edge Agreement Head.", log2file=False)

    def _load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """重写以兼容没有 edge_head 的旧模型"""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if not hasattr(self, 'model') or self.model is None:
            self._init_model()

        has_edge_head = any('edge_head' in k for k in ckpt['model_state_dict'].keys())
        if not has_edge_head:
            CLogging.LogToUser(
                "Loaded checkpoint does not contain edge_head weights. "
                "Edge head will be randomly initialized (strict=False).",
                log2file=False
            )
            strict = False

        missing_keys, unexpected_keys = self.model.load_state_dict(ckpt['model_state_dict'], strict=strict)
        if missing_keys:

            CLogging.LogToUser(f"Missing keys: {missing_keys}", log2file=False)
        if unexpected_keys:
            CLogging.LogToUser(f"Unexpected keys: {unexpected_keys}", log2file=False)

        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                CLogging.LogToUser(f"Failed to load optimizer state: {e}", log2file=False)
        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler') and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e:
                CLogging.LogToUser(f"Failed to load scheduler state: {e}", log2file=False)

        self.start_epoch = ckpt.get('epoch', 0)
        self.class_ids = ckpt.get("class_names", ["BG"])
        self.best_loss = ckpt.get('best_loss', float('inf'))
        CLogging.LogToUser(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {self.start_epoch}")