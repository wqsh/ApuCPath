import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

from Model.maskrcnn.config import CConfigParameterMaskRCNN
from Model.utils import BalancedBatchSampler
from Model.maskrcnn.dataset import img_to_tensor_predict

import torch
import torch.optim as optim

from torch.optim.lr_scheduler import (StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def collate_fn(batch):
    """将批次中的图像和目标打包成模型需要的格式"""
    return tuple(zip(*batch))

def get_checkpoint_num_class(path_check_point:str, device=torch.device('cuda')):

    state_dict = torch.load(path_check_point, device)
    try:
        num_classes = state_dict["num_classes"]
        return num_classes
    except:
        # 尝试从分类头获取
        if 'roi_heads.box_predictor.cls_score.weight' in state_dict:
            cls_out_dim = state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
            return cls_out_dim  # 包含背景的总类别数
        elif 'roi_heads.box_predictor.bbox_pred.weight' in state_dict:
            bbox_out_dim = state_dict['roi_heads.box_predictor.bbox_pred.weight'].shape[0]
            # bbox_pred 输出维度 = num_classes * 4
            num_classes = bbox_out_dim // 4
            return num_classes
        else:
            raise KeyError("Can't find the number of classes from checkpoint.")
        
class ModelMaskRCNN:
    def __init__(self, mode:str, parameter_maskrcnn: CConfigParameterMaskRCNN, checkpoint_path: str = None, add_time_folder:bool=True, new_train:bool=True):
        """
        Args:
            config_maskrcnn: 配置类实例
            backbone_name: 若提供则覆盖配置中的 BACKBONE_NAME
            num_classes: 若提供则覆盖配置中的 NUM_CLASSES
            checkpoint_path: 预训练/断点权重路径（用于恢复训练）
        """
        if parameter_maskrcnn.USE_GPU:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')

        self.parameter_maskrcnn:CConfigParameterMaskRCNN = parameter_maskrcnn
        self.build(mode, checkpoint_path, add_time_folder, new_train)
    
    def build(self, mode:str, checkpoint_path:str=None, add_time_folder:bool=True, new_train:bool=True):


        assert mode in ['train', 'inference']
        if mode == "train":
            # 输出文件夹 (使用配置的 OUTPUT_ROOT，并在后面附加时间戳)
            folder_export = self.parameter_maskrcnn.OUTPUT_ROOT
            self.start_epoch = 0

            if not checkpoint_path is None:
                # Load pretrain model for new train
                if new_train:
                    #### have problem !!
                    self._load_checkpoint(checkpoint_path)
                    if add_time_folder:
                        self.folder_export = os.path.join(folder_export, CLogging.GetLocTimeStrForName())
                    else:
                        self.folder_export = folder_export
                else:
                    # rely train
                    self.parameter_maskrcnn.NUM_CLASSES = get_checkpoint_num_class(checkpoint_path, self.device)
                    self._load_checkpoint(checkpoint_path)
                    if os.path.splitext(checkpoint_path)[0].endswith("_best") or os.path.splitext(checkpoint_path)[0].endswith("_latest"):
                        self.folder_export = os.path.join(os.path.dirname(checkpoint_path))
                    else:
                        self.folder_export = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)))
            else:
                # Init Model
                self._init_model()
                if add_time_folder:
                    self.folder_export = os.path.join(folder_export, CLogging.GetLocTimeStrForName())
                else:
                    self.folder_export = folder_export

            os.makedirs(self.folder_export, exist_ok=True)

            self.parameter_maskrcnn.save_to_file(os.path.join(self.folder_export, "parameter.txt"))
            # TensorBoard
            log_dir = os.path.join(self.folder_export, "tensorboard_logs")
            self.writer = SummaryWriter(log_dir=log_dir)

            # 模型保存子文件夹
            self.model_export_folder = os.path.join(self.folder_export, "model")
            os.makedirs(self.model_export_folder, exist_ok=True)

            self.best_loss = float('inf')
            self.class_ids = [str(i) for i in range(self.parameter_maskrcnn.NUM_CLASSES)]

        elif mode == "inference":
            self.parameter_maskrcnn.NUM_CLASSES = get_checkpoint_num_class(checkpoint_path, self.device)
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """加载断点（模型、优化器、调度器状态等）"""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_classes_num = get_checkpoint_num_class(checkpoint_path, self.device)
        if checkpoint_classes_num != self.parameter_maskrcnn.NUM_CLASSES:
            CLogging.LogError("Checkpoint model class num is not match with setting! ")
        # 构建模型
        self._init_model()

        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt.get('epoch', 0)
        self.class_ids = ckpt.get("class_names", ["BG"])
        self.best_loss = ckpt.get('best_loss', float('inf'))
        CLogging.LogToUser(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {self.start_epoch}")

    def _init_model(self):
        # 构建模型
        backbone = resnet_fpn_backbone(self.parameter_maskrcnn.BACKBONE_NAME, pretrained=self.parameter_maskrcnn.PRETRAINED_BACKBONE)
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=self.parameter_maskrcnn.NUM_CLASSES,
            min_size=self.parameter_maskrcnn.MIN_SIZE,
            max_size=self.parameter_maskrcnn.MAX_SIZE,
            rpn_pre_nms_top_n_train=self.parameter_maskrcnn.RPN_PRE_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=self.parameter_maskrcnn.RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_train=self.parameter_maskrcnn.RPN_POST_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_test=self.parameter_maskrcnn.RPN_POST_NMS_TOP_N_TEST,
            rpn_nms_thresh=self.parameter_maskrcnn.RPN_NMS_THRESH,
            rpn_fg_iou_thresh=self.parameter_maskrcnn.RPN_FG_IOU_THRESH,
            rpn_bg_iou_thresh=self.parameter_maskrcnn.RPN_BG_IOU_THRESH,
            box_score_thresh=self.parameter_maskrcnn.BOX_SCORE_THRESH,
            box_nms_thresh=self.parameter_maskrcnn.BOX_NMS_THRESH,
            box_detections_per_img=self.parameter_maskrcnn.BOX_DETECTIONS_PER_IMG,
            mask_roi_pool_size=self.parameter_maskrcnn.MASK_ROI_POOL_SIZE,
        ).to(self.device)

    def _build_optimizer(self):
        """根据配置创建优化器"""
        cfg = self.parameter_maskrcnn
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = cfg.OPTIMIZER.upper()
        if opt_name == "SGD":
            return optim.SGD(params, lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                             weight_decay=cfg.WEIGHT_DECAY)
        # if opt_name == "ADAM":
        #     return optim.Adam(params, lr=cfg.LEARNING_RATE, betas=cfg.ADAM_BETAS,
        #                       eps=cfg.ADAM_EPS, weight_decay=cfg.WEIGHT_DECAY)
        # if opt_name == "ADAMW":
        #     return optim.AdamW(params, lr=cfg.LEARNING_RATE, betas=cfg.ADAM_BETAS,
        #                        eps=cfg.ADAM_EPS, weight_decay=cfg.WEIGHT_DECAY)
        if opt_name == "RMSPROP":
            return optim.RMSprop(params, lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                                 weight_decay=cfg.WEIGHT_DECAY)
        raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER}")

    def _build_scheduler(self, optimizer, steps_per_epoch):
        """根据配置创建学习率调度器，返回 scheduler 或 None"""
        cfg = self.parameter_maskrcnn
        sched = cfg.LR_SCHEDULER.upper()
        if sched == "STEPLR":
            return StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)
        if sched == "MULTISTEPLR":
            milestones = [int(x) for x in cfg.LR_MILESTONES.split(',') if x.strip()]
            return MultiStepLR(optimizer, milestones=milestones, gamma=cfg.LR_GAMMA)
        if sched == "COSINEANNEALINGLR":
            return CosineAnnealingLR(optimizer, T_max=cfg.LR_T_MAX, eta_min=cfg.LR_ETA_MIN)
        if sched == "REDUCELRONPLATEAU":
            return ReduceLROnPlateau(optimizer, mode='min', factor=cfg.LR_FACTOR,
                                     patience=cfg.LR_PATIENCE, verbose=True)
        if sched == "ONECYCLELR":
            total_steps = cfg.NUM_EPOCHS * steps_per_epoch
            return OneCycleLR(optimizer, max_lr=cfg.LEARNING_RATE,
                              total_steps=total_steps, pct_start=cfg.LR_PCT_START)
        if sched == "FIXEDLR" or sched == "NONE":
            return None
        raise ValueError(f"Unsupported scheduler: {cfg.LR_SCHEDULER}")

    def Train(self, train_dataset, valid_dataset, balance_sample:bool=False):
        """
        完整训练流程
        Returns:
            loss_data: 包含 train_loss 和 val_loss 列表的字典
        """
        cfg = self.parameter_maskrcnn

        # 假设 train_dataset 已定义

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
            valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_VAL, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_TRAIN, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_VAL, collate_fn=collate_fn)

        # 优化器与调度器
        self.optimizer = self._build_optimizer()
        steps_per_epoch = len(train_loader)
        self.scheduler = self._build_scheduler(self.optimizer, steps_per_epoch)

        loss_data = {"train_loss": [], "val_loss": []}
        best_model_idx = 0
        best_model_loss = float('inf')

        num_epochs = cfg.NUM_EPOCHS
        for epoch in range(self.start_epoch, num_epochs):
            # ---------- 训练阶段 ----------
            self.model.train()
            train_loss_dict = {}
            total_loss = 0.0
            
            # 使用 tqdm 显示进度条
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} train", position=0, leave=False)
            for batch_idx, (images, targets) in enumerate(train_loader):

                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # with amp.autocast(enabled=cfg.USE_AMP):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                # if cfg.GRAD_CLIP_NORM is not None:
                #     scaler.unscale_(self.optimizer)
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.GRAD_CLIP_NORM)
                self.optimizer.step()

                total_loss += losses.item()
                for k, v in loss_dict.items():
                    train_loss_dict[k] = train_loss_dict.get(k, 0.0) + v.item()

                # 更新进度条
                train_loader_tqdm.update(1)  # 前进1步
                # 更新进度条显示的 loss
                train_loader_tqdm.set_postfix({'loss': f'{losses.item():.4f}', **{k: f'{v.item():.4f}' for k, v in loss_dict.items()}})

            avg_train_loss = total_loss / len(train_loader)
            self.writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
            for k, v in train_loss_dict.items():
                self.writer.add_scalar(f'Loss/train_{k}', v / len(train_loader), epoch)

            CLogging.LogToUser(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} " +
                               ', '.join(f'{k}: {v/len(train_loader):.4f}' for k, v in train_loss_dict.items()),
                               log2file=False)
            loss_data["train_loss"].append(avg_train_loss)

            # ---------- 保存 checkpoint ----------
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_loss': best_model_loss,
                'class_names': train_dataset.class_names,
                'num_classes': len(train_dataset.class_names),
            }
            torch.save(checkpoint, os.path.join(self.model_export_folder, f"maskrcnn_epoch_{epoch+1}.pth"))
            
            # ---------- 验证阶段（按 VAL_INTERVAL 执行） ----------
            val_loss = None

            self.model.train()
            val_loss_dict = {}
            total_val_loss = 0.0

            # 2. 冻结所有 BatchNorm 层，让它们表现得像 eval() 一样（不使用当前 batch 的统计量）
            def freeze_bn_and_dropout(module):
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0  # 直接关掉 Dropout
                    # 或者调用 module.eval()，但 Dropout 的 eval 其实就是 p=0
            self.model.apply(freeze_bn_and_dropout)
            # 3. 计算验证 Loss（依然关闭梯度，省显存加速）
            # 4. 切回 train() 继续训练（这会自动解冻上面被 freeze 的 BN）
            # 注意：如果后面还要继续训练，下一个 epoch 开始前执行 model.train() 即可恢复正常训练模式。
            # 如果只是做验证，后面不再训练，可以不执行下面这行。
            # model.train()

            with torch.no_grad():
                # 使用 tqdm 显示进度条
                valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} valid", position=0, leave=False)
                for images, targets in valid_loader:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    total_val_loss += losses.item()
                    for k, v in loss_dict.items():
                        val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v.item()

                    # 更新进度条
                    valid_loader_tqdm.update(1)  # 前进1步
                    # 更新进度条显示的 loss
                    valid_loader_tqdm.set_postfix({'loss': f'{losses.item():.4f}', **{k: f'{v.item():.4f}' for k, v in loss_dict.items()}})

            val_loss = total_val_loss / len(valid_loader)
            self.writer.add_scalar('Loss/val_total', val_loss, epoch)
            for k, v in val_loss_dict.items():
                self.writer.add_scalar(f'Loss/val_{k}', v / len(valid_loader), epoch)

            CLogging.LogToUser(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f} " +
                                ', '.join(f'{k}: {v/len(valid_loader):.4f}' for k, v in val_loss_dict.items()),
                                log2file=False)
            loss_data["val_loss"].append(val_loss)

            # 更新最佳模型
            if epoch == 0:
                best_model_loss = val_loss
                best_model_idx = epoch + 1
            else:
                if val_loss < best_model_loss:
                    best_model_loss = val_loss
                    best_model_idx = epoch + 1

            # 学习率调度器步进（注意 ReduceLROnPlateau 需要 val_loss）
            if cfg.LR_SCHEDULER.upper() == "REDUCELRONPLATEAU" and val_loss is not None:
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
        # 训练结束，导出最终结果（最佳模型和损失曲线）
        self._export_train_result(loss_data, best_model_idx)

    def _export_train_result(self, loss_data, best_model_idx):
        """导出最佳模型、最新模型以及损失曲线文本"""
        # 复制最佳模型
        best_src = os.path.join(self.model_export_folder, f"maskrcnn_epoch_{best_model_idx}.pth")
        best_dst = os.path.join(self.folder_export, f"maskrcnn_epoch_{best_model_idx}_best.pth")
        shutil.copy(best_src, best_dst)

        # 复制最新模型（最后一个epoch）
        last_epoch = len(loss_data["train_loss"])
        last_src = os.path.join(self.model_export_folder, f"maskrcnn_epoch_{last_epoch}.pth")
        last_dst = os.path.join(self.folder_export, f"maskrcnn_epoch_{last_epoch}_latest.pth")
        shutil.copy(last_src, last_dst)

        # 保存损失表格
        loss_txt_path = os.path.join(self.folder_export, "loss.txt")
        with open(loss_txt_path, "w") as f:
            f.write("Epoch\tTrainLoss\tValLoss\n")
            for i in range(len(loss_data["train_loss"])):
                train_val = loss_data["train_loss"][i]
                val_val = loss_data["val_loss"][i] if i < len(loss_data["val_loss"]) else "N/A"
                f.write(f"{i+1}\t{train_val:.6f}\t{val_val if val_val == 'N/A' else f'{val_val:.6f}'}\n")
        best_loss = loss_data["val_loss"][best_model_idx - 1]
    
        plt.figure()
        plt.plot(loss_data["train_loss"], label="train loss")
        plt.plot(loss_data["val_loss"], label="valid loss")
        plt.legend()
        plt.savefig(os.path.join(self.folder_export, "loss.png"))

        CLogging.LogToUser(f"Training complete. Best model at epoch {best_model_idx} with loss {best_loss:.4f}.",
                           log2file=False)
        CLogging.LogToUser('Training complete. Model saved.')

    # 可选的推理方法
    def Predict(self, images):

        assert len(images) <= self.parameter_maskrcnn.BATCH_SIZE
  
        self.model.eval()
        with torch.no_grad():
            images_tensor = [img_to_tensor_predict(img).to(self.device) for img in images]
            end_idx = len(images_tensor)
            while len(images_tensor) < self.parameter_maskrcnn.BATCH_SIZE:
                images_tensor.append(img_to_tensor_predict(np.zeros(images[0].shape, dtype=images[0].dtype)).to(self.device))
            predictions = self.model(images_tensor)
        return predictions
