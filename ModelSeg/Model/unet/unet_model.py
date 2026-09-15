import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from Model.unet.config import CConfigParameterUnet
from Model.utils import BalancedBatchSampler
from Model.unet.dataset import img_to_tensor_predict
from torch.optim.lr_scheduler import (StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR)

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

import os
from tqdm import tqdm
import numpy as np
import shutil

class DoubleConv(nn.Module):
    """(Conv3x3 + BN + ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """下采样：最大池化 + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """上采样：双线性插值 + 跳跃连接拼接 + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """
        x1: 来自下层的特征图（需要上采样）
        x2: 来自编码器的跳跃连接特征图
        """
        x1 = self.up(x1)
        # 处理尺寸不一致（由于padding可能导致微小差异）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet, self).__init__()
        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 解码器
        self.up1 = Up(1024 + 512, 512)   # 拼接后通道 = 1024 (来自down4) + 512 (来自down3)
        self.up2 = Up(512 + 256, 256)    # 拼接后通道 = 512 + 256
        self.up3 = Up(256 + 128, 128)    # 拼接后通道 = 256 + 128
        self.up4 = Up(128 + 64, 64)      # 拼接后通道 = 128 + 64

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        # 解码路径（带跳跃连接）
        x = self.up1(x5, x4)  # 输入x5(1024)和跳跃x4(512)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

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

# -------------------- 辅助函数 --------------------
def calculate_iou(pred, target, num_classes=3):
    """
    pred: [B, H, W] 预测的类别索引
    target: [B, H, W] 真实标签
    返回每个类别的 IoU 列表 (背景、圆形、矩形)
    """
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            ious.append(float('nan'))  # 该类别在 batch 中未出现
        else:
            ious.append((intersection / union).item())
    return ious

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = {0: 0.0, 1: 0.0, 2: 0.0}
    num_batches = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)          # [B, H, W]
            outputs = model(images)           # [B, 3, H, W]
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)   # [B, H, W]
            ious = calculate_iou(preds, masks, num_classes=3)
            for cls in range(3):
                if not np.isnan(ious[cls]):
                    total_iou[cls] += ious[cls]
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_iou = {cls: total_iou[cls] / num_batches for cls in total_iou}
    return avg_loss, avg_iou

def collate_fn(batch):
    """
    batch: list of tuples (image_tensor, mask_tensor)
    image_tensor shape: [C, H, W], mask_tensor shape: [H, W]
    """
    images = torch.stack([item[0] for item in batch], dim=0)  # [B, C, H, W]
    masks  = torch.stack([item[1] for item in batch], dim=0)  # [B, H, W]
    return images, masks

class ModelUnet:

    def __init__(self, mode: str, parameter_unet: CConfigParameterUnet, checkpoint_path: str = None, add_time_folder: bool = True, new_train: bool = True):
        """
        Args:
            config_maskrcnn: 配置类实例
            backbone_name: 若提供则覆盖配置中的 BACKBONE_NAME
            num_classes: 若提供则覆盖配置中的 NUM_CLASSES
            checkpoint_path: 预训练/断点权重路径（用于恢复训练）
        """
        if parameter_unet.USE_GPU:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')

        self.parameter_unet: CConfigParameterUnet = parameter_unet
        self.build(mode, checkpoint_path, add_time_folder, new_train)

    def build(self, mode:str, checkpoint_path:str=None, add_time_folder:bool=True, new_train:bool=True):

        assert mode in ['train', 'inference']
        if mode == "train":
            # 输出文件夹 (使用配置的 OUTPUT_ROOT，并在后面附加时间戳)
            folder_export = self.parameter_unet.OUTPUT_ROOT
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
                    self.parameter_unet.NUM_CLASSES = get_checkpoint_num_class(checkpoint_path, self.device)
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

            self.parameter_unet.save_to_file(os.path.join(self.folder_export, "parameter.txt"))
            # TensorBoard
            log_dir = os.path.join(self.folder_export, "tensorboard_logs")
            self.writer = SummaryWriter(log_dir=log_dir)

            # 模型保存子文件夹
            self.model_export_folder = os.path.join(self.folder_export, "model")
            os.makedirs(self.model_export_folder, exist_ok=True)

            self.best_loss = float('inf')
            self.class_ids = [str(i) for i in range(self.parameter_unet.NUM_CLASSES)]

        elif mode == "inference":
            self.parameter_unet.NUM_CLASSES = get_checkpoint_num_class(checkpoint_path, self.device)
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """加载断点（模型、优化器、调度器状态等）"""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_classes_num = get_checkpoint_num_class(checkpoint_path, self.device)
        if checkpoint_classes_num != self.parameter_unet.NUM_CLASSES:
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
        self.model = UNet(
            3,
            self.parameter_unet.NUM_CLASSES
        ).to(self.device)

    def Train(self, train_dataset, valid_dataset, balance_sample: bool = False):
        """
        完整训练流程
        Returns:
            loss_data: 包含 train_loss 和 val_loss 列表的字典
        """
        cfg = self.parameter_unet

        # 假设 train_dataset 已定义
        self.criterion = nn.CrossEntropyLoss()  # 忽略背景类？不用，直接计算所有像素
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
            valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_VAL,
                                      collate_fn=collate_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_TRAIN,
                                      collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_VAL,
                                      collate_fn=collate_fn)

        # 优化器与调度器
        self.optimizer = self._build_optimizer()
        steps_per_epoch = len(train_loader)
        self.scheduler = self._build_scheduler(self.optimizer, steps_per_epoch)

        loss_data = {"train_loss": [], "val_loss": []}
        best_model_idx = 0
        best_model_loss = float('inf')

        num_epochs = cfg.NUM_EPOCHS

        # -------------------- 训练循环 --------------------
        best_val_loss = float('inf')
        for epoch in range(self.start_epoch, num_epochs):
            self.model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} train")
            for batch_idx, (images, masks) in enumerate(progress_bar):

                images = images.to(self.device)  # images 已经是 [B,C,H,W] 张量
                masks = masks.to(self.device)  # masks 已经是 [B,H,W] 张量

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            avg_train_loss = train_loss / len(train_loader)
            loss_data["train_loss"].append(avg_train_loss)
            self.writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
            # ---------- 保存 checkpoint ----------
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                # 'best_loss': best_model_loss,
                'class_names': train_dataset.class_names,
                'num_classes': len(train_dataset.class_names),
            }
            torch.save(checkpoint, os.path.join(self.model_export_folder, f"unet_epoch_{epoch + 1}.pth"))

            total_loss = 0.0
            # total_iou = {0: 0.0, 1: 0.0, 2: 0.0}
            self.model.eval()
            num_batches = 0
            with torch.no_grad():
                progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} valid")
                for batch_idx, (images, masks) in enumerate(progress_bar):
                    images = images.to(self.device)
                    masks = masks.to(self.device)  # [B, H, W]
                    outputs = self.model(images)  # [B, 3, H, W]
                    loss = self.criterion(outputs, masks)
                    total_loss += loss.item()
                    # preds = torch.argmax(outputs, dim=1)  # [B, H, W]
                    # ious = calculate_iou(preds, masks, num_classes=3)
                    # for cls in range(3):
                    #     if not np.isnan(ious[cls]):
                    #         total_iou[cls] += ious[cls]
                    num_batches += 1
            # val_iou = {cls: total_iou[cls] / num_batches for cls in total_iou}
            # CLogging.LogToUser(f"Val IoU -> BG: {val_iou[0]:.4f}, Circle: {val_iou[1]:.4f}, Rectangle: {val_iou[2]:.4f}, mIoU: {(val_iou[0] + val_iou[1] + val_iou[2]) / 3:.4f}")

            avg_val_loss = total_loss / num_batches
            loss_data["val_loss"].append(avg_val_loss)
            self.writer.add_scalar('Loss/val_total', avg_val_loss, epoch)

            CLogging.LogToUser(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}", log2file=False)

            # 更新最佳模型
            if epoch == 0:
                best_model_loss = avg_val_loss
                best_model_idx = epoch + 1
            else:
                if avg_val_loss < best_model_loss:
                    best_model_loss = avg_val_loss
                    best_model_idx = epoch + 1

            # 学习率调度器步进（注意 ReduceLROnPlateau 需要 val_loss）
            if cfg.LR_SCHEDULER.upper() == "REDUCELRONPLATEAU" and avg_val_loss is not None:
                self.scheduler.step(avg_val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

        # 训练结束，导出最终结果（最佳模型和损失曲线）
        self._export_train_result(loss_data, best_model_idx)


    def _export_train_result(self, loss_data, best_model_idx):
        """导出最佳模型、最新模型以及损失曲线文本"""
        # 复制最佳模型
        model_str = "unet_epoch_"
        best_src = os.path.join(self.model_export_folder, f"{model_str}{best_model_idx}.pth")
        best_dst = os.path.join(self.folder_export, f"{model_str}{best_model_idx}_best.pth")
        shutil.copy(best_src, best_dst)

        # 复制最新模型（最后一个epoch）
        last_epoch = len(loss_data["train_loss"])
        last_src = os.path.join(self.model_export_folder, f"{model_str}{last_epoch}.pth")
        last_dst = os.path.join(self.folder_export, f"{model_str}{last_epoch}_latest.pth")
        shutil.copy(last_src, last_dst)

        # 保存损失表格
        loss_txt_path = os.path.join(self.folder_export, "loss.txt")
        with open(loss_txt_path, "w") as f:
            f.write("Epoch\tTrainLoss\tValLoss\n")
            for i in range(len(loss_data["train_loss"])):
                train_val = loss_data["train_loss"][i]
                val_val = loss_data["val_loss"][i] if i < len(loss_data["val_loss"]) else "N/A"
                f.write(f"{i + 1}\t{train_val:.6f}\t{val_val if val_val == 'N/A' else f'{val_val:.6f}'}\n")
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

        assert len(images) <= self.parameter_unet.BATCH_SIZE
        # images: 可以是 [B, C, H, W] 张量，或 list of [C,H,W] 张量
        images_tensor = torch.stack([img_to_tensor_predict(img).to(self.device) for img in images], dim=0)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            images_tensor = images_tensor.to(self.device)
            outputs = self.model(images_tensor)
            preds = torch.argmax(outputs, dim=1)

            for pred_idx, pred in enumerate(preds):
                one_pred = np.zeros((images[pred_idx].shape[0], images[pred_idx].shape[1], self.parameter_unet.NUM_CLASSES - 1), dtype=np.uint8)
                pred = np.array(pred.cpu())
                for cls in range(1, self.parameter_unet.NUM_CLASSES):
                    one_pred[:, :, cls - 1][pred == cls] = 1
                predictions.append(one_pred)
        return predictions

    def _build_optimizer(self):
        """根据配置创建优化器"""
        cfg = self.parameter_unet
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
        cfg = self.parameter_unet
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
