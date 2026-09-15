import os
import json
from datetime import datetime


class CConfigParameterUNIMask2Former:
    """配置类，用于存放 Mask2Former 训练/推理所需的参数"""

    def __init__(self):
        # ---------- 数据集相关 ----------
        self.DATA_ROOT = "./simulated_data"  # 数据集根目录（包含 train/val 子文件夹）
        self.NUM_CLASSES = 3  # 前景类别数量（背景为0）
        self.PATCH_SIZE = 1024  # 训练时裁剪尺寸（图像和mask同步裁剪）

        # ---------- 模型相关 ----------
        self.UNI_CHECKPOINT = "/path/to/uni/pytorch_model.bin"  # UNI ViT-L 预训练权重路径
        self.BACKBONE_FROZEN = True  # 是否冻结 UNI 编码器参数
        self.D_MODEL = 256  # Transformer 隐藏层维度
        self.NUM_QUERIES = 100  # 解码器 queries 数量

        # ---------- 损失权重 ----------
        self.weight_dict = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}

        # ---------- 训练参数 ----------
        self.BATCH_SIZE = 2  # 批次大小（根据显存调整）
        self.NUM_EPOCHS = 100  # 总训练轮数
        self.LEARNING_RATE = 1e-4  # 初始学习率
        self.WEIGHT_DECAY = 1e-4  # 权重衰减系数

        # ---------- 优化器 ----------
        self.OPTIMIZER = "ADAMW"  # 可选：SGD, ADAM, ADAMW, RMSPROP
        self.MOMENTUM = 0.9  # SGD momentum（仅当 OPTIMIZER=SGD/RMSPROP 时生效）

        # ---------- 学习率调度器 ----------
        self.LR_SCHEDULER = "None"  # 可选：STEPLR, MULTISTEPLR, COSINEANNEALINGLR, REDUCELRONPLATEAU, ONECYCLELR, FIXEDLR/NONE
        self.LR_STEP_SIZE = 30  # STEPLR 的步长
        self.LR_GAMMA = 0.5  # STEPLR / MULTISTEPLR 的衰减因子
        self.LR_MILESTONES = "50,80"  # MULTISTEPLR 的里程碑（逗号分隔）
        self.LR_T_MAX = 100  # COSINEANNEALINGLR 的周期（通常等于 NUM_EPOCHS）
        self.LR_ETA_MIN = 1e-6  # COSINEANNEALINGLR 的最小学习率
        self.LR_FACTOR = 0.5  # REDUCELRONPLATEAU 的衰减因子
        self.LR_PATIENCE = 5  # REDUCELRONPLATEAU 的耐心值
        self.LR_PCT_START = 0.3  # ONECYCLELR 的上升比例

        # ---------- 硬件与输出 ----------
        self.USE_GPU = True  # 是否使用 GPU（否则 CPU）
        self.OUTPUT_ROOT = "./mask2former_output"  # 输出根目录（内部会自动创建时间戳子目录）
        self.SHUFFLE_TRAIN = True  # 训练集是否打乱
        self.SHUFFLE_VAL = False  # 验证集是否打乱

        # ---------- 其他 ----------
        self.SEED = 42  # 随机种子（可选）

    def save_to_file(self, file_path: str):
        """将配置保存到文本文件，便于记录实验参数"""
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f"{key} = {value}\n")
        print(f"Configuration saved to {file_path}")

    def load_from_file(self, file_path: str):
        """从文件加载配置（若需要）"""
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # 尝试转换类型
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                setattr(self, key, value)
        print(f"Configuration loaded from {file_path}")


# 使用示例
if __name__ == "__main__":
    cfg = CConfigParameterUNIMask2Former()
    # 修改必要参数
    cfg.DATA_ROOT = "/mnt/data/segmentation"
    cfg.UNI_CHECKPOINT = "/mnt/models/uni_vit_large.pth"
    cfg.NUM_CLASSES = 5
    cfg.BATCH_SIZE = 4
    cfg.NUM_EPOCHS = 50
    cfg.save_to_file("./config_mask2former.txt")