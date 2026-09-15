import os

import os
from typing import Union, Tuple, Optional


class CConfigParameterUnet:
    """Unet 配置类，所有属性大写，支持文本文件 key=value 读写，布尔值保存为 0/1"""

    # ========== 模型架构 ==========
    IMG_SIZE = 256
    NUM_CLASSES = 3

    # ========== 优化器 ==========
    OPTIMIZER: str = "SGD"  # SGD, Adam, AdamW, RMSprop
    LEARNING_RATE: float = 0.001  # 初始学习率
    MOMENTUM: float = 0.9  # SGD动量
    WEIGHT_DECAY: float = 0.0001  # 权重衰减
    # ADAM_BETAS: Tuple[float, float] = (0.9, 0.999)      # Adam的beta参数
    # ADAM_EPS: float = 1e-8                              # Adam的eps

    # ========== 学习率调度器 ==========
    LR_SCHEDULER: str = "None"  # None, StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
    LR_STEP_SIZE: int = 3  # StepLR步长
    LR_GAMMA: float = 0.1  # StepLR/MultiStepLR衰减因子
    LR_MILESTONES: str = "5,10,15"  # MultiStepLR里程碑（逗号分隔）
    LR_T_MAX: int = 10  # CosineAnnealingLR周期
    LR_ETA_MIN: float = 0  # CosineAnnealingLR最小学习率
    LR_FACTOR: float = 0.1  # ReduceLROnPlateau衰减因子/
    LR_PATIENCE: int = 3  # ReduceLROnPlateau耐心值
    LR_PCT_START: float = 0.3  # OneCycleLR上升阶段比例

    # ========== 数据处理 ==========
    BATCH_SIZE: int = 4  # 批大小
    # NUM_WORKERS: int = 4                                # DataLoader工作线程数
    SHUFFLE_TRAIN: bool = True  # 训练集是否打乱
    SHUFFLE_VAL: bool = False  # 验证集是否打乱
    # PIN_MEMORY: bool = True                             # 是否使用pin_memory
    # PERSISTENT_WORKERS: bool = True                     # 是否复用worker

    # ========== 训练流程 ==========
    NUM_EPOCHS: int = 12  # 总训练轮数
    # EARLY_STOP_PATIENCE: Optional[int] = 5              # 早停耐心值（None表示不早停）
    # EARLY_STOP_MIN_DELTA: float = 0.001                 # 早停最小改善阈值
    # GRAD_CLIP_NORM: Optional[float] = 1.0               # 梯度裁剪最大范数（None表示不裁剪）
    # USE_AMP: bool = False                               # 是否使用自动混合精度
    SEED: int = 42  # 随机种子

    # ========== 日志与保存 ==========
    OUTPUT_ROOT: str = "./runs"  # 输出根目录

    USE_GPU = 1

    def __init__(self, config_file: Optional[str] = None):
        if config_file:
            self.load_from_file(config_file)

    def load_from_file(self, filepath: str):
        """从文本文件加载配置，布尔值支持0/1或true/false"""
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip().upper()
                val = val.strip()
                if hasattr(self, key):
                    setattr(self, key, self._parse_value(val))

    def save_to_file(self, filepath: str):
        """将配置保存到文本文件，布尔值保存为0/1，按定义顺序输出"""
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath, 'w') as f:
            # 遍历类字典，保持定义顺序（Python 3.7+ 保证插入顺序）
            for attr, value in self.__class__.__dict__.items():
                # 只输出大写字母开头的属性（且不是方法）
                if attr.isupper() and not callable(value):
                    # 获取当前实例的实际值（可能被修改过）
                    real_value = getattr(self, attr)
                    if isinstance(real_value, bool):
                        real_value = '1' if real_value else '0'
                    elif isinstance(real_value, (tuple, list)):
                        real_value = ','.join(str(v) for v in real_value)
                    elif real_value is None:
                        real_value = "None"
                    f.write(f"{attr} = {real_value}\n")

    @staticmethod
    def _parse_value(value: str) -> Union[bool, int, float, str, tuple, None]:
        """解析字符串值，支持0/1转为bool，逗号分隔转为tuple"""
        value = value.strip()
        if value.upper() == "NONE":
            return None
        if value == '0':
            return False
        if value == '1':
            return True
        if value.upper() == "TRUE":
            return True
        if value.upper() == "FALSE":
            return False
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        if ',' in value:
            parts = [p.strip() for p in value.split(',')]
            converted = []
            for p in parts:
                if p == '0':
                    converted.append(False)
                elif p == '1':
                    converted.append(True)
                else:
                    try:
                        converted.append(int(p) if '.' not in p else float(p))
                    except ValueError:
                        converted.append(p)
            return tuple(converted)
        return value

    def get_output_dir(self, time_str: str) -> str:
        """生成带时间戳的输出文件夹路径"""
        return os.path.join(self.OUTPUT_ROOT, time_str)


if __name__ == "__main__":
    config_mask_rcnn = CConfigParameterMaskRCNN()
    config_mask_rcnn.save_to_file(os.path.join(os.getcwd(), "parameter.txt"))