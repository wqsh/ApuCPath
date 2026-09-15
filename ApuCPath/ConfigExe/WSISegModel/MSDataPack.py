from ConfigExe.WSISegModel.MSSystem import NAME_COCO_MODEL_PARAMETER, FOLDER_MODEL, FOLDER_UNI_MODEL, FOLDER_UNI, NAME_UNI_CHECKPOINT
import os
from enum import Enum, unique

@unique
class EnumTypeModelInit(Enum):

    Scratch = 0
    Finetune = 1

class EnumTrainType(Enum):
    Org = 0
    KFold = 1

@unique
class EnumTypeModelType(Enum):
    MaskRCNN = 0
    MaskRCNNBGClassifier = 1
    Unet = 2
    UNIMask2Former = 3

class EnumTypeNegSample(Enum):

    Null = 0
    MiniBatchKMeans = 1
    Random = 2

class CConfigWSISegModel:
    A1_TYPE_FLOW:int = 1
    A2_PATH_DATASET_NORMAL:str = ""
    A3_PATH_DATASET_TRAIN:str = ""
    A4_PATH_DATASET_VALID:str = ""
    A5_PATH_DATASET_PREDICT:str = ""
    A6_FOLDER_OUTPUT:str = os.getcwd()
    A7_SKIP_CHECK_MASK:bool = False
    A8_PREDICT_SKIP_EXIST:bool = False

    B1_IMAGE_DIM:int = 512
    B2_IMAGE_HEIGHT:int = 512

    B3_CLASSES_NUM:int = 2
    B4_BATCH_SIZE:int = 4
    B5_EPOCH_NUM:int = 20
    B6_LEARNING_RATE:float = 0.000001
    B7_TRAIN_TYPE:int=0
    B8_K_FOLD:int = 5
    B9_VALID_PERCENT:float = 0.2

    C1_MODEL_INIT:EnumTypeModelInit = EnumTypeModelInit.Scratch
    C2_MODEL_PATH:str = ""#os.path.join(os.getcwd(), FOLDER_MODEL, NAME_COCO_MODEL_PARAMETER)
    C3_MODEL_TYPE:EnumTypeModelType = EnumTypeModelType.UNIMask2Former
    C4_MODEL_ADD_NEGATIVE:bool = False
    C5_MODEL_NEGATIVE_SAMPLE:EnumTypeNegSample = EnumTypeNegSample.MiniBatchKMeans
    C5_MODEL_NEGATIVE_SAMPLE_CLUSTER:int = 100
    C6_BALANCE_SAMPLE:bool = True

    D1_1_FILTER_MASK_SCORE:float = 0.7
    D2_EXPORT_PREDICT_HEATMAP:bool = False
    D3_MIN_AREA:int = 40000
    D4_MERGE_SCORE:float = 0.5
    D5_MERGE_OVERLAP:bool = True
    D6_EXPORT_PAD_MASK:bool = False

    E1_TRAIN_IMAGE_LIST_FILE:str = ""
    E2_VALID_IMAGE_LIST_FILE:str = ""
    E3_PREDICT_IMAGE_LIST_FILE:str = ""

    F1_REPLY_TRAIN:int = 0
    F2_FOLDER_REPLY_TRAIN:str = ""
    F3_PATH_REPLY_TRAIN_MODEL:str = ""

    G1_BACKBONE_1_RPN_PRE_NMS_TOP_N_TRAIN: int = 2000                 # RPN训练时NMS前保留候选框数
    G1_BACKBONE_2_RPN_POST_NMS_TOP_N_TRAIN: int = 2000                # RPN训练时NMS后保留候选框数
    G1_BACKBONE_3_RPN_FG_IOU_THRESH: float = 0.7                      # RPN正样本IoU阈值
    G1_BACKBONE_4_RPN_BG_IOU_THRESH: float = 0.3                      # RPN负样本IoU阈值

    G2_BACKBONE_1_RPN_NMS_THRESH: float = 0.7                         # RPN NMS阈值
    G2_BACKBONE_2_MASK_ROI_POOL_SIZE: int = 14                        # Mask RoI池化输出尺寸

    G3_BACKBONE_1_RPN_PRE_NMS_TOP_N_TEST: int = 1000                  # RPN推理时NMS前保留候选框数
    G3_BACKBONE_2_RPN_POST_NMS_TOP_N_TEST: int = 1000                 # RPN推理时NMS后保留候选框数
    G3_BACKBONE_3_BOX_SCORE_THRESH: float = 0.5                      # 检测框置信度阈值
    G3_BACKBONE_4_BOX_NMS_THRESH: float = 0.5                         # 检测框NMS阈值
    G3_BACKBONE_5_BOX_DETECTIONS_PER_IMG: int = 100                   # 每张图像最多检测框数

    G4_PATH_UNI_CHECKPOINT:str = os.path.join(os.getcwd(), FOLDER_UNI_MODEL, FOLDER_UNI, NAME_UNI_CHECKPOINT)

    Z1_USE_GPU:int = 1

class CDataPack:
    cfg = CConfigWSISegModel()