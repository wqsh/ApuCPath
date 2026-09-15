DATASET_NAME = "ApuCPathDataset"

from ApuCellGlobal.WSIProcess.MSDataWSIDataset import EnumDatasetType, EnumTypeExportImage

from enum import Enum, unique

import os

@unique
class EnumTypeExportPad(Enum):

    All = 0
    Org = 1

@unique
class EnumMaskType(Enum):

    Image = 0
    JSON = 1

class CConfigCreateDataset:
    A1_TYPE_DATASET:EnumDatasetType = EnumDatasetType.Predict

    A2_LIST_IMAGE_PATH:str = ""
    A3_LIST_MASK_PATH:str = ""
    A4_FOLDER_EXPORT:str = os.getcwd()
    A5_NAME_DATASET:str = DATASET_NAME
    A6_INPUT_MASK_TYPE:EnumMaskType=EnumMaskType.JSON

    B1_IMAGE_DIM:int = 512

    B2_MASK_RATIO_MIN:float = 0.2
    B3_MASK_RATIO_MAX:float = 1.0

    B4_EXPORT_TYPE:EnumTypeExportImage = EnumTypeExportImage.Pkl
    B5_EXPORT_PAD_TYPE:EnumTypeExportPad = EnumTypeExportPad.All

class CDataPack:

    cfg = CConfigCreateDataset()