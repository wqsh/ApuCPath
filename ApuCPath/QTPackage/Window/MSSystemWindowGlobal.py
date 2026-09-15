from QTPackage.Window.IconCode.MSSystemIcon import CSystemPathIconNormal, CSystemPathIconImageView
from enum import unique, Enum

import os

FOLDER_BIN = "bin"
FOLDER_CONFIG = "config"

if os.name == 'nt':
    EXE_APUCPATHCREATEDATASET = "ApuCPathCreateDataset.exe"
    EXE_MODELSEG = "ModelSegMain.exe"
elif os.name == 'posix':
    EXE_APUCPATHCREATEDATASET = "ApuCPathCreateDataset"
    EXE_MODELSEG = "ModelSegMain"
else:
    EXE_APUCPATHCREATEDATASET = None
    EXE_MODELSEG = None

@unique
class EnumSelectForDatasetFileType(Enum):

    ImageFile = 0
    ImageFolder = 1


class CSystemWindowSelectForDataset:
    FILE_TYPE = None
    PATH_IMAGE = ""
    PATH_MASK = ""

@unique
class EnumUIDatasetType(Enum):

    Train = 0
    Valid = 1
    Predict = 2
    Assist = 3

@unique
class EnumWindowState(Enum):

    Free = 0
    Running = 1
    Error = 2

LIST_IMAGE_WINDOW_STATE = [
    f"border-image: url({CSystemPathIconNormal.GREEN});",
    f"border-image: url({CSystemPathIconNormal.BLUE});",
    f"border-image: url({CSystemPathIconNormal.RED});",
]

LIST_CHECK_IMAGE = [
    f"border-image: url({CSystemPathIconNormal.RED});",
    f"border-image: url({CSystemPathIconNormal.GREEN});",
]

@unique
class EnumStackedWidget(Enum):
    ImageLabel = 0
    CreateDataset = 1
    TrainModel = 2
    Predict = 3
    ImageView = 4

LIST_STACKEDWIDGET_SUFFIX = ["", ".cfg_dataset", ".cfg_train", ".cfg_predict", ""]

@unique
class EnumModelType(Enum):
    UNIMask2Former = 0
    MaskRCNN = 1
    Unet = 2