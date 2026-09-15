from ApuCellGlobal.MSApuCellExternalPackge import *

@unique
class EnumFilterType(Enum):

    GrateThan = 1
    LessThan = 2

LIST_VISUAL_FILE_TYPE = [None, "png", "jpg", "txt", "tif", "xml", "tiff", "json", "svs"]
LIST_VISUAL_FILE_TYPE_SUFFIX = [None, ".png", ".jpg", ".txt", ".tif", ".xml", ".tiff", ".json", ".svs"]
@unique
class EnumVisualFileType(Enum):

    PNG = 1
    TXT = 3
    JPG = 2
    TIF = 4
    XML = 5
    TIFF = 6
    JSON = 7
    SVS = 8

LIST_VISUAL_FILE_IMAGE_IDX = [EnumVisualFileType.JPG.value,
                              EnumVisualFileType.PNG.value,
                              EnumVisualFileType.TIF.value,
                              EnumVisualFileType.TIFF.value,
                              # EnumVisualFileType.SVS.value,
                              ]

LIST_DATA_FILE_TYPE = [None, "info", "label", 'image', 'pkl']
LIST_DATA_FILE_TYPE_SUFFIX = [None, ".info", ".label", ".image", ".pkl"]
@unique
class EnumDataFileType(Enum):

    INFO = 1
    LABEL = 2
    IMAGE = 3
    PKL = 4

LIST_FILE_FLAG_PREFIX = ["MASK_"]
LIST_FILE_FLAG_SUFFIX = ["_MASK"]

@unique
class EnumFileFlag(Enum):

    MASK = 0