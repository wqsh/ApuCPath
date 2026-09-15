from ApuCellGlobal.MSApuCellExternalPackge import *
@unique
class EnumPadType(Enum):

    Org = 0
    Horizontal = 1
    Vertical = 2
    Diagonal = 3


class CDataMaskRCNNPredictDataSet:

    def __init__(self, folder_output, row_size:int=512, col_size:int=512):
        self.folder_output:str = folder_output
        self.list_path_input_image:list = []
        self.list_file_name:list = []
        self.list_name:list = []
        self.pad_type_list = [EnumPadType.Org, EnumPadType.Horizontal, EnumPadType.Vertical, EnumPadType.Diagonal]
        self.row_size:int = row_size
        self.col_size:int = col_size
        self.half_row_size:int = int(self.row_size / 2)
        self.half_col_size:int = int(self.col_size / 2)