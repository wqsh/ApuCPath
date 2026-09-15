from ApuCellGlobal.MSApuCellExternalPackge import *

@unique
class EnumApuCellTFMMITypeShape(Enum):

    Normal = 0

    Circle = 2
    RectAngle = 3
    Ellipse =4
    Num = 5

class CDataMMIXMLBoundaryInfo:

    def __init__(self, left:float, right:float, top:float, bottom:float):

        self.left:float = left
        self.right:float = right
        self.top:float = top
        self.bottom:float = bottom

class CDataMMIXMLInfo:

    def __init__(self):

        self.stage_inserts:CDataMMIXMLStageInsertsInfo = None

        self.slide_info:list = []
        self.slide_idx:list = []

        self.group_info:list = []
        self.group_idx:list = []

        self.shape_info:list = []

class CDataMMIXMLStageInsertsInfo:

    def __init__(self, boundary_info, slide_num:int):

        self.boundary_info = boundary_info
        self.slide_num:int = slide_num

class CDataMMIXMLSlideInfo:

    def __init__(self, name, boundary_info:CDataMMIXMLBoundaryInfo):


        self.boundary_info:CDataMMIXMLBoundaryInfo = boundary_info

        self.name:str = name
        self.list_shape_idx:list = []

class CDataMMIXMLGroupInfo:

    def __init__(self, name):

        self.name:str = name
        self.list_shape_idx:list = []


class CDataMMIXMLShapeInfo:

    def __init__(self, shape_type:int, cell_parent:str, boundary_info:CDataMMIXMLBoundaryInfo):

        self.type:int = shape_type
        self.cell_parent:str = cell_parent
        self.from_group_idx:int = 0
        self.from_slide_idx:int = 0
        self.point:list = []
        self.boundary_info:CDataMMIXMLBoundaryInfo = boundary_info