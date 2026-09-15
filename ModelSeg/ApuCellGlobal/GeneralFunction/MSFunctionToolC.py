import numpy as np

from ApuCellGlobal.MSApuCellExternalPackge import *

from ApuCellGlobal.GeneralSystem.MSSystemDLL import FOLDER_DLL, DLL_IMAGE_TOOL, SO_IMAGE_TOOL
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.GeneralSystem.MSLoggingInfo import ERROR_TO_USER

from ApuCellGlobal.Data.MSDataToolC import CDataShapeRectAngleInfo, CDataShapeRectAngleApex

class CFunctionToolImageC:

    __dll = None

    @staticmethod
    def Init():
        dll_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), FOLDER_DLL)
        if not os.path.exists(dll_path):
            dll_path = os.path.join(os.getcwd(), FOLDER_DLL)
        if os.name == 'nt':
            path_image_tool_dll = os.path.join(dll_path, DLL_IMAGE_TOOL)
        elif os.name == 'posix':
            path_image_tool_dll = os.path.join(dll_path, SO_IMAGE_TOOL)
        else:
            CLogging.LogError("Unknow system !")
        # CLogging.LogToUser(dll_path)
        if CFunctionToolImageC.__dll is None:
            if not os.path.exists(path_image_tool_dll):
                CLogging.LogError("[Error] Missing file : " + path_image_tool_dll)
            # 导入动态链接库c
            version = platform.python_version()
            if ((int(version.split('.')[0]) >= 3) and (int(version.split('.')[1]) > 7)):
                CFunctionToolImageC.__dll = CDLL(path_image_tool_dll, winmode=0)
            else:
                CFunctionToolImageC.__dll = cdll.LoadLibrary(path_image_tool_dll)
            # 这里是做函数的输入输出类型的定义
            CFunctionToolImageC.__dll.GetMaskLabelBoundarySeedPoint.argtype = (POINTER(c_longlong), POINTER(c_uint), c_ulonglong, c_uint, c_uint)

            CFunctionToolImageC.__dll.GetMaskLabelBoundarySeedPointInt.argtype = (POINTER(c_uint), POINTER(c_uint), c_ulonglong, c_uint, c_uint)

            CFunctionToolImageC.__dll.Adaboost.argtype = (POINTER(c_uint8), POINTER(c_uint), c_uint, c_uint)

            CFunctionToolImageC.__dll.GetShapePerimeter.argtype = (POINTER(c_uint8), POINTER(c_uint), c_uint, c_uint, c_uint, c_uint)

            CFunctionToolImageC.__dll.RemoveShape.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint)

            CFunctionToolImageC.__dll.BatchRemoveShapeBySize.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint, c_uint, c_uint)

            CFunctionToolImageC.__dll.CheckShapeSize.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint)

            CFunctionToolImageC.__dll.WaterShedFindHole.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint8)

            CFunctionToolImageC.__dll.CalculateShapeArea.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint)
            CFunctionToolImageC.__dll.CalculateShapeArea.restype = (c_uint)

            CFunctionToolImageC.__dll.BatchRemoveShapeByArea.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint)

            CFunctionToolImageC.__dll.BatchRemoveShapeByValidAreaRatio.argtype = (POINTER(c_uint8), c_uint, c_uint, c_double)

            CFunctionToolImageC.__dll.GetSeedShapeInfo.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint, POINTER(c_ulonglong), c_bool)

            CFunctionToolImageC.__dll.GetListSeedShapeInfo.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint, POINTER(c_uint), POINTER(c_ulonglong), c_bool)

            CFunctionToolImageC.__dll.GetLabelSeedPoint.argtype = (POINTER(c_uint), c_uint, c_uint, POINTER(c_uint), c_uint, POINTER(c_uint))

            CFunctionToolImageC.__dll.FillLabelListToValue.argtype = (POINTER(c_uint), POINTER(c_uint), c_uint, c_uint, POINTER(c_uint), c_uint, c_uint)

            CFunctionToolImageC.__dll.GetMaskLineBySeedPoint.argtype = (POINTER(c_uint8), POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint)

            CFunctionToolImageC.__dll.GetMaskLineByValue.argtype = (POINTER(c_uint8), POINTER(c_uint8), c_uint, c_uint, c_uint8)

            CFunctionToolImageC.__dll.ReplaceMatrixValue.argtype = (POINTER(c_uint8), POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint, c_uint, c_uint)

            CFunctionToolImageC.__dll.AddMatrixValue.argtype = (POINTER(c_uint8), POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint, c_uint, c_uint)

            CFunctionToolImageC.__dll.Mask2Label.argtype = (POINTER(c_uint32), POINTER(c_uint8), c_uint, c_uint)
            CFunctionToolImageC.__dll.Mask2Label.restype = (c_uint32)

            CFunctionToolImageC.__dll.FilterMatrixValueByLarge.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint8, c_bool)

            CFunctionToolImageC.__dll.FilterMatrixValueBySmall.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint8, c_bool)

            CFunctionToolImageC.__dll.SetMatrixValue.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint8)

            CFunctionToolImageC.__dll.FillPolygon.argtype = (POINTER(c_uint8), c_uint, c_uint, c_uint, c_uint, c_uint8, c_uint8)

            CFunctionToolImageC.__dll.CountValue.argtype = (POINTER(c_uint8), c_uint, c_uint, POINTER(c_ulonglong))
    @staticmethod
    def GetLabelBoundarySeedPoint(img_label:np.array, label_num:int)->np.array:

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        if img_label.dtype == np.uint64 or img_label.dtype == np.uint32 or img_label == np.uint:
            pass
        else:
            return None
        row_num = img_label.shape[0]
        col_num = img_label.shape[1]

        np_seed_point = np.zeros((1, (label_num + 1) * 2), dtype=c_uint)
        p_seed_point = np_seed_point.ctypes.data_as(POINTER(c_uint))
        if img_label.dtype == np.uint64:
            p_img_mask_label = img_label.ctypes.data_as(POINTER(c_uint64))
            CFunctionToolImageC.__dll.GetMaskLabelBoundarySeedPoint(p_img_mask_label, p_seed_point, c_ulonglong(label_num), c_uint(row_num), c_uint(col_num))
        else:
            p_img_mask_label = img_label.ctypes.data_as(POINTER(c_uint32))
            CFunctionToolImageC.__dll.GetMaskLabelBoundarySeedPointInt(p_img_mask_label, p_seed_point, c_ulonglong(label_num), c_uint(row_num), c_uint(col_num))

        return np_seed_point

    @staticmethod
    def GetImageAdaboostMatrix(image_array:np.array)->np.array:

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        np_adaboost_matrix = np.zeros(image_array.shape, dtype=np.uint32)
        new_image_array = np.array(image_array, dtype=np.uint8)
        new_image_array[new_image_array > 0] = 1
        p_adaboost_np = np_adaboost_matrix.ctypes.data_as(POINTER(c_uint))
        p_image_array = new_image_array.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.Adaboost(p_image_array, p_adaboost_np, c_uint(row_num), c_uint(col_num))
        return np_adaboost_matrix

    @staticmethod
    def GetShapePerimeter(image_array:np.array, start_point_x:int, start_point_y:int):

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        return CFunctionToolImageC.__dll.GetShapePerimeter(p_image_array, c_uint(row_num), c_uint(col_num), c_uint(start_point_x), c_uint(start_point_y))

    @staticmethod
    def RemoveShape(image_array:np.array, start_point_x:int, start_point_y:int):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.RemoveShape(p_image_array, c_uint(row_num), c_uint(col_num), c_uint(start_point_x), c_uint(start_point_y))

    @staticmethod
    def BatchRemoveShapeBySize(image_array:np.array, min_x_length:int=100, max_x_length:int=2000, min_y_length:int=100, max_y_length:int=2000):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.BatchRemoveShapeBySize(p_image_array, c_uint(row_num), c_uint(col_num), c_uint(min_x_length), c_uint(max_x_length), c_uint(min_y_length), c_uint(max_y_length))

    @staticmethod
    def CheckShapeSize(image_array:np.array, start_x:int, start_y:int):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        return CFunctionToolImageC.__dll.CheckShapeSize(p_image_array, c_uint(row_num), c_uint(col_num), c_uint(start_x), c_uint(start_y),
                                                       c_uint(0), c_uint(5000), c_uint(0),
                                                       c_uint(5000))

    @staticmethod
    def WaterShedFindHole(image_array:np.array, shed_value:int=255):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        return CFunctionToolImageC.__dll.WaterShedFindHole(p_image_array, c_uint(row_num), c_uint(col_num), c_uint8(shed_value))

    @staticmethod
    def CalculateShapeArea(image_array:np.array, seed_x:int, seed_y:int):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        return CFunctionToolImageC.__dll.CalculateShapeArea(p_image_array, c_uint(row_num), c_uint(col_num), c_uint(seed_x), c_uint(seed_y))

    @staticmethod
    def BatchRemoveShapeByArea(image_array:np.array, filter_area:int):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.BatchRemoveShapeByArea(p_image_array, c_uint(row_num), c_uint(col_num), c_uint(filter_area))

    @staticmethod
    def GetShapeInfo(image_array:np.array, seed_x:int, seed_y:int, get_perimeter:bool=True)->CDataShapeRectAngleInfo:

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        np_info = np.zeros((1, 6), dtype=np.ulonglong)
        p_np_info = np_info.ctypes.data_as(POINTER(c_ulonglong))
        CFunctionToolImageC.__dll.GetSeedShapeInfo(p_image_array, c_uint(row_num), c_uint(col_num), c_uint(seed_x), c_uint(seed_y), p_np_info, c_bool(get_perimeter))
        return CDataShapeRectAngleInfo(seed_x, seed_y, np_info)

    @staticmethod
    def GetShapeInfoList(image_array:np.array, list_seed:np.array, get_perimeter:bool=True):
        # list_seed np.array [(x, y), (x, y)...]
        seed_num = list_seed.shape[0]
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))

        np_info = np.zeros((seed_num, 6), dtype=np.ulonglong)
        p_np_info = np_info.ctypes.data_as(POINTER(c_ulonglong))

        p_np_seed_list = list_seed.ctypes.data_as(POINTER(c_uint32))

        CFunctionToolImageC.__dll.GetListSeedShapeInfo(p_image_array, c_uint(row_num), c_uint(col_num), c_uint(seed_num), p_np_seed_list, p_np_info, c_bool(get_perimeter))
        list_info = []
        for i in range(seed_num):
            list_info.append(CDataShapeRectAngleInfo(p_np_seed_list[i * 2], p_np_seed_list[i * 2 + 1], np_info[i: i+ 1, :]))
        return list_info

    @staticmethod
    def RemoveByValidAreaRatio(image_array:np.array, valid_ratio:float):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_array.shape
        if image_array.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.DATATYPE + str(np.uint8))
        p_image_array = image_array.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.BatchRemoveShapeByValidAreaRatio(p_image_array, c_uint(row_num), c_uint(col_num), c_double(valid_ratio))

    @staticmethod
    def GetLabelSeedPoint(image_label_array:np.array, search_list:np.array or list):

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        final_result = []
        if len(search_list) == 0:
            CLogging.LogWarning('There is not label need to search. ')
            return final_result
        row_num, col_num = image_label_array.shape
        if image_label_array.dtype != np.uint32:
            new_image_label_array = image_label_array.astype(np.uint32)
        else:
            new_image_label_array = image_label_array
        p_image_label_array = new_image_label_array.ctypes.data_as(POINTER(c_uint))
        if isinstance(search_list, list):
            p_search_list = np.array(search_list, dtype=np.uint32).ctypes.data_as(POINTER(c_uint))
        elif type(search_list) is np.ndarray:
            p_search_list = search_list.ctypes.data_as(POINTER(c_uint))
        else:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_TYPE)
            return
        search_num = len(search_list)
        result_list = np.zeros(search_num * 6, dtype=np.uint32)
        p_result_list = result_list.ctypes.data_as(POINTER(c_uint))
        CFunctionToolImageC.__dll.GetLabelSeedPoint(p_image_label_array, c_uint(row_num), c_uint(col_num), p_search_list, c_uint(search_num), p_result_list)
        for idx in range(search_num):
            final_result.append(CDataShapeRectAngleApex(result_list[idx * 6:(idx + 1) * 6 + 1]))
        return final_result

    @staticmethod
    def FillLabelListToValue(image_label_array:np.array, search_list:np.array or list, set_value:int):

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = image_label_array.shape
        new_image_label_array = np.array(image_label_array, dtype=np.uint32)
        value_image_array = np.array(image_label_array, dtype=np.uint32)
        p_image_label_array = new_image_label_array.ctypes.data_as(POINTER(c_uint32))
        p_value_label_array = value_image_array.ctypes.data_as(POINTER(c_uint32))
        if isinstance(search_list, list):
            p_search_list = np.array(search_list, dtype=c_uint32).ctypes.data_as(POINTER(c_uint32))
        elif type(search_list) is np.ndarray:
            p_search_list = search_list.ctypes.data_as(POINTER(c_uint32))
        else:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_TYPE)
            return
        search_num = len(search_list)
        CFunctionToolImageC.__dll.FillLabelListToValue(p_image_label_array, p_value_label_array, c_uint(row_num), c_uint(col_num), p_search_list, c_uint(search_num), c_uint(set_value))
        return value_image_array

    @staticmethod
    def GetMaskLineBySeedPoint(img_mask:np.array, img_line:np.array, seed_x:int, seed_y:int):

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = img_mask.shape
        p_img_mask_array = img_mask.ctypes.data_as(POINTER(c_uint8))
        p_img_line_array = img_line.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.GetMaskLineBySeedPoint(p_img_mask_array, p_img_line_array, c_uint(row_num), c_uint(col_num), c_uint(seed_x), c_uint(seed_y))

    @staticmethod
    def GetMaskLineByValue(img_mask:np.array, img_line:np.array, value:int):

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = img_mask.shape
        p_img_mask_array = img_mask.ctypes.data_as(POINTER(c_uint8))
        p_img_line_array = img_line.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.GetMaskLineByValue(p_img_mask_array, p_img_line_array, c_uint(row_num), c_uint(col_num), c_uint8(value))

    @staticmethod
    def ReplaceMatrixValue(org_mask:np.array, value_mask:np.array, row_offset:int=0, col_offset:int=0):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        if org_mask.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_TYPE + org_mask.dtype)
        if value_mask.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_TYPE + value_mask.dtype)

        if len(org_mask.shape) != 2 or len(value_mask.shape) != 2:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_TYPE)

        org_mask_h, org_mask_w = org_mask.shape[0], org_mask.shape[1]
        value_mask_h, value_mask_w = value_mask.shape[0], value_mask.shape[1]

        if (org_mask_h < value_mask_h + row_offset) or (org_mask_w < value_mask_w + col_offset):
            CLogging.LogError("The shape of shape matrix is not same ! ")
        p_org_mask = org_mask.ctypes.data_as(POINTER(c_uint8))
        p_value_mask = value_mask.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.ReplaceMatrixValue(p_org_mask, p_value_mask, c_uint(org_mask_h), c_uint(org_mask_w), c_uint(value_mask_h), c_uint(value_mask_w), c_uint(row_offset), c_uint(col_offset))
        
    @staticmethod
    def AddMatrixValue(org_mask:np.array, value_mask:np.array, row_offset:int=0, col_offset:int=0):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        if org_mask.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_TYPE + org_mask.dtype)
        if value_mask.dtype != np.uint8:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_TYPE + value_mask.dtype)

        if len(org_mask.shape) != 2 or len(value_mask.shape) != 2:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_TYPE)

        org_mask_h, org_mask_w = org_mask.shape[0], org_mask.shape[1]
        value_mask_h, value_mask_w = value_mask.shape[0], value_mask.shape[1]

        if (org_mask_h < value_mask_h + row_offset) and (org_mask_w < value_mask_w + col_offset):
            CLogging.LogError("The shape of shape matrix is not same ! ")
        p_org_mask = org_mask.ctypes.data_as(POINTER(c_uint8))
        p_value_mask = value_mask.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.AddMatrixValue(p_org_mask, p_value_mask, c_uint(org_mask_h), c_uint(org_mask_w), c_uint(value_mask_h), c_uint(value_mask_w), c_uint(row_offset), c_uint(col_offset))

    @staticmethod
    def Mask2Label(label:np.array, mask:np.array):
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        if mask.dtype != np.uint8:
            CLogging.LogError("Nonsupport date type !")
        if label.dtype != np.uint32:
            CLogging.LogError("Nonsupport date type !")
        mask_h, mask_w = mask.shape[0], mask.shape[1]
        p_mask = mask.ctypes.data_as(POINTER(c_uint8))
        p_label = label.ctypes.data_as(POINTER(c_uint32))
        label_num = CFunctionToolImageC.__dll.Mask2Label(p_label, p_mask, mask_h, mask_w)
        return label_num

    @staticmethod
    def FilterMatrixValueByLarge(img_mask:np.array, value:int, large_equal:bool=True):
        # 保留大于设定值的部分
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = img_mask.shape
        p_img_mask_array = img_mask.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.FilterMatrixValueByLarge(p_img_mask_array, row_num, col_num, c_uint8(value), c_bool(large_equal))

    @staticmethod
    def FilterMatrixValueBySmall(img_mask:np.array, value:int, small_equal:bool=True):
        # 保留小于设定值的部分
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = img_mask.shape
        p_img_mask_array = img_mask.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.FilterMatrixValueBySmall(p_img_mask_array, c_uint(row_num), c_uint(col_num), c_uint8(value), c_bool(small_equal))

    @staticmethod
    def SetMatrixValue(img_mask:np.array, value:int):

        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = img_mask.shape
        p_img_mask_array = img_mask.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.SetMatrixValue(p_img_mask_array, row_num, col_num, c_uint8(value))

    @staticmethod
    def FillPolygons(img_mask:np.array, seed_x:int, seed_y:int, threshold:int=254, fill_value:int=255):
        if (threshold > 255) or (threshold < 2):
            CLogging.LogError("Threshold must be in range 2-254! ")
        if img_mask.dtype != np.uint8:
            CLogging.LogError("Input mask dtype must be np.uint8! ")
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()
        row_num, col_num = img_mask.shape
        p_img_mask_array = img_mask.ctypes.data_as(POINTER(c_uint8))
        CFunctionToolImageC.__dll.FillPolygon(p_img_mask_array, row_num, col_num, seed_x, seed_y, c_uint8(threshold), c_uint8(fill_value))

    @staticmethod
    def CountValues(img_mask:np.array):

        if img_mask.dtype != np.uint8:
            CLogging.LogError("Input mask dtype must be np.uint8! ")
        if CFunctionToolImageC.__dll is None:
            CFunctionToolImageC.Init()

        row_num, col_num = img_mask.shape
        p_img_mask_array = img_mask.ctypes.data_as(POINTER(c_uint8))

        result_list = np.zeros((1, 256), dtype=np.ulonglong)
        p_result_list = result_list.ctypes.data_as(POINTER(c_ulonglong))
        CFunctionToolImageC.__dll.CountValue(p_img_mask_array, row_num, col_num, p_result_list)
        return result_list[0]