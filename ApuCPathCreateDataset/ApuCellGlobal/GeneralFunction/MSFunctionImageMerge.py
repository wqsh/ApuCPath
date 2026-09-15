import numpy as np

from ApuCellGlobal.MSApuCellExternalPackge import *
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralFunction.MSFunctionMultiThread import CFunctionMultiThread, THREAD_NUM
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging, CLogProgressPercent

from ApuCellGlobal.Data.MSDataFuntionImageMerge import CDataImageMergeInfo
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralSystem.MSSystemEnum import LIST_VISUAL_FILE_TYPE, LIST_VISUAL_FILE_IMAGE_IDX

@unique
class EnumTypeRead(Enum):

    Pkl = 0
    Image = 1


class CFunctionMaskMerge:

    @staticmethod
    def GetOverlapSeedPoint(mask1:np.array, mask2:np.array):

        mask1_tmp = copy.deepcopy(mask1)
        mask2_tmp = copy.deepcopy(mask2)

        mask1_label = CFunctionImageOperator.GetImageMaskLabels(mask1_tmp)
        mask2_label = CFunctionImageOperator.GetImageMaskLabels(mask2_tmp)

        mask1_tmp[mask1_tmp > 0] = 1
        mask2_tmp[mask2_tmp > 0] = 1

        merge_mask = np.zeros(mask1.shape, dtype=np.uint8)
        merge_mask += mask1_tmp
        merge_mask += mask2_tmp

        merge_mask[merge_mask <= 1] = 0
        merge_mask[merge_mask > 1] = 1

        merge_mask_label, label_num = CFunctionImageOperator.GetImageMask2Label(merge_mask)
        list_seed_point = CFunctionToolImageC.GetLabelBoundarySeedPoint(merge_mask_label, label_num)

        mask1_remove_label_idx_list = []
        mask2_remove_label_idx_list = []
        for seed_point_idx in range(1, label_num + 1):
            seed_point_x = list_seed_point[0][seed_point_idx * 2 + 1]
            seed_point_y = list_seed_point[0][seed_point_idx * 2]

            mask1_label_idx = mask1_label[seed_point_y, seed_point_x]
            mask2_label_idx = mask2_label[seed_point_y, seed_point_x]

            mask1_remove_label_idx_list.append(mask1_label_idx)
            mask2_remove_label_idx_list.append(mask2_label_idx)

        mask1_remove_seed_point_list = CFunctionToolImageC.GetLabelSeedPoint(mask1_label, mask1_remove_label_idx_list)
        mask2_remove_seed_point_list = CFunctionToolImageC.GetLabelSeedPoint(mask2_label, mask2_remove_label_idx_list)

        return (mask1_remove_seed_point_list, mask2_remove_seed_point_list)

    @staticmethod
    def RemoveOverlapShape(mask1:np.array, mask2:np.array):

        remove_seed_point_list = CFunctionMaskMerge.GetOverlapSeedPoint(mask1, mask2)
        mask1_remove_seed_point_list = remove_seed_point_list[0]
        mask2_remove_seed_point_list = remove_seed_point_list[1]

        for i in range(len(mask1_remove_seed_point_list)):
            mask1_remove_seed_point_info = mask1_remove_seed_point_list[i]
            mask2_remove_seed_point_info = mask2_remove_seed_point_list[i]
            CFunctionToolImageC.RemoveShape(mask1, mask1_remove_seed_point_info.seed_x, mask1_remove_seed_point_info.seed_y)
            CFunctionToolImageC.RemoveShape(mask2, mask2_remove_seed_point_info.seed_x, mask2_remove_seed_point_info.seed_y)

    @staticmethod
    def RemoveOverlapShapeOnlySecond(mask1: np.array, mask2: np.array):

        remove_seed_point_list = CFunctionMaskMerge.GetOverlapSeedPoint(mask1, mask2)
        mask1_remove_seed_point_list = remove_seed_point_list[0]
        mask2_remove_seed_point_list = remove_seed_point_list[1]

        for i in range(len(mask1_remove_seed_point_list)):
            mask2_remove_seed_point_info = mask2_remove_seed_point_list[i]
            CFunctionToolImageC.RemoveShape(mask2, mask2_remove_seed_point_info.seed_x, mask2_remove_seed_point_info.seed_y)

    @staticmethod
    def GetNonOverlapMask(multi_channel_matrix:np.array):

        if len(multi_channel_matrix.shape) != 3:
            CLogging.LogError("Image shape error ! ")
        CLogging.LogToUser("Delete overlap shape area ... ")
        channel_num = multi_channel_matrix.shape[2]
        row_size, col_size = multi_channel_matrix.shape[0], multi_channel_matrix.shape[1]
        filter_data_matrix = np.zeros((row_size, col_size), dtype=np.uint8)
        # 这里把所有预测结果都加到一个通道里
        for i in range(channel_num):
            filter_data_matrix[:, :] += multi_channel_matrix[:, :, i]
        # 只保留重叠区域
        overlap_matrix = copy.deepcopy(filter_data_matrix)
        overlap_matrix[overlap_matrix < 2] = 0
        CLogging.LogToUser("Channel data merge. ")
        # 给所有预测结果的连通域做编号
        filter_data_matrix[filter_data_matrix > 0] = 1
        filter_data_matrix_label = CFunctionImageOperator.GetImageMaskLabels(filter_data_matrix, 1)
        CFunctionMemory.DeleteVariableMemory(filter_data_matrix)
        org_label_num = np.max(filter_data_matrix_label)
        CLogging.LogToUser("Input shape area num : %d. "%org_label_num)
        overlap_record_idx_list = np.full(org_label_num + 1, False)
        # 给重叠区域编号
        overlap_matrix_label = CFunctionImageOperator.GetImageMaskLabels(overlap_matrix, 1)
        CFunctionMemory.DeleteVariableMemory(overlap_matrix)
        overlap_label_num = np.max(overlap_matrix_label)
        CLogging.LogToUser("Overlap shape area num : %d. "%overlap_label_num)
        overlap_search_list = [i + 1 for i in range(overlap_label_num)]
        # 获取重叠区域的连通域标记点
        overlap_seed_point_list = CFunctionToolImageC.GetLabelSeedPoint(overlap_matrix_label, overlap_search_list)
        CFunctionMemory.DeleteVariableMemory(overlap_matrix_label)
        delete_overlap_idx_list = []
        delete_num = 0
        for idx, overlap_seed_point in enumerate(overlap_seed_point_list):
            delete_overlap_idx = filter_data_matrix_label[overlap_seed_point.seed_y, overlap_seed_point.seed_x]
            if not overlap_record_idx_list[delete_overlap_idx]:
                delete_overlap_idx_list.append(delete_overlap_idx)
                overlap_record_idx_list[delete_overlap_idx] = True
                delete_num += 1
        CLogging.LogToUser("Delete shape area num : %d. "%delete_num)
        delete_matrix = CFunctionToolImageC.FillLabelListToValue(filter_data_matrix_label, delete_overlap_idx_list, 0)
        CFunctionMemory.DeleteVariableMemory(filter_data_matrix_label)
        CLogging.LogToUser("Delete shape area. ")
        final_num = len(np.unique(delete_matrix)) - 1
        CLogging.LogToUser("Final shape area num : %d. "%final_num)
        CLogging.LogToUser("Finish delete overlap shape area. ")
        return delete_matrix

    @staticmethod
    def RetainShapeOverlapMask(input_matrix1:np.array, input_matrix2:np.array, return_matrix:np.array):

        input_matrix1_row_length, input_matrix1_col_length = input_matrix1.shape
        input_matrix2_row_length, input_matrix2_col_length = input_matrix2.shape

        if (input_matrix1_row_length != input_matrix2_row_length) or (input_matrix1_col_length != input_matrix2_col_length):
            CLogging.LogError("Input matrix size is differ. ")
        CLogging.LogToUser("Retain overlap shape area ... ")
        return_matrix += input_matrix1
        return_matrix += input_matrix2

        # 只保留重叠区域
        overlap_matrix = copy.deepcopy(return_matrix)
        overlap_matrix[overlap_matrix < 2] = 0
        # 给重叠区域编号
        overlap_matrix_label = CFunctionImageOperator.GetImageMaskLabels(overlap_matrix, 1)
        CFunctionMemory.DeleteVariableMemory(overlap_matrix)
        overlap_label_num = np.max(overlap_matrix_label)
        overlap_search_list = [i + 1 for i in range(overlap_label_num)]
        # 获取重叠区域的连通域标记点
        overlap_seed_point_list = CFunctionToolImageC.GetLabelSeedPoint(overlap_matrix_label, overlap_search_list)
        CFunctionMemory.DeleteVariableMemory(overlap_matrix_label)
        CLogging.LogToUser("Overlap shape area num : %d. " % overlap_label_num)

        # 给所有预测结果的连通域做编号
        return_matrix[return_matrix > 0] = 1
        new_matrix_label = CFunctionImageOperator.GetImageMaskLabels(return_matrix, 1)
        # 得到所有标签
        org_label_num = np.max(new_matrix_label)
        CLogging.LogToUser("Input shape area num : %d. "%org_label_num)
        overlap_record_idx_list = np.full(org_label_num + 1, False)

        # 现确定有重叠的区域的标记zhi
        retain_overlap_idx_list = []
        retain_num = 0
        for idx, overlap_seed_point in enumerate(overlap_seed_point_list):
            retain_overlap_idx = new_matrix_label[overlap_seed_point.seed_y, overlap_seed_point.seed_x]
            if not overlap_record_idx_list[retain_overlap_idx]:
                retain_overlap_idx_list.append(retain_overlap_idx)
                overlap_record_idx_list[retain_overlap_idx] = True
                retain_num += 1
        CLogging.LogToUser("Retain shape area num : %d. " % retain_num)
        delete_idx_list = []
        for i in range(org_label_num):
            if not overlap_record_idx_list[i + 1]:
                delete_idx_list.append(i + 1)
        CLogging.LogToUser("Delete shape area num : %d. " % len(delete_idx_list))
        retain_mask = CFunctionToolImageC.FillLabelListToValue(new_matrix_label, delete_idx_list, 0)
        CFunctionMemory.DeleteVariableMemory(new_matrix_label)
        return_matrix[retain_mask == 0] = 0
        CLogging.LogToUser("Finish retain overlap shape area. ")

class CFunctionImageSplitMerge:

    @staticmethod
    def MergeImage(folder_path: str, split_symbol: str = '_', x_min_start_idx: int = 0, y_min_start_idx: int = 0, x_max_idx:int=None, y_max_idx:int=None,
                   fov_x_start: int = 0,
                   fov_x_end: int = -1,
                   fov_y_start: int = 0,
                   fov_y_end: int = -1,
                   use_datatype=None,
                   ):

        run_thread = CFunctionImageSplitMergeMultiThread(folder_path, split_symbol, x_min_start_idx, y_min_start_idx, x_max_idx, y_max_idx, "Merge image ... ",
                                                         fov_x_start,
                                                         fov_x_end,
                                                         fov_y_start,
                                                         fov_y_end,
                                                         use_datatype,
                                                         )
        run_thread.RunMultiThread()
        new_img = run_thread.GetMergeImage()
        return new_img

    @staticmethod
    def MergeImagePKL(folder_path: str, split_symbol: str = '_', x_min_start_idx: int = 0, y_min_start_idx: int = 0, x_max_idx:int=None, y_max_idx:int=None):

        run_thread = CFunctionImageSplitMergeMultiThread(folder_path, split_symbol, x_min_start_idx, y_min_start_idx, x_max_idx, y_max_idx, "Merge image ... ")
        run_thread.RunMultiThread()
        new_img = run_thread.GetMergeImage()
        return new_img

    @staticmethod
    def GetMergeInfo(folder_path:str, split_symbol:str="_", x_min_start_idx: int = 0, y_min_start_idx: int = 0, x_max_idx:int=None, y_max_idx:int=None):
        list_file = os.listdir(folder_path)
        if y_max_idx is None:
            y_max_idx = 0
        if x_max_idx is None:
            x_max_idx = 0
        y_min_idx = y_min_start_idx
        x_min_idx = x_min_start_idx
        size_flag = False
        x_size = 0
        y_size = 0
        channel = 0
        for file_name in list_file:
            try:
                y_idx = int(file_name.split('.')[0].split(split_symbol)[-2])
                x_idx = int(file_name.split('.')[0].split(split_symbol)[-1])
            except:
                CLogging.LogError("The file name does not meet merge requirements !")
            if not size_flag:
                if file_name.split('.')[-1] == "pkl":
                    image_array = CFunctionFileIO.ReadPKL(os.path.join(folder_path, file_name))
                else:
                    image_array = CFunctionFileIO.ReadImage(os.path.join(folder_path, file_name))
                if isinstance(image_array, np.ndarray):
                    if len(image_array.shape) == 2:
                        channel = 1
                    elif len(image_array.shape) == 3:
                        channel = image_array.shape[2]
                    else:
                        CLogging.LogError("Unsupport merge shape " + str(image_array.shape))
                    
                    y_size, x_size = image_array.shape[0], image_array.shape[1]
                else:
                    channel = None
                    y_size = None
                    x_size = None
                size_flag = True
            if y_idx < y_min_idx:
                y_min_idx = y_idx
            if y_idx > y_max_idx:
                y_max_idx = y_idx
            if x_idx < x_min_idx:
                x_min_idx = x_idx
            if x_idx > x_max_idx:
                x_max_idx = x_idx
        return CDataImageMergeInfo(folder_path, y_min_idx, x_min_idx, y_max_idx, x_max_idx, y_size, x_size, channel)

    @staticmethod
    def SplitImage(img:np.array, folder_output_path:str, out_name:str, split_symbol:str="_", row_size:int=512, col_size:int=512):

        x_min = 0
        x_max = img.shape[1]
        y_min = 0
        y_max = img.shape[0]

        row_num = math.ceil((y_max - y_min) / row_size)
        col_num = math.ceil((x_max - x_min) / col_size)

        CLogging.LogToUser("Split image ... ")

        progress_percent = CLogProgressPercent(0, row_num - 1, 0.02, out_name)

        channel:bool = False
        if len(img.shape) == 3:
            channel = True
        for i in range(row_num):
            progress_percent.ShowProgressPercent(i)
            for j in range(col_num):
                if channel:
                    CFunctionFileIO.WriteImage(folder_output_path, out_name + split_symbol + "%d"%i + split_symbol + "%d.tif" %j, img[i * row_size: (i + 1) * row_size, j * col_size: (j + 1) * col_size, :])
                else:
                    CFunctionFileIO.WriteImage(folder_output_path, out_name + split_symbol + "%d"%i + split_symbol + "%d.tif" %j, img[i * row_size: (i + 1) * row_size, j * col_size: (j + 1) * col_size])

    @staticmethod
    def SplitImageFile(image_path:str, folder_output_path:str, out_name:str, split_symbol:str="_", row_size:int=512, col_size:int=512):

        img = CFunctionFileIO.ReadImage(image_path)
        CFunctionImageSplitMerge.SplitImage(img, folder_output_path, out_name, split_symbol, row_size, col_size)

class CFunctionImageSplitMergeMultiThread(CFunctionMultiThread):

    def __init__(self, folder_path:str, split_symbol:str="_", x_min_start_idx: int = 0, y_min_start_idx: int = 0, x_max_idx:int=None, y_max_idx:int=None, progress_log_info:str="",
                 fov_x_start:int=0,
                 fov_x_end:int=-1,
                 fov_y_start:int=0,
                 fov_y_end:int=-1,
                 use_datatype=None,
                 ):

        self.__split_symbol:str = split_symbol
        self.__folder_path:str = folder_path

        self.fov_x_start: int = fov_x_start
        self.fov_x_end: int = fov_x_end
        self.fov_y_start: int = fov_y_start
        self.fov_y_end: int = fov_y_end

        self.use_datatype = use_datatype

        self.__merge_folder_info = CFunctionImageSplitMerge.GetMergeInfo(folder_path, split_symbol, x_min_start_idx, y_min_start_idx, x_max_idx, y_max_idx)

        list_file = os.listdir(folder_path)
        super(CFunctionImageSplitMergeMultiThread, self).__init__(list_file, THREAD_NUM, progress_log_info, True, False)

        self.__type_read:EnumTypeRead = None

        self.__img_dtype = np.uint8
        path_file = list_file[0]
        image_type = path_file.split('.')[-1]
        image_type = str.lower(image_type)
        if image_type in LIST_VISUAL_FILE_TYPE:
            image_type_idx = LIST_VISUAL_FILE_TYPE.index(image_type)
            if image_type_idx in LIST_VISUAL_FILE_IMAGE_IDX:
                self.__type_read = EnumTypeRead.Image
            else:
                CLogging.LogError('Unknown read file type ! ')
        elif image_type == 'pkl':
            self.__type_read = EnumTypeRead.Pkl
            self.__img_dtype = CFunctionFileIO.ReadPKL(os.path.join(self.__folder_path, self.data_list[0])).dtype
        else:
            CLogging.LogError('Unknown read file type ! ')
        if not self.use_datatype is None:
            self.__img_dtype = self.use_datatype
        self.__new_img = np.zeros((self.__merge_folder_info.channel, self.__merge_folder_info.y_full_size, self.__merge_folder_info.x_full_size), dtype=self.__img_dtype)

    def UserFunction(self, idx:int):

        file_name = self.data_list[idx]
        y_idx = 0
        x_idx = 0
        try:
            y_idx = int(file_name.split('.')[0].split(self.__split_symbol)[-2])
            x_idx = int(file_name.split('.')[0].split(self.__split_symbol)[-1])
        except:
            CLogging.LogError("The file name doReadImagees not meet merge requirements !")
        if self.__type_read == EnumTypeRead.Image:
            org_image_array = CFunctionFileIO.ReadImage(os.path.join(self.__folder_path, file_name))
        elif self.__type_read == EnumTypeRead.Pkl:
            org_image_array = CFunctionFileIO.ReadPKL(os.path.join(self.__folder_path, file_name))
        else:
            return

        if self.use_datatype is None:
            image_array = np.zeros(org_image_array.shape, org_image_array.dtype)
            multi = 1
        else:
            if np.max(org_image_array) > 1 or np.min(org_image_array) < 0:
                CLogging.LogWarning("Data may overflow or truncation! ")
            image_array = np.zeros(org_image_array.shape, self.use_datatype)
            if self.use_datatype == np.uint8:
                multi = 1e2
            elif self.use_datatype == np.uint16:
                multi = 1e4
            elif self.use_datatype == np.uint32:
                multi = 1e8
            else:
                multi = 1

        if len(image_array.shape) == 2:
            tmp_array = org_image_array[self.fov_y_start:self.fov_y_end, self.fov_x_start: self.fov_x_end] * multi
            if self.use_datatype is None:
                tmp_array = tmp_array.astype(self.__img_dtype)
            else:
                tmp_array = tmp_array.astype(self.use_datatype)
            image_array[self.fov_y_start:self.fov_y_end, self.fov_x_start: self.fov_x_end] += tmp_array

            self.__new_img[0, (y_idx - self.__merge_folder_info.y_min_idx) * self.__merge_folder_info.y_size: (y_idx - self.__merge_folder_info.y_min_idx + 1) * self.__merge_folder_info.y_size, (x_idx - self.__merge_folder_info.x_min_idx) * self.__merge_folder_info.x_size: (x_idx - self.__merge_folder_info.x_min_idx + 1) * self.__merge_folder_info.x_size] += image_array
        else:
            tmp_array = org_image_array[self.fov_y_start:self.fov_y_end, self.fov_x_start: self.fov_x_end, :] * multi
            if self.use_datatype is None:
                tmp_array = tmp_array.astype(self.__img_dtype)
            else:
                tmp_array = tmp_array.astype(self.use_datatype)
            image_array[self.fov_y_start:self.fov_y_end, self.fov_x_start: self.fov_x_end,:] += tmp_array
            image_array = image_array.transpose((2, 0, 1))
            self.__new_img[:, (y_idx - self.__merge_folder_info.y_min_idx) * self.__merge_folder_info.y_size: (y_idx - self.__merge_folder_info.y_min_idx + 1) * self.__merge_folder_info.y_size, (x_idx - self.__merge_folder_info.x_min_idx) * self.__merge_folder_info.x_size: (x_idx - self.__merge_folder_info.x_min_idx + 1) * self.__merge_folder_info.x_size] += image_array

    def GetMergeImage(self):

        return self.__new_img

