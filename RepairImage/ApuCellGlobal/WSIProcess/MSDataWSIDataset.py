from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.MaskRCNN.DataSet.MSDataMaskRCNNDataSet import EnumPadType
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralSystem.MSLogging import CLogProgressPercent
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralSystem.MSSystemEnum import LIST_DATA_FILE_TYPE_SUFFIX, EnumDataFileType, LIST_VISUAL_FILE_TYPE_SUFFIX, EnumVisualFileType

from ApuCellGlobal.GeneralFunction.MSFunctionMultiThread import CFunctionMultiThread, THREAD_NUM

from enum import Enum, unique
import numpy as np
import os
import cv2

@unique
class EnumDatasetType(Enum):

    Normal = 0
    Train = 1
    Valid = 2
    Test = 3
    Predict = 4
    Assist = 5

@unique
class EnumTypeOutputWSICrop(Enum):

    Image= 0
    Mask = 1

@unique
class EnumTypeExportImage(Enum):

    Pkl = 0
    Tif = 1

class CDataWSIDatasetBrevity:

    def __init__(self, name:str, image_path:str, mask_path:str, row_size:int=512, col_size:int=512, dataset_type:EnumDatasetType=EnumDatasetType.Predict):

        self.name:str = name
        self.image_path:str = image_path
        self.mask_path:str = mask_path
        self.row_size:int = row_size
        self.col_size:int = col_size

        self.image_height:int = 0
        self.image_width:int = 0

        self.org_height:int = 0
        self.org_width:int = 0

        self.diagonal_height:int = 0
        self.diagonal_width:int = 0

        self.matrix_type_idx:list = []
        self.matrix_idx:list = []

        self.dataset_type = dataset_type

        self.__captainFillInfo()

    def __captainFillInfo(self):

        image_w, image_h = CFunctionFileIO.GetImageFileSize(self.image_path)
        self.image_height = image_h
        self.image_width = image_w
        if not self.mask_path is None:
            if self.mask_path.endswith(LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIF.value]) or self.mask_path.endswith(LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value]):
                mask_w, mask_h = CFunctionFileIO.GetImageFileSize(self.mask_path)
                if (image_w != mask_w) or (image_h != mask_h):
                    CLogging.LogError("Image and mask size are not match ! ")

        half_row_size = int(self.row_size / 2)
        half_col_size = int(self.col_size / 2)

        x_min = 0
        x_max = image_w
        y_min = 0
        y_max = image_h

        row_length = y_max - y_min
        col_length = x_max - x_min


        if (y_max - y_min) % self.row_size == 0:
            row_num = (y_max - y_min) // self.row_size
        else:
            row_num = (y_max - y_min) // self.row_size + 1
        self.org_height = row_num * self.row_size
        self.diagonal_height = (row_num + 1) * self.row_size

        if (x_max - x_min) % self.col_size == 0:
            col_num = (x_max - x_min) // self.col_size
        else:
            col_num = (x_max - x_min) // self.col_size + 1
        self.org_width = col_num * self.col_size
        self.diagonal_width = (col_num + 1) * self.col_size

        for pad_type in EnumPadType:
            cur_matrix_type_idx = np.zeros((row_num + 1, col_num + 1, 12), dtype=np.int32)
            if pad_type == EnumPadType.Org:
                valid_row_num_start, valid_row_num_end = 0, row_num
                valid_col_num_start, valid_col_num_end = 0, col_num
            elif pad_type == EnumPadType.Horizontal:
                valid_row_num_start, valid_row_num_end = 0, row_num
                valid_col_num_start, valid_col_num_end = 0, col_num + 1
            elif pad_type == EnumPadType.Vertical:
                valid_row_num_start, valid_row_num_end = 0, row_num + 1
                valid_col_num_start, valid_col_num_end = 0, col_num
            elif pad_type == EnumPadType.Diagonal:
                valid_row_num_start, valid_row_num_end = 0, row_num + 1
                valid_col_num_start, valid_col_num_end = 0, col_num + 1
            else:
                continue
            for i in range(valid_row_num_start, valid_row_num_end):
                for j in range(valid_col_num_start, valid_col_num_end):
                    cur_matrix_type_idx[i, j, 2] = i * self.row_size
                    cur_matrix_type_idx[i, j, 3] = (i + 1) * self.row_size
                    cur_matrix_type_idx[i, j, 8] = j * self.col_size
                    cur_matrix_type_idx[i, j, 9] = (j + 1) * self.row_size
                    if pad_type == EnumPadType.Vertical or pad_type == EnumPadType.Diagonal:
                        if i == 0:
                            cur_matrix_type_idx[i, j ,0] = 0
                            cur_matrix_type_idx[i, j, 1] = half_row_size
                            cur_matrix_type_idx[i, j, 2] = 0
                            cur_matrix_type_idx[i, j, 3] = half_row_size
                        else:
                            cur_matrix_type_idx[i, j, 2] = i * self.row_size - half_row_size
                            cur_matrix_type_idx[i, j, 3] = (i + 1) * self.row_size - half_row_size
                    if pad_type == EnumPadType.Horizontal or pad_type == EnumPadType.Diagonal:
                        if j == 0:
                            cur_matrix_type_idx[i, j, 6] = 0
                            cur_matrix_type_idx[i, j, 7] = half_col_size
                            cur_matrix_type_idx[i, j, 8] = 0
                            cur_matrix_type_idx[i, j, 9] = half_col_size
                        else:
                            cur_matrix_type_idx[i, j, 8] = j * self.col_size - half_col_size
                            cur_matrix_type_idx[i, j, 9] = (j + 1) * self.col_size - half_col_size

                    if (cur_matrix_type_idx[i, j, 3] > image_h):
                        if cur_matrix_type_idx[i, j, 2] > image_h:
                            cur_matrix_type_idx[i, j, 5] = self.row_size
                            cur_matrix_type_idx[i, j, 4] = 0
                            cur_matrix_type_idx[i, j, 3] = cur_matrix_type_idx[i, j, 2]
                        else:
                            cur_matrix_type_idx[i, j, 5] = cur_matrix_type_idx[i, j, 3] - image_h
                            cur_matrix_type_idx[i, j, 4] = 0
                            cur_matrix_type_idx[i, j, 3] = image_h

                    if (cur_matrix_type_idx[i, j, 9] > image_w):
                        if cur_matrix_type_idx[i, j, 8] > image_w:
                            cur_matrix_type_idx[i, j, 9] = cur_matrix_type_idx[i, j, 8]
                            cur_matrix_type_idx[i, j, 10] = 0
                            cur_matrix_type_idx[i, j, 11] = self.col_size
                        else:
                            cur_matrix_type_idx[i, j, 11] = cur_matrix_type_idx[i, j, 9] - image_w
                            cur_matrix_type_idx[i, j, 10] = 0
                            cur_matrix_type_idx[i, j, 9] = image_w

            self.matrix_type_idx.append(pad_type)
            self.matrix_idx.append(cur_matrix_type_idx)
            cur_matrix_type_idx[cur_matrix_type_idx < 0] = 0

    def OutputCropImageFullPadType(self, path_folder:str, file_export_type:EnumTypeExportImage=EnumTypeExportImage.Pkl):

        path_folder = os.path.join(path_folder, self.name)
        run_thread_export_wsi = CFunctionOutputWSIMultiThread(self, [], None, EnumTypeOutputWSICrop.Image, path_folder, file_export_type, thread_num=THREAD_NUM)

        for pad_type_idx, pad_type in enumerate(self.matrix_type_idx):
            if pad_type in self.matrix_type_idx:

                index = self.matrix_type_idx.index(pad_type)
                matrix_idx = self.matrix_idx[index]
                crop_row_num, crop_col_num = matrix_idx.shape[0], matrix_idx.shape[1]

                valid_idx_list = []
                for i in range(crop_row_num):
                    for j in range(crop_col_num):
                        valid_idx_list.append((i, j))

                run_thread_export_wsi.data_list = valid_idx_list
                run_thread_export_wsi.ChangePadType(pad_type)
                run_thread_export_wsi.RunMultiThread()

        del run_thread_export_wsi
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()

    def OutputCropMaskFullPadType(self, path_folder:str, file_export_type:EnumTypeExportImage=EnumTypeExportImage.Pkl):

        if self.mask_path is None:
            CLogging.LogWarning("This is not a dataset with mask ! ")
            return
        path_folder = os.path.join(path_folder, self.name)

        CFunctionMemory.GetCurMemoryInfo()
        run_thread_export_wsi = CFunctionOutputWSIMultiThread(self, [], None, EnumTypeOutputWSICrop.Mask, path_folder, file_export_type, thread_num=THREAD_NUM)
        CFunctionMemory.GetCurMemoryInfo()

        for pad_type_idx, pad_type in enumerate(self.matrix_type_idx):
            if pad_type in self.matrix_type_idx:

                path_folder = os.path.join(path_folder, self.name, pad_type.name)
                if not os.path.exists(path_folder):
                    os.makedirs(path_folder)

                index = self.matrix_type_idx.index(pad_type)
                matrix_idx = self.matrix_idx[index]
                crop_row_num, crop_col_num = matrix_idx.shape[0], matrix_idx.shape[1]

                valid_idx_list = []
                for i in range(crop_row_num):
                    for j in range(crop_col_num):
                        valid_idx_list.append((i, j))

                run_thread_export_wsi.data_list = valid_idx_list
                run_thread_export_wsi.ChangePadType(pad_type)
                run_thread_export_wsi.RunMultiThread()

        del run_thread_export_wsi
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()

    def OutputCropImage(self, path_folder:str, pad_type:EnumPadType, file_export_type:EnumTypeExportImage=EnumTypeExportImage.Pkl):

        if pad_type in self.matrix_type_idx:
            path_folder = os.path.join(path_folder, self.name)
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)

            CFunctionMemory.GetCurMemoryInfo()

            index = self.matrix_type_idx.index(pad_type)
            matrix_idx = self.matrix_idx[index]
            crop_row_num, crop_col_num = matrix_idx.shape[0], matrix_idx.shape[1]

            valid_idx_list = []
            for i in range(crop_row_num):
                for j in range(crop_col_num):
                    valid_idx_list.append((i, j))
            run_thread_export_wsi = CFunctionOutputWSIMultiThread(self, valid_idx_list, pad_type, EnumTypeOutputWSICrop.Image, path_folder, file_export_type, thread_num=THREAD_NUM)

            CFunctionMemory.GetCurMemoryInfo()

            run_thread_export_wsi.RunMultiThread()
            del run_thread_export_wsi

            CFunctionMemory.CleanMemory()
            CFunctionMemory.GetCurMemoryInfo()

    def OutputCropMask(self, path_folder: str, pad_type: EnumPadType, file_export_type:EnumTypeExportImage=EnumTypeExportImage.Pkl):
        if self.mask_path is None:
            CLogging.LogWarning("This is not a dataset with mask ! ")
            return
        if pad_type in self.matrix_type_idx:

            path_folder = os.path.join(path_folder, self.name)
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)

            CFunctionMemory.GetCurMemoryInfo()

            index = self.matrix_type_idx.index(pad_type)
            matrix_idx = self.matrix_idx[index]
            crop_row_num, crop_col_num = matrix_idx.shape[0], matrix_idx.shape[1]

            valid_idx_list = []
            for i in range(crop_row_num):
                for j in range(crop_col_num):
                    valid_idx_list.append((i, j))
            run_thread_export_wsi = CFunctionOutputWSIMultiThread(self, valid_idx_list, pad_type, EnumTypeOutputWSICrop.Mask, path_folder, file_export_type, thread_num=THREAD_NUM)

            CFunctionMemory.GetCurMemoryInfo()

            run_thread_export_wsi.RunMultiThread()
            del run_thread_export_wsi

            CFunctionMemory.CleanMemory()
            CFunctionMemory.GetCurMemoryInfo()

    def OutputImageMaskValidDatasetFullPadType(self, path_folder:str, valid_area_up:float=1.0, valid_area_down:float=0.0, file_export_type:EnumTypeExportImage=EnumTypeExportImage.Pkl):
        if self.mask_path is None:
            CLogging.LogWarning("This is not a dataset with mask !")
            return
        path_folder = os.path.join(path_folder, self.name)
        valid_idx_lists = []
        for pad_type_idx, pad_type in enumerate(self.matrix_type_idx):

            if pad_type in self.matrix_type_idx:

                CLogging.LogToUser("Get valid " + pad_type.name + " idx ... ")
                valid_idx_list = self.__captainGetValidDatasetByValidArea(pad_type, valid_area_up, valid_area_down)
                valid_idx_lists.append(valid_idx_list)
                CLogging.LogToUser("Finish get valid " + pad_type.name + " idx. ")
            else:
                valid_idx_lists.append([])

        run_thread_export_wsi = CFunctionOutputWSIMultiThread(self, [], None, None, path_folder, file_export_type, thread_num=THREAD_NUM)
        run_thread_export_wsi.ChangeExportType(EnumTypeOutputWSICrop.Image)
        for pad_type_idx, pad_type in enumerate(self.matrix_type_idx):
            if pad_type in self.matrix_type_idx:
                run_thread_export_wsi.data_list = valid_idx_lists[pad_type_idx]
                run_thread_export_wsi.ChangePadType(pad_type)
                run_thread_export_wsi.RunMultiThread()

        run_thread_export_wsi.ChangeExportType(EnumTypeOutputWSICrop.Mask)
        for pad_type_idx, pad_type in enumerate(self.matrix_type_idx):
            if pad_type in self.matrix_type_idx:
                run_thread_export_wsi.data_list = valid_idx_lists[pad_type_idx]
                run_thread_export_wsi.ChangePadType(pad_type)
                run_thread_export_wsi.RunMultiThread()

        del run_thread_export_wsi
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()

    def OutputImageMaskValidDataset(self, path_folder:str, pad_type:EnumPadType, valid_area_up:float=1.0, valid_area_down:float=0.0, file_export_type:EnumTypeExportImage=EnumTypeExportImage.Pkl):

        if self.mask_path is None:
            CLogging.LogWarning("This is not a dataset with mask !")
            return

        if pad_type in self.matrix_type_idx:

            path_folder = os.path.join(path_folder, self.name)
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)

            CLogging.LogToUser("Get valid image idx ...")
            valid_idx_list = self.__captainGetValidDatasetByValidArea(pad_type, valid_area_up, valid_area_down)
            CLogging.LogToUser("Finish get valid image idx. ")

            run_thread_export_wsi = CFunctionOutputWSIMultiThread(self, valid_idx_list, pad_type, EnumTypeOutputWSICrop.Image, path_folder, file_export_type, thread_num=THREAD_NUM)
            run_thread_export_wsi.RunMultiThread()

            run_thread_export_wsi.ChangeExportType(EnumTypeOutputWSICrop.Mask)
            run_thread_export_wsi.RunMultiThread()

            del run_thread_export_wsi
            CFunctionMemory.CleanMemory()
            CFunctionMemory.GetCurMemoryInfo()

    def __captainGetValidDatasetByValidArea(self, pad_type:EnumPadType, valid_area_up:float=1.0, valid_area_down:float=0.0):

        valid_idx_list = []
        if pad_type in self.matrix_type_idx:
            mask = self.GetMask()
            mask_channel = mask.shape[2]
            index = self.matrix_type_idx.index(pad_type)
            matrix_idx = self.matrix_idx[index]
            crop_row_num, crop_col_num = matrix_idx.shape[0], matrix_idx.shape[1]
            progress = CLogProgressPercent(0, crop_row_num - 1, 0.02, "Get valid idx " + self.name + ", " + pad_type.name + " : ")
            for i in range(crop_row_num):
                progress.ShowProgressPercent(i)
                for j in range(crop_col_num):
                    valid_img: bool = False
                    if ((matrix_idx[i, j, 3] == 0) or (matrix_idx[i, j, 9] == 0)):
                        continue
                    if ((matrix_idx[i, j, 1] - matrix_idx[i, j, 0] != 0) or
                            (matrix_idx[i, j, 5] - matrix_idx[i, j, 4] != 0) or
                            (matrix_idx[i, j, 7] - matrix_idx[i, j, 6] != 0) or
                            (matrix_idx[i, j, 11] - matrix_idx[i, j, 10] != 0)):
                        tmp_row_idx_start = 0
                        tmp_row_idx_end = self.row_size
                        tmp_col_idx_start = 0
                        tmp_col_idx_end = self.col_size

                        if (matrix_idx[i, j, 1] - matrix_idx[i, j, 0] != 0):
                            tmp_row_idx_start = matrix_idx[i, j, 1]
                        if (matrix_idx[i, j, 5] - matrix_idx[i, j, 4] != 0):
                            tmp_row_idx_end = self.row_size - matrix_idx[i, j, 5] % self.row_size
                        if (matrix_idx[i, j, 7] - matrix_idx[i, j, 6] != 0):
                            tmp_col_idx_start = matrix_idx[i, j, 7]
                        if (matrix_idx[i, j, 11] - matrix_idx[i, j, 10] != 0):
                            tmp_col_idx_end = self.col_size - matrix_idx[i, j, 11] % self.col_size
                        tmp_mask = np.zeros((self.row_size, self.col_size, mask_channel), dtype=np.uint8)
                        if ((matrix_idx[i, j, 2] == matrix_idx[i, j, 3]) or (matrix_idx[i, j, 8] == matrix_idx[i, j, 9])):
                            pass
                        else:
                            tmp_mask[tmp_row_idx_start: tmp_row_idx_end, tmp_col_idx_start: tmp_col_idx_end, :] += mask[matrix_idx[i, j, 2]: matrix_idx[i, j, 3], matrix_idx[i, j, 8]: matrix_idx[i, j, 9], :]

                    else:
                        # 不需要填充零的
                        tmp_mask = mask[matrix_idx[i, j, 2]: matrix_idx[i, j, 3], matrix_idx[i, j, 8]: matrix_idx[i, j, 9], :]
                    tmp_mask[tmp_mask > 0] = 1
                    valid_percent = np.sum(tmp_mask) / (self.row_size * self.col_size)
                    if (valid_percent >= valid_area_down) and (valid_percent <= valid_area_up):
                        valid_img = True
                    if valid_img:
                        valid_idx_list.append((i, j))
                    del tmp_mask
            del mask
            CFunctionMemory.CleanMemory()
        return valid_idx_list

    def GetMask(self):

        if self.mask_path.endswith(
                LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIF.value]) or self.mask_path.endswith(
                LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value]):
            mask = CFunctionFileIO.ReadImage(self.mask_path)
        elif self.mask_path.endswith(LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.JSON.value]):
            list_idx_name = ['BG', 'Pos', 'Neg']

            w, h = CFunctionFileIO.GetImageFileSize(self.image_path)
            mask = np.zeros((h, w, len(list_idx_name)), dtype=np.uint8)
            # mask = mask.transpose((1, 2, 0))
            mask_info = CFunctionFileIO.ReadJson(self.mask_path)
            list_idx_info = [[] for i in range(len(list_idx_name))]
            for label_info in mask_info['labels']:
                if label_info['name'] in list_idx_name:
                    idx = list_idx_name.index(label_info['name'])
                    list_idx_info[idx] += label_info['polygons']
                    continue
            label_num = 0
            for idx, idx_info in enumerate(list_idx_info):
                CLogging.LogToUser("Read %s info ... "%list_idx_name[idx])
                channel_data = [0 for i in range(len(list_idx_name))]
                channel_data[label_num] = 1
                fill_value = [0 for i in range(len(list_idx_name))]
                fill_value[label_num] = 255
                label_num += 1
                for polygon_info in idx_info:
                    list_point_info = polygon_info['points']
                    cv2.fillPoly(mask, np.array([list_point_info], dtype=np.int32), fill_value)
                CLogging.LogToUser("Finish read %s info . " % list_idx_name[idx])
        else:
            CLogging.LogError("Unknown mask type ! ")
        return mask

    def GetImage(self):

        return CFunctionFileIO.ReadImage(self.image_path)

class CFunctionOutputWSIMultiThread(CFunctionMultiThread):


    def __init__(self, wsi_dataset_brevity:CDataWSIDatasetBrevity, valid_idx_list:list, pad_type:EnumPadType, export_type:EnumTypeOutputWSICrop, path_folder:str, file_export_type:EnumTypeExportImage=EnumTypeExportImage.Pkl, thread_num:int=1):

        super(CFunctionOutputWSIMultiThread, self).__init__(data_list=valid_idx_list, thread_num=thread_num, progress_log_info='', show_progress_log_info=True, show_thread_idx=False)

        self.__image:np.array = None
        self.__mask:np.array = None
        self.__pad_type:EnumPadType = None
        self.__out_folder:str = None
        self.__export_type:EnumTypeOutputWSICrop = None

        self.__path_folder:str = path_folder
        self.__wsi_dataset_brevity:CDataWSIDatasetBrevity = wsi_dataset_brevity
        self.__file_export_type = file_export_type

        self.ChangePadType(pad_type)
        self.ChangeExportType(export_type)

    def UserFunction(self, idx:int):

        if self.__export_type == EnumTypeOutputWSICrop.Image:
            self.__captaineOutputCropImages(idx)
        elif self.__export_type == EnumTypeOutputWSICrop.Mask:
            self.__captaineOutputCropMasks(idx)
        else:
            CLogging.LogWarning('Output unknown WSI type ! ')

    def ChangePadType(self, pad_type:EnumPadType):
        self.__pad_type = pad_type
        if pad_type is None:
            pad_type_name = ''
            output_folder = None
        else:
            pad_type_name = pad_type.name
            output_folder = os.path.join(self.__path_folder, pad_type_name)

        if self.__export_type == EnumTypeOutputWSICrop.Image:
            progress_log_info = 'Export valid images ... ' + pad_type_name + " " + self.__wsi_dataset_brevity.name
            if not output_folder is None:
                self.__out_folder = os.path.join(output_folder, 'images')
                if not os.path.exists(self.__out_folder):
                    os.makedirs(self.__out_folder)
        elif self.__export_type == EnumTypeOutputWSICrop.Mask:
            progress_log_info = 'Export valid masks ... ' + pad_type_name + " " + self.__wsi_dataset_brevity.name
            if not output_folder is None:
                self.__out_folder = os.path.join(output_folder, 'masks')
                if not os.path.exists(self.__out_folder):
                    os.makedirs(self.__out_folder)
        else:
            progress_log_info = 'Export none ... ' + pad_type_name

        self.count_idx = 0
        self.progress_log_info = progress_log_info

    def ChangeExportType(self, export_type:EnumTypeOutputWSICrop):

        if export_type == EnumTypeOutputWSICrop.Image:
            self.__export_type = export_type
            image = self.__image
            self.__image = None
            del image
            mask = self.__mask
            self.__mask = None
            del mask
            CFunctionMemory.CleanMemory()
            CLogging.LogToUser("Read image ... " + self.__wsi_dataset_brevity.image_path)
            CFunctionMemory.GetCurMemoryInfo()
            self.__image = self.__wsi_dataset_brevity.GetImage()
            CLogging.LogToUser("Finish read image. " + self.__wsi_dataset_brevity.image_path)
            CFunctionMemory.GetCurMemoryInfo()

        elif export_type == EnumTypeOutputWSICrop.Mask:
            image = self.__image
            self.__image = None
            del image
            mask = self.__mask
            self.__mask = None
            del mask
            self.__export_type = export_type
            CLogging.LogToUser("Read masks ... " + self.__wsi_dataset_brevity.mask_path)
            CFunctionMemory.GetCurMemoryInfo()

            self.__mask = self.__wsi_dataset_brevity.GetMask()

            CLogging.LogToUser("Finish read masks. " + self.__wsi_dataset_brevity.mask_path)
            CFunctionMemory.GetCurMemoryInfo()

        else:
            self.__export_type = None
            CLogging.LogWarning("Unknown type of WSI crop . ")
            image = self.__image
            self.__image = None
            del image
            mask = self.__mask
            self.__mask = None
            del mask
            CFunctionMemory.CleanMemory()

        self.ChangePadType(self.__pad_type)

    def __captaineOutputCropImages(self, idx:int):

        index = self.__wsi_dataset_brevity.matrix_type_idx.index(self.__pad_type)
        matrix_idx = self.__wsi_dataset_brevity.matrix_idx[index]

        image = self.__image

        pad_type = self.__pad_type
        out_folder = self.__out_folder

        self.row_size = self.__wsi_dataset_brevity.row_size
        self.col_size = self.__wsi_dataset_brevity.col_size
        self.name = self.__wsi_dataset_brevity.name

        item = self.data_list[idx]
        if ((matrix_idx[item[0], item[1], 3] == 0) or (matrix_idx[item[0], item[1], 9] == 0)):
            return
        if ((matrix_idx[item[0], item[1], 1] - matrix_idx[item[0], item[1], 0] != 0) or
                (matrix_idx[item[0], item[1], 5] - matrix_idx[item[0], item[1], 4] != 0) or
                (matrix_idx[item[0], item[1], 7] - matrix_idx[item[0], item[1], 6] != 0) or
                (matrix_idx[item[0], item[1], 11] - matrix_idx[item[0], item[1], 10] != 0)):
            tmp_row_idx_start = 0
            tmp_row_idx_end = self.row_size
            tmp_col_idx_start = 0
            tmp_col_idx_end = self.col_size

            if (matrix_idx[item[0], item[1], 1] - matrix_idx[item[0], item[1], 0] != 0):
                tmp_row_idx_start = matrix_idx[item[0], item[1], 1]
            if (matrix_idx[item[0], item[1], 5] - matrix_idx[item[0], item[1], 4] != 0):
                tmp_row_idx_end = self.row_size - matrix_idx[item[0], item[1], 5] % self.row_size
            if (matrix_idx[item[0], item[1], 7] - matrix_idx[item[0], item[1], 6] != 0):
                tmp_col_idx_start = matrix_idx[item[0], item[1], 7]
            if (matrix_idx[item[0], item[1], 11] - matrix_idx[item[0], item[1], 10] != 0):
                tmp_col_idx_end = self.col_size - matrix_idx[item[0], item[1], 11] % self.col_size
            tmp_image = np.zeros((self.row_size, self.col_size, 3), dtype=np.uint8)
            if ((matrix_idx[item[0], item[1], 2] == matrix_idx[item[0], item[1], 3]) or
                (matrix_idx[item[0], item[1], 8] == matrix_idx[item[0], item[1], 9])
                ):
                pass
            else:
                tmp_image[tmp_row_idx_start: tmp_row_idx_end, tmp_col_idx_start: tmp_col_idx_end, :] += image[
                                                                                                        matrix_idx[item[0], item[1], 2]: matrix_idx[item[0], item[1], 3],
                                                                                                        matrix_idx[item[0], item[1], 8]: matrix_idx[item[0], item[1], 9],
                                                                                                        :]

        else:
            # 不需要填充零的
            tmp_image = image[matrix_idx[item[0], item[1], 2]: matrix_idx[item[0], item[1], 3], matrix_idx[item[0], item[1], 8]: matrix_idx[item[0], item[1], 9], :]
        if self.__file_export_type == EnumTypeExportImage.Pkl:
            out_name = self.name + "_" + pad_type.name + "_" + "%d_%d" % (item[0], item[1]) + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value]
            CFunctionFileIO.WritePKL(out_folder, out_name, tmp_image)
        elif self.__file_export_type == EnumTypeExportImage.Tif:
            out_name = self.name + "_" + pad_type.name + "_" + "%d_%d" % (item[0], item[1]) + LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIF.value]
            CFunctionFileIO.WriteImage(out_folder, out_name, tmp_image)
        else:
            CLogging.LogWarning('Unknown export file type for crop ! ')

    def __captaineOutputCropMasks(self, idx:int):

        index = self.__wsi_dataset_brevity.matrix_type_idx.index(self.__pad_type)
        matrix_idx = self.__wsi_dataset_brevity.matrix_idx[index]

        mask = self.__mask

        pad_type = self.__pad_type
        out_folder = self.__out_folder

        self.row_size = self.__wsi_dataset_brevity.row_size
        self.col_size = self.__wsi_dataset_brevity.col_size
        self.name = self.__wsi_dataset_brevity.name

        item = self.data_list[idx]
        if ((matrix_idx[item[0], item[1], 3] == 0) or (matrix_idx[item[0], item[1], 9] == 0)):
            return
        if ((matrix_idx[item[0], item[1], 1] - matrix_idx[item[0], item[1], 0] != 0) or
                (matrix_idx[item[0], item[1], 5] - matrix_idx[item[0], item[1], 4] != 0) or
                (matrix_idx[item[0], item[1], 7] - matrix_idx[item[0], item[1], 6] != 0) or
                (matrix_idx[item[0], item[1], 11] - matrix_idx[item[0], item[1], 10] != 0)):
            tmp_row_idx_start = 0
            tmp_row_idx_end = self.row_size
            tmp_col_idx_start = 0
            tmp_col_idx_end = self.col_size

            if (matrix_idx[item[0], item[1], 1] - matrix_idx[item[0], item[1], 0] != 0):
                tmp_row_idx_start = matrix_idx[item[0], item[1], 1]
            if (matrix_idx[item[0], item[1], 5] - matrix_idx[item[0], item[1], 4] != 0):
                tmp_row_idx_end = self.row_size - matrix_idx[item[0], item[1], 5] % self.row_size
            if (matrix_idx[item[0], item[1], 7] - matrix_idx[item[0], item[1], 6] != 0):
                tmp_col_idx_start = matrix_idx[item[0], item[1], 7]
            if (matrix_idx[item[0], item[1], 11] - matrix_idx[item[0], item[1], 10] != 0):
                tmp_col_idx_end = self.col_size - matrix_idx[item[0], item[1], 11] % self.col_size
            tmp_mask = np.zeros((self.row_size, self.col_size, self.__mask.shape[2]), dtype=np.uint8)
            if ((matrix_idx[item[0], item[1], 2] == matrix_idx[item[0], item[1], 3]) or
                (matrix_idx[item[0], item[1], 8] == matrix_idx[item[0], item[1], 9])
                ):
                pass
            else:
                tmp_mask[tmp_row_idx_start: tmp_row_idx_end, tmp_col_idx_start: tmp_col_idx_end, :] += mask[
                                                                                                     matrix_idx[item[0], item[1], 2]: matrix_idx[item[0], item[1], 3],
                                                                                                     matrix_idx[item[0], item[1], 8]: matrix_idx[item[0], item[1], 9],
                                                                                                     :]
        else:
            # 不需要填充零的
            tmp_mask = mask[matrix_idx[item[0], item[1], 2]: matrix_idx[item[0], item[1], 3], matrix_idx[item[0], item[1], 8]: matrix_idx[item[0], item[1], 9], :]
        if self.__file_export_type == EnumTypeExportImage.Pkl:
            out_name = self.name + "_" + pad_type.name + "_" + "%d_%d" % (item[0], item[1]) + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value]
            CFunctionFileIO.WritePKL(out_folder, out_name, tmp_mask)
        elif self.__file_export_type == EnumTypeExportImage.Tif:
            out_name = self.name + "_" + pad_type.name + "_" + "%d_%d" % (item[0], item[1]) + LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIF.value]
            CFunctionFileIO.WriteImage(out_folder, out_name, tmp_mask)
        else:
            CLogging.LogWarning('Unknown export file type for crop ! ')

