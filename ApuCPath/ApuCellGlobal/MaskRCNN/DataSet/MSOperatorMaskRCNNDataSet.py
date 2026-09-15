from ApuCellGlobal.MSApuCellExternalPackge import *
from ApuCellGlobal.MaskRCNN.DataSet.MSDataMaskRCNNDataSet import CDataMaskRCNNPredictDataSet, EnumPadType
from ApuCellGlobal.GeneralFunction.MSFunctionImageMerge import CFunctionImageSplitMerge
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory

class COperatorMaskRCNNPredict:

    def __init__(self, dataset:CDataMaskRCNNPredictDataSet):

        self.__dataset:CDataMaskRCNNPredictDataSet = dataset

    def AddImage(self, path_image:str, name:str):

        if os.path.exists(path_image):
            self.__dataset.list_path_input_image.append(path_image)
            self.__dataset.list_file_name.append(os.path.basename(path_image))
            self.__dataset.list_name.append(name)
        else:
            CLogging.LogWarning("FIle is not exist ! " + path_image)

    def CreatDataSet(self):
        CLogging.LogToUser("Start create predict dataset ... ")
        for idx, file_name in enumerate(self.__dataset.list_file_name):
            file_path = self.__dataset.list_path_input_image[idx]
            out_folder_name = self.__dataset.list_name[idx]
            out_path:str = os.path.join(self.__dataset.folder_output, out_folder_name)

            img = CFunctionFileIO.ReadImage(file_path)
            x_min = 0
            x_max = img.shape[1]
            y_min = 0
            y_max = img.shape[0]

            row_num = math.ceil((y_max - y_min) / self.__dataset.row_size)
            col_num = math.ceil((x_max - x_min) / self.__dataset.col_size)
            row_length = self.__dataset.row_size * row_num
            col_length = self.__dataset.col_size * col_num
            new_img = np.zeros((self.__dataset.row_size * row_num, self.__dataset.col_size * col_num, 3), dtype=np.uint8)
            new_img[y_min: y_max, x_min: x_max, :] = img[y_min: y_max, x_min: x_max, :]

            for pad_type in self.__dataset.pad_type_list:
                CLogging.LogToUser("Create dataset : " + out_folder_name + " " + pad_type.name)

                if pad_type == EnumPadType.Org:
                    pad_img = new_img
                    out_path_pad = os.path.join(out_path, EnumPadType.Org.name)

                elif pad_type == EnumPadType.Horizontal:
                    pad_img = np.zeros((row_length, self.__dataset.col_size * (col_num + 1), 3), dtype=np.uint8)
                    pad_img[: row_length, self.__dataset.half_col_size: col_length + self.__dataset.half_col_size, :] += new_img[:, :, :]
                    out_path_pad = os.path.join(out_path, EnumPadType.Horizontal.name)

                elif pad_type == EnumPadType.Vertical:
                    pad_img = np.zeros((self.__dataset.row_size * (row_num + 1), col_length, 3), dtype=np.uint8)
                    pad_img[self.__dataset.half_row_size: row_length + self.__dataset.half_row_size, : col_length, :] = new_img[:, :, :]
                    out_path_pad = os.path.join(out_path, EnumPadType.Vertical.name)

                elif pad_type == EnumPadType.Diagonal:
                    pad_img = np.zeros((self.__dataset.row_size * (row_num + 1), self.__dataset.col_size * (col_num + 1), 3), dtype=np.uint8)
                    pad_img[self.__dataset.half_row_size: row_length + self.__dataset.half_row_size, self.__dataset.half_col_size: col_length + self.__dataset.half_col_size, :] = new_img[:, :, :]
                    out_path_pad = os.path.join(out_path, EnumPadType.Diagonal.name)
                else:
                    CLogging.LogWarning("Unknown pad type !")
                    continue
                out_path_pad = os.path.join(out_path_pad, "images")
                if not os.path.exists(out_path_pad):
                    os.makedirs(out_path_pad)
                CFunctionImageSplitMerge.SplitImage(pad_img, out_path_pad, file_name, row_size=self.__dataset.row_size, col_size=self.__dataset.col_size)
                CFunctionMemory.DeleteVariableMemory(pad_img)
        CLogging.LogToUser("Finish create predict dataset. ")

