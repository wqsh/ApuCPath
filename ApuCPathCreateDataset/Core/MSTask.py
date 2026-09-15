from Core.MSDataPack import CDataPack, EnumTypeExportPad

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.WSIProcess.MSFunctionGenerateWSIDataset import CFunctionGenerateWSIDataset
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import EnumDatasetType
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from Core.MSSystem import FOLDER_WSI_INFO, FOLDER_DATA

from ApuCellGlobal.WSIProcess.MSDataWSIDataset import CDataWSIDatasetBrevity
from ApuCellGlobal.MaskRCNN.DataSet.MSDataMaskRCNNDataSet import EnumPadType

import os

class CTaskCreateTrainDataset:

    def __init__(self, inputDP:CDataPack):

        self.dp = inputDP
        self.output_path:str = os.path.join(self.dp.cfg.A4_FOLDER_EXPORT, self.dp.cfg.A5_NAME_DATASET  + EnumDatasetType.Train.name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def do(self):
        # 先把图像和mask关系绑定
        output_file_list = []
        image_list = self.dp.cfg.A2_LIST_IMAGE_PATH.split(';')
        mask_list = self.dp.cfg.A3_LIST_MASK_PATH.split(";")
        if self.dp.cfg.B2_MASK_RATIO_MIN < 0.0001:
            self.dp.cfg.B2_MASK_RATIO_MIN = 0.0001
            CLogging.LogWarning('Train Dataset must choose mask ratio large than 0 ! ')
        if len(image_list) != len(mask_list):
            CLogging.LogError("The number of image and mask is not matched ! ")

        for image_idx, image_path in enumerate(image_list):
            if len(image_path) == 0:
                continue
            if not os.path.exists(image_path):
                CLogging.LogForLackPath(image_path)
            mask_path = mask_list[image_idx]
            output_file_path = CFunctionGenerateWSIDataset.GenerateDatasetOneImage(image_path, mask_path, os.path.join(self.output_path, FOLDER_WSI_INFO), self.dp.cfg.B1_IMAGE_DIM, self.dp.cfg.B1_IMAGE_DIM, EnumDatasetType.Train)
            output_file_list.append(output_file_path)
            wsi_dataset: CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(output_file_path)
            if self.dp.cfg.B5_EXPORT_PAD_TYPE == EnumTypeExportPad.Org:
                wsi_dataset.OutputImageMaskValidDataset(os.path.join(self.output_path, FOLDER_DATA), EnumPadType.Org, self.dp.cfg.B3_MASK_RATIO_MAX, self.dp.cfg.B2_MASK_RATIO_MIN, self.dp.cfg.B4_EXPORT_TYPE)
            elif self.dp.cfg.B5_EXPORT_PAD_TYPE == EnumTypeExportPad.All:
                wsi_dataset.OutputImageMaskValidDatasetFullPadType(os.path.join(self.output_path, FOLDER_DATA), self.dp.cfg.B3_MASK_RATIO_MAX, self.dp.cfg.B2_MASK_RATIO_MIN, self.dp.cfg.B4_EXPORT_TYPE)

        CFunctionFileIO.WritePKL(self.dp.cfg.A4_FOLDER_EXPORT, self.dp.cfg.A5_NAME_DATASET + '.' + EnumDatasetType.Train.name, output_file_list)

class CTaskCreateValidDataset:

    def __init__(self, inputDP:CDataPack):

        self.dp = inputDP
        self.output_path:str = os.path.join(self.dp.cfg.A4_FOLDER_EXPORT, self.dp.cfg.A5_NAME_DATASET  + EnumDatasetType.Valid.name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def do(self):
        # 先把图像和mask关系绑定
        output_file_list = []
        image_list = self.dp.cfg.A2_LIST_IMAGE_PATH.split(';')
        mask_list = self.dp.cfg.A3_LIST_MASK_PATH.split(";")
        if self.dp.cfg.B2_MASK_RATIO_MIN < 0.0001:
            self.dp.cfg.B2_MASK_RATIO_MIN = 0.0001
            CLogging.LogWarning('Train Dataset must choose mask ratio large than 0 ! ')
        if len(image_list) != len(mask_list):
            CLogging.LogError("The number of image and mask is not matched ! ")

        for image_idx, image_path in enumerate(image_list):
            if len(image_path) == 0:
                continue
            if not os.path.exists(image_path):
                CLogging.LogForLackPath(image_path)
            mask_path = mask_list[image_idx]
            output_file_path = CFunctionGenerateWSIDataset.GenerateDatasetOneImage(image_path, mask_path, os.path.join(self.output_path, FOLDER_WSI_INFO), self.dp.cfg.B1_IMAGE_DIM, self.dp.cfg.B1_IMAGE_DIM, EnumDatasetType.Train)
            output_file_list.append(output_file_path)
            wsi_dataset: CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(output_file_path)
            if self.dp.cfg.B5_EXPORT_PAD_TYPE == EnumTypeExportPad.Org:
                wsi_dataset.OutputImageMaskValidDataset(os.path.join(self.output_path, FOLDER_DATA), EnumPadType.Org, self.dp.cfg.B3_MASK_RATIO_MAX, self.dp.cfg.B2_MASK_RATIO_MIN, self.dp.cfg.B4_EXPORT_TYPE)
            elif self.dp.cfg.B5_EXPORT_PAD_TYPE == EnumTypeExportPad.All:
                wsi_dataset.OutputImageMaskValidDatasetFullPadType(os.path.join(self.output_path, FOLDER_DATA), self.dp.cfg.B3_MASK_RATIO_MAX, self.dp.cfg.B2_MASK_RATIO_MIN, self.dp.cfg.B4_EXPORT_TYPE)

        CFunctionFileIO.WritePKL(self.dp.cfg.A4_FOLDER_EXPORT, self.dp.cfg.A5_NAME_DATASET + '.' + EnumDatasetType.Valid.name, output_file_list)

class CTaskCreateAssistDataset:

    def __init__(self, inputDP:CDataPack):

        self.dp = inputDP
        self.output_path:str = os.path.join(self.dp.cfg.A4_FOLDER_EXPORT, self.dp.cfg.A5_NAME_DATASET + EnumDatasetType.Assist.name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def do(self):
        # 先把图像和mask关系绑定
        output_file_list = []
        image_list = self.dp.cfg.A2_LIST_IMAGE_PATH.split(';')
        mask_list = self.dp.cfg.A3_LIST_MASK_PATH.split(";")
        if self.dp.cfg.B2_MASK_RATIO_MIN < 0:
            self.dp.cfg.B2_MASK_RATIO_MIN = 0
            CLogging.LogWarning('Train Dataset must choose mask ratio large than 0 ! ')
        if len(image_list) != len(mask_list):
            CLogging.LogError("The number of image and mask is not matched ! ")

        for image_idx, image_path in enumerate(image_list):
            if len(image_path) == 0:
                continue
            if not os.path.exists(image_path):
                CLogging.LogForLackPath(image_path)
            mask_path = mask_list[image_idx]
            output_file_path = CFunctionGenerateWSIDataset.GenerateDatasetOneImage(image_path, mask_path, os.path.join(self.output_path, FOLDER_WSI_INFO), self.dp.cfg.B1_IMAGE_DIM, self.dp.cfg.B1_IMAGE_DIM, EnumDatasetType.Assist)
            output_file_list.append(output_file_path)
            wsi_dataset: CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(output_file_path)
            if self.dp.cfg.B5_EXPORT_PAD_TYPE == EnumTypeExportPad.Org:
                wsi_dataset.OutputImageMaskValidDataset(os.path.join(self.output_path, FOLDER_DATA), EnumPadType.Org, self.dp.cfg.B3_MASK_RATIO_MAX, self.dp.cfg.B2_MASK_RATIO_MIN, self.dp.cfg.B4_EXPORT_TYPE)
            elif self.dp.cfg.B5_EXPORT_PAD_TYPE == EnumTypeExportPad.All:
                wsi_dataset.OutputImageMaskValidDatasetFullPadType(os.path.join(self.output_path, FOLDER_DATA), self.dp.cfg.B3_MASK_RATIO_MAX, self.dp.cfg.B2_MASK_RATIO_MIN, self.dp.cfg.B4_EXPORT_TYPE)

        CFunctionFileIO.WritePKL(self.dp.cfg.A4_FOLDER_EXPORT, self.dp.cfg.A5_NAME_DATASET + '.' + EnumDatasetType.Assist.name, output_file_list)

class CTaskCreatePredictDataset:

    def __init__(self, inputDP:CDataPack):

        self.dp = inputDP
        self.output_path:str = os.path.join(self.dp.cfg.A4_FOLDER_EXPORT, self.dp.cfg.A5_NAME_DATASET + EnumDatasetType.Predict.name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def do(self):

        output_file_path_list = []
        image_list = self.dp.cfg.A2_LIST_IMAGE_PATH.split(';')
        for image_path in image_list:
            if len(image_path) == 0:
                continue
            if not os.path.exists(image_path):
                CLogging.LogForLackPath(image_path)

            output_file_path = CFunctionGenerateWSIDataset.GenerateDatasetOneImage(image_path, None, os.path.join(self.output_path, FOLDER_WSI_INFO), self.dp.cfg.B1_IMAGE_DIM, self.dp.cfg.B1_IMAGE_DIM, EnumDatasetType.Predict)
            output_file_path_list.append(output_file_path)
            wsi_dataset: CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(output_file_path)
            if self.dp.cfg.B5_EXPORT_PAD_TYPE == EnumTypeExportPad.Org:
                wsi_dataset.OutputCropImage(os.path.join(self.output_path, FOLDER_DATA), EnumPadType.Org, self.dp.cfg.B4_EXPORT_TYPE)
            elif self.dp.cfg.B5_EXPORT_PAD_TYPE == EnumTypeExportPad.All:
                wsi_dataset.OutputCropImageFullPadType(os.path.join(self.output_path, FOLDER_DATA), self.dp.cfg.B4_EXPORT_TYPE)
            else:
                wsi_dataset.OutputCropImage(os.path.join(self.output_path, FOLDER_DATA), EnumPadType.Org, self.dp.cfg.B4_EXPORT_TYPE)
        CFunctionFileIO.WritePKL(self.dp.cfg.A4_FOLDER_EXPORT, self.dp.cfg.A5_NAME_DATASET + "." + EnumDatasetType.Predict.name, output_file_path_list)