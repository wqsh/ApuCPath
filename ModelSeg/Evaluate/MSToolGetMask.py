from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.Route.MSFunctionDataJsonImageLabel import CFunctionDataJsonImageLabel
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import CDataWSIDatasetBrevity
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import EnumDatasetType
from ApuCellGlobal.MaskRCNN.DataSet.MSDataMaskRCNNDataSet import EnumPadType

from ApuCPathCreateDataset.MSSystem import FOLDER_DATA, FOLDER_IMAGE, FOLDER_MASK

from Core.MSDataModel import CDataModelDataset

from Evaluate.MSFunctionSegmentEvaluation import CFunctionWSIInstance

import os

def GetDataset(path_wsi_dataset:str, wsi_name:str):

    img_full_name = os.path.basename(path_wsi_dataset)
    img_name = img_full_name.split('.')[0]
    wsi_dataset_info:CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(path_wsi_dataset)
    img_data_folder = os.path.join(os.path.dirname(os.path.dirname(path_wsi_dataset)), FOLDER_DATA, img_name)
    class_name_list = ["Pos", "Neg"]
    if (wsi_dataset_info.dataset_type == EnumDatasetType.Predict.value):
        dataset_list = []
        for pad_type in wsi_dataset_info.matrix_type_idx:
            folder_pad_image = os.path.join(img_data_folder, pad_type.name, 'images')
            if not os.path.exists(folder_pad_image):
                continue
            list_image_name = os.listdir(folder_pad_image)
            list_path_image = []
            for image_name in list_image_name:
                list_path_image.append(os.path.join(folder_pad_image, image_name))

            dataset = CDataModelDataset(wsi_dataset_info.row_size, class_name_list, wsi_name, pad_type.name)
            dataset.CreatePredictDatasets(list_path_image)
            dataset.prepare()
            dataset_list.append(dataset)
        return dataset_list
    elif (wsi_dataset_info.dataset_type == EnumDatasetType.Assist.value):
        dataset_list = []
        for pad_type in wsi_dataset_info.matrix_type_idx:
            folder_pad_image = os.path.join(img_data_folder, pad_type.name, 'images')
            if not os.path.exists(folder_pad_image):
                continue
            list_image_name = os.listdir(folder_pad_image)
            list_path_image = []
            for image_name in list_image_name:
                list_path_image.append(os.path.join(folder_pad_image, image_name))

            dataset = CDataModelDataset(wsi_dataset_info.row_size, class_name_list, wsi_name, pad_type.name)
            dataset.CreatePredictDatasets(list_path_image)
            dataset.prepare()
            dataset_list.append(dataset)
        return dataset_list
    elif (wsi_dataset_info.dataset_type == EnumDatasetType.Train.value):
        dataset_list = []
        for pad_type in wsi_dataset_info.matrix_type_idx:
            if not os.path.exists(os.path.join(img_data_folder, pad_type.name)):
                continue
            folder_pad_image = os.path.join(img_data_folder, pad_type.name, FOLDER_IMAGE)
            if not os.path.exists(folder_pad_image):
                CLogging.LogWarning('Path is not exist ; ' + folder_pad_image)
                continue
            list_image_name = os.listdir(folder_pad_image)

            folder_pad_mask = os.path.join(img_data_folder, pad_type.name, FOLDER_MASK)
            list_mask_name = os.listdir(folder_pad_mask)

            if len(list_mask_name) != len(list_image_name):
                CLogging.LogError("The num of images and masks from ApuCPathDataset is not match !")

            list_image_name.sort()
            list_mask_name.sort()

            list_path_image = []
            list_path_mask = []
            for idx, image_name in enumerate(list_image_name):
                mask_name = list_mask_name[idx]
                if mask_name.split('.')[0] != image_name.split('.')[0]:
                    CLogging.LogError("The name of image and mask is not match !")
                list_path_image.append(os.path.join(folder_pad_image, image_name))
                list_path_mask.append(os.path.join(folder_pad_mask, mask_name))
            dataset = CDataModelDataset(wsi_dataset_info.row_size, class_name_list, wsi_name, pad_type.name)
            dataset.CreateTrainDatasets(list_path_image, list_path_mask)
            dataset.prepare()
            dataset_list.append(dataset)
        return dataset_list

def GetAnnotationMask(path_image:str="", path_org_json:str="", x_length_large:int=0,y_length_large:int=0):
    img_org_mask_pos = CFunctionDataJsonImageLabel.GetJson2MaskForSimple(path_image, path_org_json, "Pos", 1, x_length_large, y_length_large)
    CLogging.LogToUser(" Get pos")
    CFunctionMemory.GetCurMemoryInfo()

    img_org_mask_neg = CFunctionDataJsonImageLabel.GetJson2MaskForSimple(path_image, path_org_json, "Neg", 2, x_length_large, y_length_large)
    CLogging.LogToUser(" Get neg")
    CFunctionMemory.GetCurMemoryInfo()

    CFunctionToolImageC.AddMatrixValue(img_org_mask_pos, img_org_mask_neg)
    del img_org_mask_neg
    CFunctionMemory.CleanMemory()
    CFunctionToolImageC.FilterMatrixValueBySmall(img_org_mask_pos, 1)
    CLogging.LogToUser("Finish get label pos")
    CFunctionMemory.GetCurMemoryInfo()
    CFunctionToolImageC.SetMatrixValue(img_org_mask_pos, 255)
    return img_org_mask_pos

def GetPredictMultiMask(path_image:str="", path_predict_json:str="", label_name="label"):

    img_predict_mask = CFunctionDataJsonImageLabel.GetJson2Mask(path_image, path_predict_json, label_name, 255)
    return img_predict_mask