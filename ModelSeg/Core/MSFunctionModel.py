from Core.MSDataPack import CDataPack, EnumTypeModelInit, EnumTypeModelType
from Core.MSDataModel import CDataModelDataset
from Core.MSSystem import FOLDER_PREDICT, FOLDER_TRAIN_RESULT
from Core.MSFunctionConfig import CFunctionConfig

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging, CLogProgressPercent
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralSystem.MSSystem import NAME_CONFIG
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import EnumDatasetType, CDataWSIDatasetBrevity, WSI_DATASET_FOLDER_IMAGE, WSI_DATASET_FOLDER_MASK, WSI_DATASET_FOLDER_NEGATIVEIMAGE
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralFunction.MSFunctionMultiThread import CFunctionMultiThread
from ApuCellGlobal.MaskRCNN.DataSet.MSDataMaskRCNNDataSet import EnumPadType
from ApuCellGlobal.GeneralSystem.MSSystemEnum import EnumVisualFileType, EnumDataFileType, LIST_DATA_FILE_TYPE_SUFFIX, LIST_VISUAL_FILE_TYPE_SUFFIX, LIST_DATA_FILE_TYPE, LIST_VISUAL_FILE_TYPE
from ApuCPathCreateDataset.MSSystem import FOLDER_DATA
from ApuCellGlobal.WSIProcess.MSFunctionImageColorClusterSample import CFunctionImageColorClusterSample

from Model.maskrcnn.maskrcnn_model import ModelMaskRCNN
from Model.maskrcnn.maskrcnnedge_model import ModelMaskRCNNWithEdge
from Model.maskrcnn.maskrcnn_backgroundedg_model import ModelMaskRCNNWithEdgeBGClassifier
from Model.maskrcnn.maskrcnn_bgclassifier_model import ModelMaskRCNNBGClassifier
from Model.UNIMask2Former.unimask2former_model import ModelUNIMask2Former
from Model.maskrcnn.config import CConfigParameterMaskRCNN
from Model.UNIMask2Former.config import CConfigParameterUNIMask2Former

from Model.unet.unet_model import ModelUnet
from Model.unet.config import CConfigParameterUnet

import os
import numpy as np
if os.name == 'posix':
    import matplotlib
    matplotlib.use('Agg')
import random
import math
from threading import Thread
import torch
import copy

from Core.MSSystem import FOLDER_TORCH_CACHE


class CFunctionModelDataset():

    def __init__(self, inputDP:CDataPack):
        self.dp:CDataPack = inputDP

    def GetDataset(self, path_wsi_dataset:str, wsi_name:str, add_neg:bool=False):

        img_full_name = os.path.basename(path_wsi_dataset)
        img_name = img_full_name.split('.')[0]
        wsi_dataset_info:CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(path_wsi_dataset)
        img_data_folder = os.path.join(os.path.dirname(os.path.dirname(path_wsi_dataset)), FOLDER_DATA, img_name)
        if (wsi_dataset_info.dataset_type == EnumDatasetType.Predict):
            dataset_list = []
            for pad_type in wsi_dataset_info.matrix_type_idx:
                folder_pad_image = os.path.join(img_data_folder, pad_type.name, WSI_DATASET_FOLDER_IMAGE)
                if not os.path.exists(folder_pad_image):
                    continue
                list_image_name = os.listdir(folder_pad_image)
                list_path_image = []
                for image_name in list_image_name:
                    list_path_image.append(os.path.join(folder_pad_image, image_name))

                dataset = CDataModelDataset(wsi_dataset_info.row_size, wsi_dataset_info.class_ids, wsi_name, pad_type.name)
                dataset.CreatePredictDatasets(list_path_image)
                dataset.prepare()
                dataset_list.append(dataset)
            return dataset_list
        elif (wsi_dataset_info.dataset_type == EnumDatasetType.Assist):
            dataset_list = []
            for pad_type in wsi_dataset_info.matrix_type_idx:
                folder_pad_image = os.path.join(img_data_folder, pad_type.name, WSI_DATASET_FOLDER_IMAGE)
                if not os.path.exists(folder_pad_image):
                    continue
                list_image_name = os.listdir(folder_pad_image)
                list_path_image = []
                for image_name in list_image_name:
                    list_path_image.append(os.path.join(folder_pad_image, image_name))

                dataset = CDataModelDataset(wsi_dataset_info.row_size, wsi_dataset_info.class_ids, wsi_name, pad_type.name)
                dataset.CreatePredictDatasets(list_path_image)
                dataset.prepare()
                dataset_list.append(dataset)
            return dataset_list
        elif (wsi_dataset_info.dataset_type == EnumDatasetType.Train) or (wsi_dataset_info.dataset_type == EnumDatasetType.Valid):
            dataset_list = []
            for pad_type in wsi_dataset_info.matrix_type_idx:
                if not os.path.exists(os.path.join(img_data_folder, pad_type.name)):
                    continue
                folder_pad_image = os.path.join(img_data_folder, pad_type.name, WSI_DATASET_FOLDER_IMAGE)
                if not os.path.exists(folder_pad_image):
                    CLogging.LogWarning('Path is not exist ; ' + folder_pad_image)
                    continue
                list_image_name = os.listdir(folder_pad_image)

                folder_pad_mask = os.path.join(img_data_folder, pad_type.name, WSI_DATASET_FOLDER_MASK)
                list_mask_name = os.listdir(folder_pad_mask)

                if len(list_mask_name) != len(list_image_name):
                    CLogging.LogError("The num of images and masks from ApuCPathDataset is not match !")
                else:
                    pass
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

                if len(list_mask_name) != len(list_image_name):
                    CLogging.LogError("The num of images and masks from ApuCPathDataset is not match !")

                list_image_name.sort()
                list_mask_name.sort()

                if add_neg:
                    if pad_type == EnumPadType.Org:

                        folder_negative_image = os.path.join(img_data_folder, pad_type.name, WSI_DATASET_FOLDER_NEGATIVEIMAGE)
                        list_negative_name = os.listdir(folder_negative_image)

                        for idx, negative_name in enumerate(list_negative_name):
                            path_nagative = os.path.join(folder_negative_image, negative_name)
                            list_path_image.append(path_nagative)
                            list_path_mask.append(None)

                        if len(list_mask_name) != len(list_image_name):
                            CLogging.LogError("The num of images and masks from ApuCPathDataset is not match !")
                    else:
                        pass
                else:
                    pass
                dataset = CDataModelDataset(wsi_dataset_info.row_size, wsi_dataset_info.class_ids, wsi_name, pad_type.name)
                dataset.CreateTrainDatasets(list_path_image, list_path_mask)
                dataset.prepare()
                dataset_list.append(dataset)
            return dataset_list
        else:
            dataset_list = []
            return dataset_list

    def MergeDatasetList(self, dataset_list):
        if len(dataset_list) == 0:
            return dataset_list
        class_ids = []
        for dataset in dataset_list:
            for class_name in dataset.class_names:
                if class_name in class_ids:
                    continue
                else:
                    class_ids.append(class_name)
        merge_dataset:CDataModelDataset = CDataModelDataset(self.dp.cfg.B1_IMAGE_DIM, class_ids, "TrainImageDataset")
        if self.dp.cfg.A7_SKIP_CHECK_MASK:
            CLogging.LogToUser("Skip the mask check. ")
        for dataset in dataset_list:
            path_info = dataset.GetImageAndMaskPathList()
            list_valid_info_img = []
            list_valid_info_mask = []
            for idx, path_img in enumerate(path_info[0]):
                path_mask = path_info[1][idx]
                if not self.dp.cfg.A7_SKIP_CHECK_MASK:
                    if path_mask is None:
                        pass
                    else:
                        if not os.path.exists(path_info[0][idx]):
                            continue
                        if path_mask.endswith(LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value]):
                            mask = CFunctionFileIO.ReadPKL(path_mask)
                        elif (path_mask.endswith(LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIF.value]) or
                            path_mask.endswith(LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value]) or
                            path_mask.endswith(LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.PNG.value]) or
                            path_mask.endswith(LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.JPG.value])
                            ):
                            mask = CFunctionFileIO.ReadImage(path_mask)
                        else:
                            CLogging.LogError("Mask is not support type! ")
                        valid_mask_idx = np.sum(mask, axis=(0, 1)) > 0
                        if True not in valid_mask_idx:
                            CLogging.LogWarning("Lack the valid mask, skip the image : " + path_img)
                            continue
                list_valid_info_img.append(path_img)
                list_valid_info_mask.append(path_mask)
            merge_dataset.CreateTrainDatasets(list_valid_info_img, list_valid_info_mask, dataset.dataset_name)
        merge_dataset.prepare()
        return merge_dataset

    def RandomTrainValidDataset(self, input_dataset:CDataModelDataset, valid_percent:float=0.2):

        class_ids = []

        for class_name in input_dataset.class_names:
            if class_name in class_ids:
                continue
            else:
                class_ids.append(class_name)

        train_dataset:CDataModelDataset = CDataModelDataset(self.dp.cfg.B1_IMAGE_DIM, class_ids, "TrainImageDataset")
        valid_dataset:CDataModelDataset = CDataModelDataset(self.dp.cfg.B1_IMAGE_DIM, class_ids, "ValidImageDataset")

        images_num = input_dataset.num_images
        valid_images_num = int(images_num * valid_percent)
        valid_images_idx_list = random.sample(range(0, images_num), valid_images_num)
        images_distribute_list = np.zeros(images_num, dtype=np.uint8)
        images_distribute_list[valid_images_idx_list] = 1
        # train_images_idx_list = np.where(images_distribute_list == 0)[0].tolist()

        list_train_info_img = []
        list_train_info_mask = []
        list_valid_info_img = []
        list_valid_info_mask = []

        path_info = input_dataset.GetImageAndMaskPathList()
        for idx, path_img in enumerate(path_info[0]):
            path_mask = path_info[1][idx]
            if not os.path.exists(path_info[0][idx]):
                continue
            if images_distribute_list[idx] == 0:
                list_train_info_img.append(path_img)
                list_train_info_mask.append(path_mask)
            else:
                list_valid_info_img.append(path_img)
                list_valid_info_mask.append(path_mask)
        train_dataset.CreateTrainDatasets(list_train_info_img, list_train_info_mask)
        valid_dataset.CreateTrainDatasets(list_valid_info_img, list_valid_info_mask)

        train_dataset.prepare()
        valid_dataset.prepare()

        return train_dataset, valid_dataset

    def KFoldDataset(self, input_dataset:CDataModelDataset, k_fold:int=5):

        images_num = input_dataset.num_images
        valid_percent = 1 / k_fold

        path_info = input_dataset.GetImageAndMaskPathList()
        valid_images_num = math.floor(images_num * valid_percent)

        k_fold_dataset_list = []
        k_folder_idx = 0
        k_folder_valid_idx_list = [0]
        for k in range(1, k_fold + 1):
            k_folder_valid_idx_list.append(k * valid_images_num)
        for k in range(1, k_fold + 1):
            train_dataset: CDataModelDataset = CDataModelDataset(self.dp.cfg.B1_IMAGE_DIM, self.class_name_list, "TrainImageDataset")
            valid_dataset: CDataModelDataset = CDataModelDataset(self.dp.cfg.B1_IMAGE_DIM, self.class_name_list, "ValidImageDataset")

            if k == 1:
                list_train_images = path_info[0][k_folder_valid_idx_list[k]: ]
                list_train_masks = path_info[1][k_folder_valid_idx_list[k]: ]

                list_valid_images = path_info[0][k_folder_valid_idx_list[k - 1]: k_folder_valid_idx_list[k]]
                list_valid_masks = path_info[1][k_folder_valid_idx_list[k - 1]: k_folder_valid_idx_list[k]]

            elif k == k_fold:
                list_train_images = path_info[0][: k_folder_valid_idx_list[k - 1]]
                list_train_masks = path_info[1][: k_folder_valid_idx_list[k - 1]]

                list_valid_images = path_info[0][k_folder_valid_idx_list[k - 1]: ]
                list_valid_masks = path_info[1][k_folder_valid_idx_list[k - 1]: ]

            else:
                list_train_images = path_info[0][: k_folder_valid_idx_list[k - 1]] +path_info[0][k_folder_valid_idx_list[k]:]
                list_train_masks = path_info[1][: k_folder_valid_idx_list[k - 1]] + path_info[1][k_folder_valid_idx_list[k]:]

                list_valid_images = path_info[0][k_folder_valid_idx_list[k - 1]: k_folder_valid_idx_list[k]]
                list_valid_masks = path_info[1][k_folder_valid_idx_list[k - 1]: k_folder_valid_idx_list[k]]

            train_dataset.CreateTrainDatasets(list_train_images, list_train_masks)
            valid_dataset.CreateTrainDatasets(list_valid_images, list_valid_masks)

            train_dataset.prepare()
            valid_dataset.prepare()
            CLogging.LogToUser("Train images: %d Valid images: %d"%(train_dataset.num_images, valid_dataset.num_images))

            k_fold_dataset_list.append([train_dataset, valid_dataset])
        return k_fold_dataset_list

    def SetImagesToDataset(self, dataset:CDataModelDataset, path_image_name:str):

        org_image_list = []
        for image_info in dataset.image_info:
            org_image_list.append(image_info["path"])
        new_dataset = CDataModelDataset(dataset.image_dim, dataset.class_names[1:], None, None)
        with open(path_image_name, 'r') as f:
            buffer = f.readlines()
        for line in buffer:
            if len(line.strip()) == 0:
                continue
            list_line = line.strip().split('\t')
            path_img = list_line[0]
            path_mask = list_line[1]
            try:
                idx = org_image_list.index(path_img)
            except:
                CLogging.LogWarning("Cannot find the image: " + path_img)
                continue
            new_dataset.add_image("CPath", idx, path=path_img, width=dataset.image_dim, height=dataset.image_dim, mask=path_mask, classes=dataset.class_names)
        new_dataset.prepare()
        return new_dataset

    def GetDatasetWithSampleNegative(self, dataset:CDataModelDataset):

        path_info = dataset.GetImageAndMaskPathList()
        pos_img_num = 0
        list_neg_img_path = []
        list_pos_img_path = []
        list_pos_img_mask_path = []
        for idx, path_img in enumerate(path_info[0]):
            path_mask = path_info[1][idx]
            if path_mask is None:
                list_neg_img_path.append(path_img)
            else:
                pos_img_num += 1
                list_pos_img_path.append(path_img)
                list_pos_img_mask_path.append(path_mask)
        sample_neg_img_list = CFunctionImageColorClusterSample.GetsSamplePathList(list_neg_img_path, pos_img_num, self.dp.cfg.C5_MODEL_NEGATIVE_SAMPLE_CLUSTER)
        sample_neg_img_mask_list = [None for i in range(len(sample_neg_img_list))]

        list_img_path = list_pos_img_path + sample_neg_img_list
        list_img_mask_path = list_pos_img_mask_path + sample_neg_img_mask_list
        sample_dataset: CDataModelDataset = CDataModelDataset(self.dp.cfg.B1_IMAGE_DIM, dataset.class_names, "TrainImageDatasetSampleNegative")
        sample_dataset.CreateTrainDatasets(list_img_path, list_img_mask_path, dataset.dataset_name)
        sample_dataset.prepare()
        return sample_dataset

    def GetDatasetWithRandomNegative(self, dataset:CDataModelDataset):

        path_info = dataset.GetImageAndMaskPathList()
        pos_img_num = 0
        list_neg_img_path = []
        list_pos_img_path = []
        list_pos_img_mask_path = []
        for idx, path_img in enumerate(path_info[0]):
            path_mask = path_info[1][idx]
            if path_mask is None:
                list_neg_img_path.append(path_img)
            else:
                pos_img_num += 1
                list_pos_img_path.append(path_img)
                list_pos_img_mask_path.append(path_mask)
        sample_neg_img_list = random.sample(list_neg_img_path, min(pos_img_num, len(list_neg_img_path)))
        sample_neg_img_mask_list = [None for i in range(len(sample_neg_img_list))]

        list_img_path = list_pos_img_path + sample_neg_img_list
        list_img_mask_path = list_pos_img_mask_path + sample_neg_img_mask_list
        sample_dataset: CDataModelDataset = CDataModelDataset(self.dp.cfg.B1_IMAGE_DIM, dataset.class_names, "TrainImageDatasetRandomNegative")
        sample_dataset.CreateTrainDatasets(list_img_path, list_img_mask_path, dataset.dataset_name)
        sample_dataset.prepare()
        return sample_dataset

class CFunctionModel:

    def __init__(self, inputDP:CDataPack):

        self.dp:CDataPack = inputDP
        self.parameter_maskrcnn = CConfigParameterMaskRCNN()
        self.parameter_unet = CConfigParameterUnet()
        self.parameter_unimask2former = CConfigParameterUNIMask2Former()

        path_cache = os.path.join(os.getcwd(), FOLDER_TORCH_CACHE)
        if not os.path.exists(path_cache):
            os.makedirs(path_cache)
        torch.hub.set_dir(os.path.join(os.getcwd(), FOLDER_TORCH_CACHE))

    def _captainConfig2Paramter(self):

        if ((self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNN) or
            (self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNNBGClassifier)
        ):

            self.parameter_maskrcnn = CConfigParameterMaskRCNN()
            self.parameter_maskrcnn.BATCH_SIZE = self.dp.cfg.B4_BATCH_SIZE
            self.parameter_maskrcnn.LEARNING_RATE = self.dp.cfg.B6_LEARNING_RATE
            self.parameter_maskrcnn.MIN_SIZE = self.dp.cfg.B1_IMAGE_DIM
            self.parameter_maskrcnn.MAX_SIZE = self.dp.cfg.B1_IMAGE_DIM
            self.parameter_maskrcnn.NUM_EPOCHS = self.dp.cfg.B5_EPOCH_NUM
            self.parameter_maskrcnn.NUM_CLASSES = self.dp.cfg.B3_CLASSES_NUM
            self.parameter_maskrcnn.OUTPUT_ROOT = self.dp.cfg.A6_FOLDER_OUTPUT
            # TRAIN
            self.parameter_maskrcnn.RPN_PRE_NMS_TOP_N_TRAIN = self.dp.cfg.G1_BACKBONE_1_RPN_PRE_NMS_TOP_N_TRAIN
            self.parameter_maskrcnn.RPN_POST_NMS_TOP_N_TRAIN = self.dp.cfg.G1_BACKBONE_2_RPN_POST_NMS_TOP_N_TRAIN
            self.parameter_maskrcnn.RPN_FG_IOU_THRESH = self.dp.cfg.G1_BACKBONE_3_RPN_FG_IOU_THRESH
            self.parameter_maskrcnn.RPN_BG_IOU_THRESH = self.dp.cfg.G1_BACKBONE_4_RPN_BG_IOU_THRESH
            # TRAIN PREDICT
            self.parameter_maskrcnn.RPN_NMS_THRESH = self.dp.cfg.G2_BACKBONE_1_RPN_NMS_THRESH
            self.parameter_maskrcnn.MASK_ROI_POOL_SIZE = self.dp.cfg.G2_BACKBONE_2_MASK_ROI_POOL_SIZE
            # PREDICT
            self.parameter_maskrcnn.RPN_PRE_NMS_TOP_N_TEST = self.dp.cfg.G3_BACKBONE_1_RPN_PRE_NMS_TOP_N_TEST
            self.parameter_maskrcnn.RPN_POST_NMS_TOP_N_TEST = self.dp.cfg.G3_BACKBONE_2_RPN_POST_NMS_TOP_N_TEST
            self.parameter_maskrcnn.BOX_SCORE_THRESH = self.dp.cfg.G3_BACKBONE_3_BOX_SCORE_THRESH
            self.parameter_maskrcnn.BOX_NMS_THRESH = self.dp.cfg.G3_BACKBONE_4_BOX_NMS_THRESH
            self.parameter_maskrcnn.BOX_DETECTIONS_PER_IMG = self.dp.cfg.G3_BACKBONE_5_BOX_DETECTIONS_PER_IMG

            self.parameter_maskrcnn.USE_GPU = self.dp.cfg.Z1_USE_GPU
        elif (self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.Unet):
            self.parameter_unet.IMG_SIZE = self.dp.cfg.B1_IMAGE_DIM
            self.parameter_unet.BATCH_SIZE = self.dp.cfg.B4_BATCH_SIZE
            self.parameter_unet.LEARNING_RATE = self.dp.cfg.B6_LEARNING_RATE
            self.parameter_unet.NUM_EPOCHS = self.dp.cfg.B5_EPOCH_NUM
            self.parameter_unet.NUM_CLASSES = self.dp.cfg.B3_CLASSES_NUM
            self.parameter_unet.USE_GPU = self.dp.cfg.Z1_USE_GPU
        elif (self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.UNIMask2Former):
            self.parameter_unimask2former.PATCH_SIZE = self.dp.cfg.B1_IMAGE_DIM
            self.parameter_unimask2former.BATCH_SIZE = self.dp.cfg.B4_BATCH_SIZE
            self.parameter_unimask2former.LEARNING_RATE = self.dp.cfg.B6_LEARNING_RATE
            self.parameter_unimask2former.BACKBONE_FROZEN = True
            self.parameter_unimask2former.NUM_CLASSES = self.dp.cfg.B3_CLASSES_NUM
            self.parameter_unimask2former.NUM_EPOCHS = self.dp.cfg.B5_EPOCH_NUM
            if not os.path.exists(self.dp.cfg.G4_PATH_UNI_CHECKPOINT):
                CLogging.LogError("G4_PATH_UNI_CHECKPOINT is not exist! path: " + self.dp.cfg.G4_PATH_UNI_CHECKPOINT)
            self.parameter_unimask2former.UNI_CHECKPOINT = self.dp.cfg.G4_PATH_UNI_CHECKPOINT
            self.parameter_unimask2former.USE_GPU = self.dp.cfg.Z1_USE_GPU
            # self.parameter_unimask2former.D_MODEL = self.
        else:
            pass

class CFunctionModelTrain(CFunctionModel):

    def __init__(self, inputDP:CDataPack):

        super().__init__(inputDP)
        self.name_train_dataset_info = "train_dataset"
        self.name_valid_dataset_info = "valid_dataset"

        self._captainConfig2Paramter()

    def Train(self, dataset_train:CDataModelDataset, dataset_val:CDataModelDataset):

        CLogging.LogToUser("Init model ... ")
        CLogging.LogToUser("Use model: " + self.dp.cfg.C3_MODEL_TYPE.name)

        if self.dp.cfg.C1_MODEL_INIT == EnumTypeModelInit.Scratch:

            self.folder_output: str = os.path.join(self.dp.cfg.A6_FOLDER_OUTPUT, self.dp.cfg.C3_MODEL_TYPE.name, FOLDER_TRAIN_RESULT, CLogging.GetLocTimeStrForName())
            for train_class_idx, train_class_name in enumerate(dataset_train.class_names):
                val_dataset_name = dataset_val.class_names[train_class_idx]
                if train_class_name != val_dataset_name:
                    CLogging.LogError("Dataset label don't match! ")

            if not os.path.exists(self.folder_output):
                os.makedirs(self.folder_output)
            CFunctionConfig.Config2File(self.dp.cfg, self.folder_output, NAME_CONFIG)

            CFunctionFileIO.WritePKL(self.folder_output, self.name_train_dataset_info + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value], dataset_train)
            CFunctionFileIO.WritePKL(self.folder_output, self.name_valid_dataset_info + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value], dataset_val)

            CLogging.LogToUser("Init model ... ")
            CLogging.LogToUser("Use model: " + self.dp.cfg.C3_MODEL_TYPE.name)

            if self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNN:
                self.parameter_maskrcnn.OUTPUT_ROOT = self.folder_output
                self.parameter_maskrcnn.NUM_CLASSES = len(dataset_train.class_names)
                self.model = ModelMaskRCNN("train", self.parameter_maskrcnn, None, False,  True)
            elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNNBGClassifier:
                self.parameter_maskrcnn.OUTPUT_ROOT = self.folder_output
                self.parameter_maskrcnn.NUM_CLASSES = len(dataset_train.class_names)
                self.model = ModelMaskRCNNBGClassifier("train", self.parameter_maskrcnn, None, False, True)
            elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.Unet:
                self.parameter_unet.OUTPUT_ROOT = self.folder_output
                self.parameter_unet.NUM_CLASSES = len(dataset_train.class_names)
                self.model = ModelUnet("train", self.parameter_unet, None, False, True)
            elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.UNIMask2Former:
                self.parameter_unimask2former.OUTPUT_ROOT = self.folder_output
                self.parameter_unimask2former.NUM_CLASSES = len(dataset_train.class_names)
                self.model = ModelUNIMask2Former("train", self.parameter_unimask2former, None, False, True)
            else:
                CLogging.LogError("Unknow model type!")
        elif self.dp.cfg.C1_MODEL_INIT == EnumTypeModelInit.Finetune:
            if self.dp.cfg.F1_REPLY_TRAIN:
                if os.path.splitext(self.dp.cfg.C2_MODEL_PATH)[0].endswith("_best") or os.path.splitext(self.dp.cfg.C2_MODEL_PATH)[0].endswith("_latest"):
                    self.folder_output = os.path.join(os.path.dirname(self.dp.cfg.C2_MODEL_PATH))
                else:
                    self.folder_output = os.path.join(os.path.dirname(os.path.dirname(self.dp.cfg.C2_MODEL_PATH)))

                dataset_train = CFunctionFileIO.ReadPKL(os.path.join(self.folder_output, self.name_train_dataset_info + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value]))
                dataset_val = CFunctionFileIO.ReadPKL(os.path.join(self.folder_output, self.name_valid_dataset_info + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value]))


                if self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNN:
                    self.parameter_maskrcnn.OUTPUT_ROOT = self.folder_output
                    self.model = ModelMaskRCNN("train", self.parameter_maskrcnn, self.dp.cfg.C2_MODEL_PATH, False, False)
                elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNNBGClassifier:
                    self.parameter_maskrcnn.OUTPUT_ROOT = self.folder_output
                    self.model = ModelMaskRCNNBGClassifier("train", self.parameter_maskrcnn, self.dp.cfg.C2_MODEL_PATH, False, False)
                elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.Unet:
                    self.parameter_unet.OUTPUT_ROOT = self.folder_output
                    self.model = ModelUnet("train", self.parameter_unet, self.dp.cfg.C2_MODEL_PATH, False, False)
                else:
                    CLogging.LogError("Unknow model type!")
            else:

                self.folder_output: str = os.path.join(self.dp.cfg.A6_FOLDER_OUTPUT, self.dp.cfg.C3_MODEL_TYPE.name, FOLDER_TRAIN_RESULT, CLogging.GetLocTimeStrForName())

                for train_class_idx, train_class_name in enumerate(dataset_train.class_names):
                    val_dataset_name = dataset_val.class_names[train_class_idx]
                    if train_class_name != val_dataset_name:
                        CLogging.LogError("Dataset label don't match! ")

                CFunctionFileIO.WritePKL(self.folder_output, self.name_train_dataset_info + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value], dataset_train)
                CFunctionFileIO.WritePKL(self.folder_output, self.name_valid_dataset_info + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value], dataset_val)

                if not os.path.exists(self.folder_output):
                    os.makedirs(self.folder_output)
                CFunctionConfig.Config2File(self.dp.cfg, self.folder_output, NAME_CONFIG)

                if self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNN:
                    self.parameter_maskrcnn.OUTPUT_ROOT = self.folder_output
                    self.parameter_maskrcnn.NUM_CLASSES = len(dataset_train.class_names)
                    self.model = ModelMaskRCNN("train", self.parameter_maskrcnn, self.dp.cfg.C2_MODEL_PATH, False, True)
                elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNNBGClassifier:
                    self.parameter_maskrcnn.OUTPUT_ROOT = self.folder_output
                    self.parameter_maskrcnn.NUM_CLASSES = len(dataset_train.class_names)
                    self.model = ModelMaskRCNNBGClassifier("train", self.parameter_maskrcnn, self.dp.cfg.C2_MODEL_PATH, False, True)
                elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.Unet:
                    self.parameter_unet.OUTPUT_ROOT = self.folder_output
                    self.parameter_unet.NUM_CLASSES = len(dataset_train.class_names)
                    self.model = ModelUnet("train", self.parameter_unet, self.dp.cfg.C2_MODEL_PATH, False, True)
                elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.UNIMask2Former:
                    self.parameter_unimask2former.OUTPUT_ROOT = self.folder_output
                    self.parameter_unimask2former.NUM_CLASSES = len(dataset_train.class_names)
                    self.model = ModelUNIMask2Former("train", self.parameter_unimask2former, self.dp.cfg.C2_MODEL_PATH, False, True)
                else:
                    CLogging.LogError("Unknow model type!")
        else:
           CLogging.LogError("Unknow weights for loading! ")
        CLogging.LogToUser("Finish init model. ")

        CLogging.LogToUser("Train images: %d Valid images: %d"%(dataset_train.num_images, dataset_val.num_images))
        dataset_train.SetModleType(self.dp.cfg.C3_MODEL_TYPE)
        dataset_val.SetModleType(self.dp.cfg.C3_MODEL_TYPE)
        if self.dp.cfg.C4_MODEL_ADD_NEGATIVE:
            self.model.Train(dataset_train, dataset_val, self.dp.cfg.C6_BALANCE_SAMPLE)
        else:
            self.model.Train(dataset_train, dataset_val, False)
        CLogging.LogToUser("Finish train model. ")

class CFunctionModelPredict(CFunctionModel):

    def __init__(self, inputDP:CDataPack):

        super().__init__(inputDP)

        self._captainConfig2Paramter()

        self.folder_output = os.path.join(self.dp.cfg.A6_FOLDER_OUTPUT, self.dp.cfg.C3_MODEL_TYPE.name, FOLDER_PREDICT)
        if not os.path.exists(self.folder_output):
            os.makedirs(self.folder_output)
        CFunctionConfig.Config2File(self.dp.cfg, self.folder_output, NAME_CONFIG)

        if self.dp.cfg.Z1_USE_GPU:
            ckpt = torch.load(self.dp.cfg.C2_MODEL_PATH, map_location='cuda')
        else:
            ckpt = torch.load(self.dp.cfg.C2_MODEL_PATH, map_location="cpu")
        self.class_ids = ckpt.get("class_names", ["BG"])
        self.best_loss = ckpt.get('best_loss', float('inf'))

        CLogging.LogToUser(f"Init {self.dp.cfg.C3_MODEL_TYPE.name} ... ")
        # Recreate the model in inference mode
        if not os.path.exists(self.dp.cfg.C2_MODEL_PATH):
            CLogging.LogForLackPath(self.dp.cfg.C2_MODEL_PATH)
        print("Loading weights from ", self.dp.cfg.C2_MODEL_PATH)
        CLogging.LogToUser(f"Finish load weights of {self.dp.cfg.C3_MODEL_TYPE.name}. ")
        CLogging.LogToUser("Use model " + self.dp.cfg.C3_MODEL_TYPE.name)
        if self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNN:
            self.parameter_maskrcnn.OUTPUT_ROOT = self.folder_output
            self.model = ModelMaskRCNN("inference", self.parameter_maskrcnn, self.dp.cfg.C2_MODEL_PATH)
        elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNNBGClassifier:
            self.parameter_maskrcnn.OUTPUT_ROOT = self.folder_output
            self.model = ModelMaskRCNNBGClassifier("inference", self.parameter_maskrcnn, self.dp.cfg.C2_MODEL_PATH)
        elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.Unet:
            self.parameter_unet.OUTPUT_ROOT = self.folder_output
            self.model = ModelUnet("inference", self.parameter_unet, self.dp.cfg.C2_MODEL_PATH)
        elif self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.UNIMask2Former:
            self.parameter_unimask2former.OUTPUT_ROOT = self.folder_output
            self.model = ModelUNIMask2Former("inference", self.parameter_unimask2former, self.dp.cfg.C2_MODEL_PATH)
        else:
            pass
        CLogging.LogToUser(f"Finish load weights of {self.dp.cfg.C3_MODEL_TYPE.name}. ")

    def Predict(self, predict_dataset:CDataModelDataset):

        path_out = os.path.join(self.folder_output, predict_dataset.wsi_name, predict_dataset.pad_name)
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        progress_percent = CLogProgressPercent(0, predict_dataset.num_images - 1, 0.05, "Predict images ... ")
        tmp_num = 0
        img_list = []
        img_name_list = []
        for image_idx in range(predict_dataset.num_images):
            progress_percent.ShowProgressPercent(image_idx)
            if tmp_num == self.parameter_maskrcnn.BATCH_SIZE:
                self.__captainBatchSizeExportPredict(img_list, img_name_list, path_out)
                tmp_num = 0
                img_list = []
                img_name_list = []
            path_image = predict_dataset.image_info[image_idx]["path"]
            image_name = os.path.basename(path_image).split('.')[0]
            image_name_suffix = os.path.basename(path_image).split('.')[-1]
            if image_name_suffix == LIST_DATA_FILE_TYPE[EnumDataFileType.PKL.value]:
                image_array = CFunctionFileIO.ReadPKL(path_image)
            elif ((image_name_suffix == LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIF.value]) or
                  (image_name_suffix == LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIFF.value]) or
                  (image_name_suffix == LIST_VISUAL_FILE_TYPE[EnumVisualFileType.JPG.value]) or
                  (image_name_suffix == LIST_VISUAL_FILE_TYPE[EnumVisualFileType.PNG.value])
                ) :
                image_array = CFunctionFileIO.ReadImage(path_image)
            else:
                CLogging.LogError('Unsupported format for predict ! ' + path_image)
            img_list.append(image_array)
            img_name_list.append(image_name)
            tmp_num += 1
        if tmp_num > 0:
            self.__captainBatchSizeExportPredict(img_list, img_name_list, path_out)

    @staticmethod
    def GetPadIndexInfo(one_predict_dataset: CDataModelDataset, wsi_dataset_info: CDataWSIDatasetBrevity):
        if one_predict_dataset.pad_name == EnumPadType.Org.name:
            row_offset = (wsi_dataset_info.diagonal_height - wsi_dataset_info.org_height) // 2
            col_offset = (wsi_dataset_info.diagonal_width - wsi_dataset_info.org_width) // 2
            x_max_idx = wsi_dataset_info.org_width // wsi_dataset_info.col_size - 1
            y_max_idx = wsi_dataset_info.org_height // wsi_dataset_info.row_size - 1
        elif one_predict_dataset.pad_name == EnumPadType.Horizontal.Horizontal.name:
            row_offset = (wsi_dataset_info.diagonal_height - wsi_dataset_info.org_height) // 2
            col_offset = 0
            x_max_idx = wsi_dataset_info.diagonal_width // wsi_dataset_info.col_size - 1
            y_max_idx = wsi_dataset_info.org_height // wsi_dataset_info.row_size - 1
        elif one_predict_dataset.pad_name == EnumPadType.Vertical.name:
            row_offset = 0
            col_offset = (wsi_dataset_info.diagonal_width - wsi_dataset_info.org_width) // 2
            x_max_idx = wsi_dataset_info.org_width // wsi_dataset_info.col_size - 1
            y_max_idx = wsi_dataset_info.diagonal_height // wsi_dataset_info.row_size - 1
        elif one_predict_dataset.pad_name == EnumPadType.Diagonal.name:
            row_offset = 0
            col_offset = 0
            x_max_idx = wsi_dataset_info.diagonal_width // wsi_dataset_info.col_size - 1
            y_max_idx = wsi_dataset_info.diagonal_height // wsi_dataset_info.row_size - 1
        else:
            row_offset = (wsi_dataset_info.diagonal_height - wsi_dataset_info.org_height) // 2
            col_offset = (wsi_dataset_info.diagonal_width - wsi_dataset_info.org_width) // 2
            x_max_idx = wsi_dataset_info.org_width // wsi_dataset_info.col_size - 1
            y_max_idx = wsi_dataset_info.org_height // wsi_dataset_info.row_size - 1
        # if wsi_dataset_info.image_width % wsi_dataset_info.col_size == 0:
        #     pass
        # else:
        #     x_max_idx += 1
        #
        # if wsi_dataset_info.image_height % wsi_dataset_info.row_size == 0:
        #     pass
        # else:
        #     y_max_idx += 1

        return row_offset, col_offset, x_max_idx, y_max_idx

    def __captainBatchSizeExportPredict(self, img_list, img_name_list, path_out):

        predictions = self.model.Predict(img_list)
        for idx, prediction in enumerate(predictions):
            if idx == len(img_name_list):
                break
            result_img_name = img_name_list[idx].split('.')[0]
            if ((self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNN) or
                (self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.MaskRCNNBGClassifier)
                ):
                final_mask = np.zeros((self.dp.cfg.B1_IMAGE_DIM, self.dp.cfg.B1_IMAGE_DIM, len(self.model.class_ids) - 1), dtype=np.float32)
                masks = prediction['masks'].cpu().numpy()
                labels = prediction["labels"].cpu().numpy()
                scores = prediction["scores"].cpu().numpy()
                for label_idx, label in enumerate(labels):
                    score = scores[label_idx]
                    if score < self.dp.cfg.D1_1_FILTER_MASK_SCORE:
                        continue
                    detect_mask = copy.deepcopy(masks[label_idx][label - 1, :, :])
                    # detect_mask[detect_mask < self.dp.cfg.D1_2_FILTER_MASK_SCORE] = 0
                    final_mask[:, :, label - 1] += detect_mask
                final_mask[final_mask > 1] = 1
            elif (self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.Unet):
                final_mask = prediction.astype(np.float32)
            elif (self.dp.cfg.C3_MODEL_TYPE == EnumTypeModelType.UNIMask2Former):
                final_mask = np.zeros((self.dp.cfg.B1_IMAGE_DIM, self.dp.cfg.B1_IMAGE_DIM, len(self.model.class_ids) - 1), dtype=np.float32)
                for prediction_result in prediction:
                    if prediction_result["score"] < self.dp.cfg.D1_1_FILTER_MASK_SCORE:
                        continue
                    detect_mask = copy.deepcopy(prediction_result['mask'])
                    final_mask[:, :, prediction_result["class_id"] - 1] += detect_mask
            else:
                final_mask = np.zeros((self.dp.cfg.B1_IMAGE_DIM, self.dp.cfg.B1_IMAGE_DIM, len(self.model.class_ids) - 1), dtype=np.float32)
            CFunctionFileIO.WritePKL(path_out, result_img_name + '.pkl', final_mask)

class CFunctionMultiThreadPredictRead(CFunctionMultiThread):

    def __init__(self, data_list:list=[], thread_num:int=1, progress_log_info:str="", show_progress_log_info:bool=False, show_thread_idx:bool=False):
        super(CFunctionMultiThreadPredictRead).__init__(data_list, thread_num, progress_log_info, show_progress_log_info, show_thread_idx)
        self.__final_data_list = [None for i in range(len(self.data_list))]
        self.__final_data_name_list = [None for i in range(len(self.data_list))]

    def UserFunction(self, idx:int):
        image_idx = idx
        path_image = self.data_list[image_idx]["path"]
        image_name = os.path.basename(path_image).split('.')[0]
        image_name_suffix = os.path.basename(path_image).split('.')[-1]
        if image_name_suffix == 'pkl':
            image_array = CFunctionFileIO.ReadPKL(path_image)
        elif image_name_suffix == 'tif':
            image_array = CFunctionFileIO.ReadImage(path_image)
        else:
            CLogging.LogError('Unsupported format for predict ! ' + path_image)
        self.__final_data_list[image_idx] = image_array
        self.__final_data_name_list[image_idx] = image_name

    def GetImageDataList(self):

        return self.__final_data_list

    def GetImageNameList(self):

        return self.__final_data_name_list

class CFunctionMultiThreadPredictWrite(CFunctionMultiThread):

    def __init__(self, data_list:list[list, list], inputDP:CDataPack, maskrcnn_parameter, path_out:str, path_out_heatmap:str, thread_num:int=1):
        self.results = data_list[0]
        self.image_name_list = data_list[1]

        self.dp = inputDP
        self.maskrcnn_parameter = maskrcnn_parameter
        self.path_out_heatmap:str = path_out_heatmap
        self.path_out:str= path_out

        progress_log_info: str = ""
        show_progress_log_info:bool = False
        show_thread_idx:bool = False
        super(CFunctionMultiThreadPredictRead).__init__(data_list[0], thread_num, progress_log_info, show_progress_log_info, show_thread_idx)

    def UserFunction(self, idx:int):
        result_idx = idx

        final_mask = np.zeros((self.maskrcnn_parameter.IMAGE_MIN_DIM, self.maskrcnn_parameter.IMAGE_MIN_DIM), dtype=np.uint8)
        if self.dp.cfg.D2_EXPORT_PREDICT_HEATMAP:
            heatmap_mask = np.zeros((self.maskrcnn_parameter.IMAGE_MIN_DIM, self.maskrcnn_parameter.IMAGE_MIN_DIM),
                                    dtype=np.double)
        result_img_name = self.image_name_list[result_idx].split('.')[0]
        for class_id_idx, class_id in enumerate(self.data_list[idx]["class_ids"]):
            if self.data_list[idx]['scores'][class_id_idx] < self.dp.cfg.D1_1_FILTER_CLASS_SCORE:
                pass
            else:
                detect_mask = self.data_list[idx]["masks"][:, :, class_id_idx].astype(np.uint8)
                final_mask[:, :][detect_mask > 0] = 255
            if self.dp.cfg.D2_EXPORT_PREDICT_HEATMAP:
                detect_heatmap_mask = self.data_list[idx]["masks"][:, :, class_id_idx]
                score_mask = np.zeros(detect_heatmap_mask.shape, dtype=np.double)
                score_mask[detect_heatmap_mask > 0] += self.data_list[idx]['scores'][class_id_idx]
                heatmap_mask[:, :] += score_mask
                del score_mask
                CFunctionMemory.CleanMemory()
        if self.dp.cfg.D2_EXPORT_PREDICT_HEATMAP:
            CFunctionFileIO.WritePKL(self.path_out_heatmap, result_img_name + '.pkl', heatmap_mask)
            del heatmap_mask
            CFunctionMemory.CleanMemory()
        CFunctionFileIO.WritePKL(self.path_out, result_img_name + '.pkl', final_mask)


class CFunctionToolMultiThreadIOPredict:

    __thread_num = 4
    __suffix_thread_predict = LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value] + "_thread"

    @staticmethod
    def WriteMultiChannelMatrix(folder:str, file_name:str, multi_channel_matrix:np.array):
        name, suffix = os.path.splitext(file_name)
        if multi_channel_matrix.shape[0] > CFunctionToolMultiThreadIOPredict.__thread_num:
            CLogging.LogError("Exceed the max thread num, pleas set the thread num and run again! ")
        # multi_channel_matrix 输入必须是通道在最前面：n*height*width
        thread_list = []
        for i in range(multi_channel_matrix.shape[0]):
            new_name = name + "_%d"%i + CFunctionToolMultiThreadIOPredict.__suffix_thread_predict
            thread_list.append(Thread(target=CFunctionFileIO.WritePKL, args=(folder, new_name, multi_channel_matrix[i])))
            thread_list[i].start()

        for i in range(multi_channel_matrix.shape[i]):
            thread_list[i].join()

    @staticmethod
    def ReadMulitChannelMatrixByIdx(path_file:str, idx:int):

        folder = os.path.dirname(path_file)
        file_name = os.path.basename(path_file)
        name, suffix = os.path.splitext(file_name)
        if suffix != CFunctionToolMultiThreadIOPredict.__suffix_thread_predict:
            CLogging.LogError("Please input the file type match this funciton. ")
        else:
            new_path_file = os.path.join(folder, name + "_%d"%idx + suffix)
        return CFunctionFileIO.ReadPKL(new_path_file)

    @staticmethod
    def ReadMultiChannelMatrix(folder:str, key_name:str, channel_num:int):
        list_thread_file = []
        list_folder_file = os.listdir(folder)
        for file in list_thread_file:
            name, suffix = os.path.splitext(file)
            if suffix != CFunctionToolMultiThreadIOPredict.__suffix_thread_predict:
                continue
            if not name.startswith(key_name):
                continue
            list_folder_file.append([os.path.join(folder, name), int(name.split('_'))[-1]])
        list_folder_file.sort(key=lambda x:x[1], reverse=False)
        predict_mask:np.array = None
        for i in range(len(list_folder_file)):
            predict_mask_one_channel = CFunctionFileIO.ReadPKL(list_folder_file[i][0])
            if i == 0:
                predict_mask = predict_mask_one_channel
            else:
                predict_mask = np.dstack(predict_mask, predict_mask_one_channel)
        return predict_mask

    @staticmethod
    def SetThreadNum(thread_num:int):

        CFunctionToolMultiThreadIOPredict.__thread_num = thread_num
