import time

from Core.MSDataPack import CDataPack, EnumTrainType, EnumTypeNegSample
from Core.MSDataModel import CDataModelDataset
from Core.MSFunctionModel import CFunctionModelTrain, CFunctionModelPredict, CFunctionModelDataset
from Core.MSSystem import SUFFIX_HEATMAP, FOLDER_TMP, FOLDER_PROCESSED
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import CDataWSIDatasetBrevity

from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralFunction.MSFunctionImageMerge import CFunctionImageSplitMerge
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.GeneralFunction.MSFunctionNumpyConditionAssign import CFunctionNumpyConditionAssign
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory

from ApuCellGlobal.GeneralSystem.MSSystemEnum import LIST_VISUAL_FILE_TYPE_SUFFIX, EnumVisualFileType, LIST_DATA_FILE_TYPE_SUFFIX, EnumDataFileType

from ApuCellGlobal.Route.MSFunctionDataJsonImageLabel import CFunctionDataJsonImageLabel
from ApuCellGlobal.MaskRCNN.DataSet.MSDataMaskRCNNDataSet import EnumPadType
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import EnumDatasetType

import os
import numpy as np
import math
import shutil
import time

class CTaskTrain:

    def __init__(self, inputDP:CDataPack):

        self.dp = inputDP

    def TrainModel(self):

        functionMaskRCNNDataset = CFunctionModelDataset(self.dp)

        path_dataset: str = self.dp.cfg.A3_PATH_DATASET_TRAIN
        list_path_wsi_dataset = CFunctionFileIO.ReadPKL(path_dataset)

        list_all_train_dataset = []
        for path_wsi_dataset in list_path_wsi_dataset:
            wsi_name = os.path.basename(path_wsi_dataset).split('.')[0]
            list_train_dataset = functionMaskRCNNDataset.GetDataset(path_wsi_dataset, wsi_name, self.dp.cfg.C4_MODEL_ADD_NEGATIVE)
            list_all_train_dataset += list_train_dataset
        all_dataset = functionMaskRCNNDataset.MergeDatasetList(list_all_train_dataset)

        if EnumTrainType(self.dp.cfg.B7_TRAIN_TYPE) == EnumTrainType.Org:
            train_dataset, valid_dataset = functionMaskRCNNDataset.RandomTrainValidDataset(all_dataset, self.dp.cfg.B9_VALID_PERCENT)
            if (self.dp.cfg.C4_MODEL_ADD_NEGATIVE) and (self.dp.cfg.C5_MODEL_NEGATIVE_SAMPLE == EnumTypeNegSample.MiniBatchKMeans):
                CLogging.LogToUser("Use neg sample MiniBatchKMeans ... ")
                final_train_dataset = functionMaskRCNNDataset.GetDatasetWithSampleNegative(train_dataset)
            elif (self.dp.cfg.C4_MODEL_ADD_NEGATIVE) and (
                    self.dp.cfg.C5_MODEL_NEGATIVE_SAMPLE == EnumTypeNegSample.Random):
                CLogging.LogToUser("Use neg sample random ... ")
                final_train_dataset = functionMaskRCNNDataset.GetDatasetWithRandomNegative(train_dataset)
            else:
                CLogging.LogToUser("Use pos ...  ")
                final_train_dataset = train_dataset

            functionMaskRCNNTrain = CFunctionModelTrain(self.dp)
            functionMaskRCNNTrain.Train(final_train_dataset, valid_dataset)

        elif EnumTrainType(self.dp.cfg.B7_TRAIN_TYPE) == EnumTrainType.KFold:
            pass
            # list_train_dataset_for_train = functionMaskRCNNDataset.KFoldDataset(all_dataset, self.dp.cfg.B8_K_FOLD)
            #
            # functionMaskRCNNTrain = CFunctionModelTrain(self.dp)
            # functionMaskRCNNTrain.KFoldTrain(list_train_dataset_for_train)
        else:
            pass


class CTaskTrainDatasetValidDatasetForTrain():

    def __init__(self, inputDP:CDataPack):

        self.dp = inputDP

    def TrainModel(self):

        train_dataset = self.__captainGetDataset(self.dp.cfg.A3_PATH_DATASET_TRAIN, True)
        valid_dataset = self.__captainGetDataset(self.dp.cfg.A4_PATH_DATASET_VALID, False)

        functionMaskRCNNTrain = CFunctionModelTrain(self.dp)
        functionMaskRCNNTrain.Train(train_dataset, valid_dataset)

    def __captainGetDataset(self, path_dataset: str, train_dataset:bool=True):

        list_path_wsi_dataset = CFunctionFileIO.ReadPKL(path_dataset)
        functionMaskRCNNDataset = CFunctionModelDataset(self.dp)

        list_dataset = []
        for path_wsi_dataset in list_path_wsi_dataset:
            wsi_name = os.path.basename(path_wsi_dataset).split('.')[0]
            if train_dataset:
                list_train_dataset = functionMaskRCNNDataset.GetDataset(path_wsi_dataset, wsi_name, self.dp.cfg.C4_MODEL_ADD_NEGATIVE)
            else:
                list_train_dataset = functionMaskRCNNDataset.GetDataset(path_wsi_dataset, wsi_name, False)
            list_dataset += list_train_dataset
        merge_dataset = functionMaskRCNNDataset.MergeDatasetList(list_dataset)
        merge_dataset.prepare()

        if train_dataset:
            if (self.dp.cfg.C4_MODEL_ADD_NEGATIVE) and (self.dp.cfg.C5_MODEL_NEGATIVE_SAMPLE == EnumTypeNegSample.MiniBatchKMeans):
                CLogging.LogToUser("Use neg sample MiniBatchKMeans ... ")
                return functionMaskRCNNDataset.GetDatasetWithSampleNegative(merge_dataset)
            elif (self.dp.cfg.C4_MODEL_ADD_NEGATIVE) and (self.dp.cfg.C5_MODEL_NEGATIVE_SAMPLE == EnumTypeNegSample.Random):
                CLogging.LogToUser("Use neg sample random ... ")
                return functionMaskRCNNDataset.GetDatasetWithRandomNegative(merge_dataset)
            else:
                CLogging.LogToUser("Use pos ...  ")
                return merge_dataset
        else:
            CLogging.LogToUser("Use pos ...  ")
            return merge_dataset

    def __captainGetSampleNegDataset(self, dataset:CDataModelDataset):

        functionMaskRCNNDataset = CFunctionModelDataset(self.dp)
        return functionMaskRCNNDataset.GetDatasetWithSampleNegative(dataset)


class CTaskPredict:
    def __init__(self, inputDP:CDataPack, skip_model_predict:bool=False, skip_merge_wsi_predict:bool=False, skip_multi_channel_wsi_merge:bool=False):

        self.dp = inputDP

        self.functionMaskRCNNDataset = CFunctionModelDataset(self.dp)
        self.functionMaskRCNNPredict = CFunctionModelPredict(self.dp)

        self.show_part_wsi:bool = False

        self.skip_model_predict: bool = skip_model_predict
        self.skip_merge_wsi_predict:bool = skip_merge_wsi_predict
        self.skip_multi_channel_wsi_merge:bool = skip_multi_channel_wsi_merge

    def Predict(self):
        CLogging.CHECK_RUN_STATE = False

        path_dataset: str = self.dp.cfg.A5_PATH_DATASET_PREDICT
        list_path_wsi_dataset = CFunctionFileIO.ReadPKL(path_dataset)

        self.wsi_part_time = []
        for path_wsi_dataset in list_path_wsi_dataset:
            time1 = time.time()
            if self.dp.cfg.A8_PREDICT_SKIP_EXIST:
                wsi_name = os.path.basename(path_wsi_dataset).split('.')[0]
                folder_result = os.path.join(self.functionMaskRCNNPredict.folder_output, wsi_name)
                if os.path.exists(folder_result):
                    CLogging.LogToUser("WSI name : " + wsi_name + " has predict and will skip. If want to overlap, please set A8_PREDICT_SKIP_EXIST! ")
                    continue
            wsi_name = os.path.basename(path_wsi_dataset).split('.')[0]
            # if wsi_name != "patient_075_node_4":
            #     continue
            self.wsi_part_time.append([])
            self.MergePredictResult(path_wsi_dataset)
            CFunctionMemory.CleanMemory()

            time2 = time.time()
            CLogging.LogToUser(f"{wsi_name} Use time : " + "%.4fs" % (time2 - time1))
            self.wsi_part_time[-1].append((wsi_name, "%.4fs" % (time2 - time1)))

        # with open(os.path.join(os.getcwd(), self.dp.cfg.C3_MODEL_TYPE.name + CLogging.GetLocTimeStrForName() + ".txt"), "w") as f:
        #     for wsi_time_info in self.wsi_part_time:
        #         for item in wsi_time_info:
        #             f.write(item[0])
        #             f.write("\t")
        #             f.write(item[1])
        #             f.write('\n')

    def MergePredictResult(self, path_wsi_dataset):

        wsi_name = os.path.basename(path_wsi_dataset).split('.')[0]
        CLogging.LogToUser()
        CLogging.LogToUser("Predict wsi : " + wsi_name)

        wsi_dataset_info: CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(path_wsi_dataset)
        list_predict_dataset = self.functionMaskRCNNDataset.GetDataset(path_wsi_dataset, wsi_name)
        if list_predict_dataset == None:
            CLogging.LogWarning("The dataset is not support ! ")
            return
        folder_work = os.path.join(self.functionMaskRCNNPredict.folder_output, wsi_name)
        if not os.path.exists(folder_work):
            os.mkdir(folder_work)

        CLogging.LogToUser("Size of merge mask " + str(wsi_dataset_info.diagonal_height) + "*" + str(wsi_dataset_info.diagonal_width))
        CFunctionMemory.GetCurMemoryInfo()
        merge_mask = np.zeros((wsi_dataset_info.diagonal_height, wsi_dataset_info.diagonal_width, len(self.functionMaskRCNNPredict.model.class_ids) - 1), dtype=np.uint16)
        # merge_mask = np.ones((wsi_dataset_info.diagonal_height, wsi_dataset_info.diagonal_width, len(self.functionMaskRCNNPredict.model.class_ids) - 1), dtype=np.uint16)
        CFunctionMemory.GetCurMemoryInfo()

        final_mask_h_start_idx = wsi_dataset_info.row_size // 2
        final_mask_h_end_idx = wsi_dataset_info.row_size // 2 + wsi_dataset_info.image_height

        final_mask_w_start_idx = wsi_dataset_info.col_size // 2
        final_mask_w_end_idx = wsi_dataset_info.col_size // 2 + wsi_dataset_info.image_width

        folder_processed = os.path.join(folder_work, FOLDER_PROCESSED)
        if not os.path.exists(folder_processed):
            os.mkdir(folder_processed)

        folder_tmp = os.path.join(folder_work, FOLDER_TMP)
        if not os.path.exists(folder_tmp):
            os.mkdir(folder_tmp)

        CFunctionMemory.GetCurMemoryInfo()

        for one_predict_dataset in list_predict_dataset:

            time_predict_start = time.time()

            CLogging.LogToUser()
            CLogging.LogToUser("Start predict wsi patch: " + one_predict_dataset.dataset_name + '...')
            self.functionMaskRCNNPredict.Predict(one_predict_dataset)
            CLogging.LogToUser("Finish predict wsi patch: " + one_predict_dataset.dataset_name)
            CLogging.LogToUser()

            folder_predict_result = os.path.join(self.functionMaskRCNNPredict.folder_output, wsi_name, one_predict_dataset.pad_name)
            row_offset, col_offset, x_max_idx, y_max_idx = self.functionMaskRCNNPredict.GetPadIndexInfo(one_predict_dataset, wsi_dataset_info)
            if row_offset == 0:
                y_start = row_offset
                y_end = merge_mask.shape[0]
            else:
                y_start = row_offset
                y_end = merge_mask.shape[0] - row_offset
            if col_offset == 0:
                x_start = col_offset
                x_end = merge_mask.shape[1]
            else:
                x_start = col_offset
                x_end = merge_mask.shape[1] - col_offset
            CLogging.LogToUser("Get predict mask ... " + wsi_name + " pad name: " + one_predict_dataset.pad_name)
            CFunctionMemory.GetCurMemoryInfo()

            if not self.dp.cfg.D5_MERGE_OVERLAP:
                CLogging.LogToUser("Use non-overlap ... " + wsi_name + " pad name: " + one_predict_dataset.pad_name)
                predict_wsi_mask = CFunctionImageSplitMerge.MergeImage(folder_predict_result, '_', 0, 0, x_max_idx,
                                                                       y_max_idx,
                                                                       wsi_dataset_info.col_size // 4,
                                                                       wsi_dataset_info.col_size // 4 * 3,
                                                                       wsi_dataset_info.row_size // 4,
                                                                       wsi_dataset_info.row_size // 4 * 3,
                                                                       use_datatype=np.uint16,
                                                                       )
            else:
                CLogging.LogToUser("Use overlap ... " + wsi_name + " pad name: " + one_predict_dataset.pad_name)
                predict_wsi_mask = CFunctionImageSplitMerge.MergeImage(folder_predict_result, '_', 0, 0, x_max_idx, y_max_idx, use_datatype=np.uint16)

            if one_predict_dataset.pad_name != EnumPadType.Org.name:
                path_out = os.path.join(self.functionMaskRCNNPredict.folder_output, wsi_name, one_predict_dataset.pad_name)
                if os.path.exists(path_out):
                    shutil.rmtree(path_out)

            time_predict_end = time.time()
            self.wsi_part_time[-1].append([f"predict_{wsi_name}_{one_predict_dataset.pad_name}", str(time_predict_end - time_predict_start)])

            CLogging.LogToUser("Finish get predict mask. " + wsi_name + " pad name: " + one_predict_dataset.pad_name)
            CFunctionMemory.GetCurMemoryInfo()

            time_predict_merge_start = time.time()

            for idx in range(1, len(self.functionMaskRCNNPredict.model.class_ids)):
                class_id_name = self.functionMaskRCNNPredict.model.class_ids[idx]
                predict_wsi_mask_class = predict_wsi_mask[idx - 1, :, :]

                # 将一个WSI的不同pad得到的mask进行融合
                CLogging.LogToUser("Merge " + wsi_name + " predict class: " + class_id_name + " from pad name : " + one_predict_dataset.pad_name + " ... ")
                CFunctionNumpyConditionAssign.set_gt(predict_wsi_mask_class, 1e4, 1e4)
                CFunctionNumpyConditionAssign.set_lt(predict_wsi_mask_class, int(self.dp.cfg.D1_1_FILTER_MASK_SCORE * 1e4), 0)

                if self.dp.cfg.D6_EXPORT_PAD_MASK:
                    export_wsi_pad = True
                else:
                    if (one_predict_dataset.pad_name == EnumPadType.Org.name):
                        export_wsi_pad = True
                    else:
                        export_wsi_pad = False
                if export_wsi_pad:
                    CLogging.LogToUser(" Export " + wsi_name + " predict class" + class_id_name + " info from pad name : " + one_predict_dataset.pad_name + " ... ")
                    predict_wsi_mask_class_pad = np.zeros((predict_wsi_mask.shape[1], predict_wsi_mask.shape[2]), dtype=np.uint8)
                    CFunctionMemory.GetCurMemoryInfo()
                    CFunctionNumpyConditionAssign.where_gt(predict_wsi_mask_class_pad, predict_wsi_mask_class, 0, 255)
                    CFunctionFileIO.WriteTIFByGDAL(folder_tmp, wsi_name + "_" + one_predict_dataset.pad_name + "_" + class_id_name + LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value], predict_wsi_mask_class_pad)
                    del predict_wsi_mask_class_pad
                    CLogging.LogToUser(" Finish export " + wsi_name + " predict class" + class_id_name + " info from pad name : " + one_predict_dataset.pad_name + ". ")
                    CFunctionMemory.CleanMemory()
                    CFunctionMemory.GetCurMemoryInfo()

                merge_mask[y_start: y_end, x_start: x_end, idx - 1] += predict_wsi_mask_class
                del predict_wsi_mask_class
                CFunctionMemory.CleanMemory()
                CFunctionMemory.GetCurMemoryInfo()
                CLogging.LogToUser("Finish merge " + wsi_name + " predict class: " + class_id_name + " from pad name : " + one_predict_dataset.pad_name + ". ")
            time_predict_merge_end = time.time()
            self.wsi_part_time[-1].append([f"predict_merge_{wsi_name}_{one_predict_dataset.pad_name}", str(time_predict_merge_end - time_predict_merge_start)])

            del predict_wsi_mask
            CFunctionMemory.CleanMemory()
            CFunctionMemory.GetCurMemoryInfo()

        time_mca_start = time.time()

        CLogging.LogToUser()
        CLogging.LogToUser("Get rough mask " + wsi_name + " ... ")
        CFunctionMemory.GetCurMemoryInfo()
        for idx in range(1, len(self.functionMaskRCNNPredict.model.class_ids)):
            class_id_name = self.functionMaskRCNNPredict.model.class_ids[idx]
            CLogging.LogToUser("Start get rough mask from " + wsi_name + ", predict class: " + class_id_name)
            CFunctionMemory.GetCurMemoryInfo()

            merge_mask_class = merge_mask[:, :, idx - 1]
            CFunctionMemory.GetCurMemoryInfo()

            CFunctionNumpyConditionAssign.set_lt(merge_mask_class, int(self.dp.cfg.D4_MERGE_SCORE * 1e4), 0)
            CFunctionNumpyConditionAssign.set_gt(merge_mask_class, int(self.dp.cfg.D4_MERGE_SCORE * 1e4), 1, True)

            merge_mask_class_uint8 = merge_mask_class[final_mask_h_start_idx: final_mask_h_end_idx, final_mask_w_start_idx: final_mask_w_end_idx].astype(np.uint8)
            CFunctionMemory.GetCurMemoryInfo()
            del merge_mask_class
            CFunctionMemory.CleanMemory()
            CFunctionMemory.GetCurMemoryInfo()

            CFunctionToolImageC.SetMatrixValue(merge_mask_class_uint8, 255)

            CLogging.LogToUser("Export rough mask from " + wsi_name + ", predict class: " + class_id_name + " ... ")
            CFunctionFileIO.WriteNpy(folder_processed, wsi_name + "_" + class_id_name + "_Rough" + ".npy", merge_mask_class_uint8)
            del merge_mask_class_uint8
            CFunctionMemory.CleanMemory()
            CFunctionMemory.GetCurMemoryInfo()
            CLogging.LogToUser("Finish get rough mask from " + wsi_name + ", predict class: " + class_id_name)
        CLogging.LogToUser("Finish get rough mask " + wsi_name + ". ")
        del merge_mask
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()

        CLogging.LogToUser()
        CLogging.LogToUser("Start refine " + wsi_name + " ... ")
        CFunctionMemory.GetCurMemoryInfo()
        for idx in range(1, len(self.functionMaskRCNNPredict.model.class_ids)):
            class_id_name = self.functionMaskRCNNPredict.model.class_ids[idx]
            CLogging.LogToUser("Refine " + wsi_name + ", predict class: " + class_id_name)
            CFunctionMemory.GetCurMemoryInfo()

            merge_mask_class_uint8 = CFunctionFileIO.ReadNpy(os.path.join(folder_processed, wsi_name + "_" + class_id_name + "_Rough" + ".npy"))
            CFunctionMemory.GetCurMemoryInfo()

            # 填充之后的mask
            CLogging.LogToUser('Start fill hole ... ' + wsi_name + ", predict class: " + class_id_name)
            CFunctionImageOperator.FillHoleMask(merge_mask_class_uint8)
            CLogging.LogToUser('Finish fill hole. ' + wsi_name + ", predict class: " + class_id_name)
            # 填充之后的mask

            # 去除最小面积
            CLogging.LogToUser("Filter area ... " + wsi_name + ", predict class: " + class_id_name)
            CFunctionToolImageC.BatchRemoveShapeByArea(merge_mask_class_uint8, self.dp.cfg.D3_MIN_AREA)
            CLogging.LogToUser("Finish filter area: " + wsi_name + ", predict class: " + class_id_name)
            # 去除最小面积

            if (wsi_dataset_info.dataset_type == EnumDatasetType.Assist):
                CLogging.LogToUser("Annotating assist region ... ")
                assist_region_mask_tmp = wsi_dataset_info.GetMask()
                assist_region_mask = np.zeros((assist_region_mask_tmp.shape[0], assist_region_mask_tmp.shape[1]), dtype=np.uint8)
                CFunctionMemory.CleanMemory()
                CFunctionMemory.GetCurMemoryInfo()
                assist_region_mask[:, :] += assist_region_mask_tmp[:, :, 0]
                del assist_region_mask_tmp
                CFunctionMemory.CleanMemory()
                CFunctionMemory.GetCurMemoryInfo()

                CFunctionToolImageC.SetMatrixValue(assist_region_mask, 1)
                CFunctionToolImageC.SetMatrixValue(merge_mask_class_uint8, 1)
                CFunctionToolImageC.AddMatrixValue(merge_mask_class_uint8, assist_region_mask)
                CFunctionToolImageC.FilterMatrixValueByLarge(merge_mask_class_uint8, 2, True)
                del assist_region_mask
                CFunctionMemory.CleanMemory()
                CFunctionMemory.GetCurMemoryInfo()
                CLogging.LogToUser("Finish annotating assist region. ")

            CFunctionFileIO.WriteTIFByGDAL(folder_work, wsi_name + "_" + class_id_name + LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value], merge_mask_class_uint8)
            path_annotation = os.path.join(folder_work, wsi_name + "_" + class_id_name + LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.JSON.value])
            image_label_file = CFunctionDataJsonImageLabel.Mask2JsonFile(merge_mask_class_uint8, wsi_dataset_info.image_path, True, path_annotation=path_annotation, path_mask=wsi_dataset_info.mask_path)
            CFunctionDataJsonImageLabel.WriteJsonInfo(os.path.dirname(path_annotation), os.path.basename(path_annotation), image_label_file)
            CLogging.LogToUser(f"Finish export boundary, class id: {class_id_name}. " + wsi_name + LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.JSON.value])
            del image_label_file
            del merge_mask_class_uint8
            CFunctionMemory.CleanMemory()
            CFunctionMemory.GetCurMemoryInfo()
        CLogging.LogToUser("Finish refine " + wsi_name + ". ")

        time_mca_end = time.time()
        self.wsi_part_time[-1].append([f"mca_{wsi_name}", str(time_mca_end - time_mca_start)])


