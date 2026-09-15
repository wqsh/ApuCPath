import numpy as np

import copy

from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import CDataWSIDatasetBrevity

from Evaluate.MSToolGetMask import GetPredictMultiMask, GetAnnotationMask, GetDataset


import os


def RunWSIInstance(folder_wsi_instance_mask, folder_predict_json, path_dataset, folder_output, label_name="label"):


    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    folder_output_pkl = os.path.join(folder_output, "pkl")

    if not os.path.exists(folder_output_pkl):
        os.makedirs(folder_output_pkl)

    list_wsi_instance_mask = os.listdir(folder_wsi_instance_mask)

    CLogging.LogToUser("Start ================ ")

    list_path_wsi_dataset = CFunctionFileIO.ReadPKL(path_dataset)
    list_finish_get_wsi_instance = os.listdir(folder_output)

    for name_wsi_instance_mask in list_wsi_instance_mask:
        # wsi_name = os.path.basename(name_wsi_instance_mask).split('.')[0].split('_')[0] + '_' + os.path.basename(name_wsi_instance_mask).split('.')[0].split('_')[1]
        wsi_name = os.path.basename(name_wsi_instance_mask).split('.')[0]
        CLogging.LogToUser("wsi : " + wsi_name)
        # if wsi_name != "test_105":
        #     continue
        # finish_get_dice = False
        # for finish_wsi in list_finish_get_wsi_instance:
        #     if finish_wsi.startswith(wsi_name):
        #         finish_get_dice = True
        # if finish_get_dice:
        #     CLogging.LogToUser(wsi_name + " is finish. ")
        #     continue

        predict_mask = None
        for path_wsi_dataset in list_path_wsi_dataset:
            if wsi_name != os.path.basename(path_wsi_dataset).split('.')[0]:
                continue
            wsi_dataset_info: CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(path_wsi_dataset)
            path_image = wsi_dataset_info.image_path
            CLogging.LogToUser("Get predict wsi mask: " + wsi_name)
            path_predict_json = None
            for item in os.listdir(os.path.join(folder_predict_json, wsi_name)):
                if item.endswith("json"):
                    path_predict_json = os.path.join(folder_predict_json, wsi_name, item)
                    break
            if path_predict_json is None:
                CLogging.LogToUser("Not find the path_predict_json! ")
            predict_mask = GetPredictMultiMask(path_image, path_predict_json, label_name)
            break
        if predict_mask is None:
            CLogging.LogToUser("Not find the predict result! ")
            continue

        CLogging.LogToUser("Get annotation mask ... ")
        annotation_wsi_instance_mask_RGB = CFunctionFileIO.ReadImage(os.path.join(folder_wsi_instance_mask, name_wsi_instance_mask))
        annotation_wsi_instance_mask = CFunctionImageOperator.RGB2Gray(annotation_wsi_instance_mask_RGB)
        del annotation_wsi_instance_mask_RGB
        CLogging.LogToUser("Finish get annotation mask ... ")
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()

        # annotation_h, annotation_w = annotation_wsi_instance_mask.shape
        # predict_h, predict_w = predict_mask.shape
        # check_img = np.zeros((predict_h, predict_w, 3), dtype=np.uint8)
        # check_img[:annotation_h, :annotation_w, 0] += annotation_wsi_instance_mask[:annotation_h, :annotation_w]
        # check_img[:predict_h, :predict_w, 1] += predict_mask[:predict_h, :predict_w]
        # CFunctionFileIO.WriteTIFByGDAL(folder_output, "check.tiff", check_img)

        CLogging.LogToUser("Get insection info ... ")
        insection_mask = copy.deepcopy(predict_mask)
        CFunctionToolImageC.SetMatrixValue(insection_mask, 1)
        CFunctionToolImageC.SetMatrixValue(annotation_wsi_instance_mask, 2)
        CFunctionToolImageC.AddMatrixValue(insection_mask, annotation_wsi_instance_mask)
        CFunctionToolImageC.FilterMatrixValueByLarge(insection_mask, 3)
        CLogging.LogToUser("Get insection mask. ")
        insection_mask_label, insection_mask_label_num = CFunctionImageOperator.GetImageMask2Label(insection_mask)
        CLogging.LogToUser("Insection mask label num %d. "%insection_mask_label_num)
        np_insection_mask_label_seed_point = CFunctionToolImageC.GetLabelSeedPoint(insection_mask_label, [i + 1 for i in range(insection_mask_label_num)])
        CLogging.LogToUser("Get insection mask label seed point. ")
        del insection_mask_label
        CFunctionMemory.CleanMemory()

        insection_mask_label_seed_point_list = np.array([[seed_point.seed_x, seed_point.seed_y] for seed_point in np_insection_mask_label_seed_point], dtype=np.uint32)
        insection_mask_label_shape_info_list = CFunctionToolImageC.GetShapeInfoList(insection_mask, insection_mask_label_seed_point_list, False)
        insection_mask_label_shape_info_list.insert(0, None)
        CLogging.LogToUser("Finish get insection info. ")

        CLogging.LogToUser("Get annotation info. ")
        CFunctionToolImageC.SetMatrixValue(annotation_wsi_instance_mask, 255)
        annotation_wsi_instance_mask_label, annotation_wsi_instance_mask_label_num = CFunctionImageOperator.GetImageMask2Label(annotation_wsi_instance_mask)
        CLogging.LogToUser("Annotation mask label num %d. " % annotation_wsi_instance_mask_label_num)
        np_annotation_wsi_instance_mask_label_seed_point = CFunctionToolImageC.GetLabelSeedPoint(annotation_wsi_instance_mask_label, [i + 1 for i in range(annotation_wsi_instance_mask_label_num)])
        CLogging.LogToUser("Get annotation mask label seed point. ")
        annotation_wsi_instance_mask_label_seed_point_list = np.array([[seed_point.seed_x, seed_point.seed_y] for seed_point in np_annotation_wsi_instance_mask_label_seed_point], dtype=np.uint32)
        annotation_wsi_instance_mask_label_shape_info_list = CFunctionToolImageC.GetShapeInfoList(annotation_wsi_instance_mask, annotation_wsi_instance_mask_label_seed_point_list, False)
        annotation_wsi_instance_mask_label_shape_info_list.insert(0, None)
        CLogging.LogToUser("Finish get annotation info. ")

        CLogging.LogToUser("Get insection 2 annotation label index ... ")
        insection_mask_label2annotation_wsi_instance_mask_label = [None for i in range(insection_mask_label_num + 1)]
        for insection_idx in range(1, insection_mask_label_num + 1):
            annotation_wsi_instance_mask_label_idx = annotation_wsi_instance_mask_label[insection_mask_label_shape_info_list[insection_idx].seed_y, insection_mask_label_shape_info_list[insection_idx].seed_x]
            insection_mask_label2annotation_wsi_instance_mask_label[insection_idx] = annotation_wsi_instance_mask_label_idx
        del annotation_wsi_instance_mask_label
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()
        CLogging.LogToUser("Finish get insection 2 annotation label index. ")

        CLogging.LogToUser("Get predict info ... ")
        predict_mask_label, predict_mask_label_num = CFunctionImageOperator.GetImageMask2Label(predict_mask)
        CLogging.LogToUser("Predict mask label num %d. " % predict_mask_label_num)
        np_predict_mask_label_seed_point = CFunctionToolImageC.GetLabelSeedPoint(predict_mask_label, [i + 1 for i in range(predict_mask_label_num)])
        CLogging.LogToUser("Get predict mask label seed point. ")

        predict_mask_label_seed_point_list = np.array([[seed_point.seed_x, seed_point.seed_y] for seed_point in np_predict_mask_label_seed_point], dtype=np.uint32)
        predict_mask_label_shape_info_list = CFunctionToolImageC.GetShapeInfoList(predict_mask, predict_mask_label_seed_point_list, False)
        predict_mask_label_shape_info_list.insert(0, None)
        CLogging.LogToUser("Finish get predict info. ")

        CLogging.LogToUser("Get insection 2 predict label index ... ")
        insection_mask_label2predict_mask_label = [None for i in range(insection_mask_label_num + 1)]
        for insection_mask_label_idx in range(1, insection_mask_label_num + 1):
            insection_mask_label_idx_shape_info = insection_mask_label_shape_info_list[insection_mask_label_idx]
            predict_mask_label_idx = predict_mask_label[insection_mask_label_idx_shape_info.seed_y, insection_mask_label_idx_shape_info.seed_x]
            insection_mask_label2predict_mask_label[insection_mask_label_idx] = predict_mask_label_idx
        del predict_mask_label
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()
        CLogging.LogToUser("Finish get insection 2 predict label index. ")

        annotation_wsi_instance_mask_label2insection_mask_label = [[] for i in range(annotation_wsi_instance_mask_label_num + 1)]
        for insection_mask_label_idx in range(1, insection_mask_label_num + 1):
            annotation_wsi_instance_mask_label2insection_mask_label[insection_mask_label2annotation_wsi_instance_mask_label[insection_mask_label_idx]].append(insection_mask_label_idx)

        predict_mask_label2insection_mask_label = [[] for i in range(predict_mask_label_num + 1)]
        for insection_mask_label_idx in range(1, insection_mask_label_num + 1):
            predict_mask_label2insection_mask_label[insection_mask_label2predict_mask_label[insection_mask_label_idx]].append(insection_mask_label_idx)

        annotation_wsi_instance_mask_label2predict_mask_label = [[] for i in range(annotation_wsi_instance_mask_label_num + 1)]
        for annotation_wsi_instance_mask_label_idx in range(1, annotation_wsi_instance_mask_label_num + 1):
            annotation_wsi_instance_mask_label2insection_mask_label_list = annotation_wsi_instance_mask_label2insection_mask_label[annotation_wsi_instance_mask_label_idx]
            for insection_mask_label_idx in annotation_wsi_instance_mask_label2insection_mask_label_list:
                if insection_mask_label2predict_mask_label[insection_mask_label_idx] in annotation_wsi_instance_mask_label2predict_mask_label[annotation_wsi_instance_mask_label_idx]:
                    continue
                else:
                    annotation_wsi_instance_mask_label2predict_mask_label[annotation_wsi_instance_mask_label_idx].append(insection_mask_label2predict_mask_label[insection_mask_label_idx])
        annotation_wsi_instance_mask_label_dice_info_list = [[] for i in range(annotation_wsi_instance_mask_label_num + 1)]
        for annotation_wsi_instance_mask_label_idx in range(1, annotation_wsi_instance_mask_label_num + 1):
            predict_mask_label_idx_list = annotation_wsi_instance_mask_label2predict_mask_label[
                annotation_wsi_instance_mask_label_idx]
            predict_mask_label_idx2dice_list = []
            if len(predict_mask_label_idx_list) == 0:
                TP = 0
                FP = 0
                FN = 0
                annotation_area = annotation_wsi_instance_mask_label_shape_info_list[
                    annotation_wsi_instance_mask_label_idx].area
                predict_area = 0
                dice_info_dice = {'TP': TP, 'FP': FP, "FN": FN, "InstanceDice": 0, 'PredictLabelIdx': -1,
                                  "annotation_area": annotation_area, "predict_area": predict_area}
                annotation_wsi_instance_mask_label_dice_info_list[annotation_wsi_instance_mask_label_idx].append(
                    dice_info_dice)
            else:
                for predict_mask_label_idx in predict_mask_label_idx_list:
                    TP = 0
                    insection_mask_label_idx_list = predict_mask_label2insection_mask_label[predict_mask_label_idx]
                    for insection_mask_label_idx in insection_mask_label_idx_list:
                        if insection_mask_label2annotation_wsi_instance_mask_label[
                            insection_mask_label_idx] != annotation_wsi_instance_mask_label_idx:
                            continue
                        TP += insection_mask_label_shape_info_list[insection_mask_label_idx].area
                    FN = annotation_wsi_instance_mask_label_shape_info_list[
                             annotation_wsi_instance_mask_label_idx].area - TP
                    FP = predict_mask_label_shape_info_list[predict_mask_label_idx].area - TP
                    annotation_area = annotation_wsi_instance_mask_label_shape_info_list[
                        annotation_wsi_instance_mask_label_idx].area
                    predict_area = predict_mask_label_shape_info_list[predict_mask_label_idx].area
                    dice_info_dice = {'TP': TP, 'FP': FP, "FN": FN, "InstanceDice": 2 * TP / (2 * TP + FN + FP),
                                      'PredictLabelIdx': predict_mask_label_idx, "annotation_area": annotation_area,
                                      "predict_area": predict_area}
                    annotation_wsi_instance_mask_label_dice_info_list[annotation_wsi_instance_mask_label_idx].append(dice_info_dice)
        CFunctionFileIO.WritePKL(folder_output_pkl, wsi_name + '.pkl', annotation_wsi_instance_mask_label_dice_info_list)
        del predict_mask
        del insection_mask
        del annotation_wsi_instance_mask
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()
        CLogging.LogToUser("Finish get " + wsi_name + " instance dice! ")

def ExportWSIInstanceResult(folder_wsi_instance_dice, folder_predict_json, folder_wsi_instance_mask, folder_output, folder_img, label_name="label"):

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    show_result = False

    folder_output_mask = os.path.join(folder_output, 'mask')

    if not os.path.exists(folder_output_mask):
        os.makedirs(folder_output_mask)

    path_output = os.path.join(folder_output, 'multi_wsi_instance_dice.txt')

    list_wsi_instance_dice = os.listdir(folder_wsi_instance_dice)
    list_wsi_instance_result = []
    for wsi_instance_dice_pkl in list_wsi_instance_dice:
        wsi_name = wsi_instance_dice_pkl.split('.')[0]
        # if not wsi_name.startswith('test_121'):
        #     continue
        path_wsi_instance_dice_pkl = os.path.join(folder_wsi_instance_dice, wsi_instance_dice_pkl)
        wsi_instance_dice = CFunctionFileIO.ReadPKL(path_wsi_instance_dice_pkl)
        one_wsi_instance_dice = [None]
        best_annotation2predict_idx_list = [None]
        for wsi_instance_idx in range(1, len(wsi_instance_dice)):
            max_dice = 0
            max_dice_idx = None
            for predict_instance_dice_idx, predict_instance_dice_dict_list in enumerate(
                    wsi_instance_dice[wsi_instance_idx]):
                if max_dice_idx is None:
                    max_dice = predict_instance_dice_dict_list["InstanceDice"]
                    max_dice_idx = predict_instance_dice_idx
                else:
                    if max_dice < predict_instance_dice_dict_list["InstanceDice"]:
                        max_dice = predict_instance_dice_dict_list["InstanceDice"]
                        max_dice_idx = predict_instance_dice_idx
            if max_dice_idx is None:
                one_wsi_instance_dice.append(None)
                best_annotation2predict_idx_list.append(None)
            else:
                one_wsi_instance_dice.append(wsi_instance_dice[wsi_instance_idx][max_dice_idx])
                best_annotation2predict_idx_list.append(
                    wsi_instance_dice[wsi_instance_idx][max_dice_idx]["PredictLabelIdx"])
        list_wsi_instance_result.append([wsi_name, one_wsi_instance_dice])

        if show_result:

            CLogging.LogToUser("Get predict wsi mask: " + wsi_name)
            path_image = os.path.join(folder_img, wsi_name + ".tif")
            path_predict_json = os.path.join(folder_predict_json, wsi_name, wsi_name + ".json")
            predict_mask = GetPredictMultiMask(path_image, path_predict_json, label_name)
            CLogging.LogToUser("Get predict info ... ")
            predict_mask_label, predict_mask_label_num = CFunctionImageOperator.GetImageMask2Label(predict_mask)
            CLogging.LogToUser("Predict mask label num %d. " % predict_mask_label_num)
            np_predict_mask_label_seed_point = CFunctionToolImageC.GetLabelSeedPoint(predict_mask_label, [i + 1 for i in
                                                                                                          range(
                                                                                                              predict_mask_label_num)])
            CLogging.LogToUser("Get predict mask label seed point. ")
            del predict_mask_label
            CFunctionMemory.CleanMemory()
            for predict_mask_label_idx in range(1, predict_mask_label_num + 1):
                if predict_mask_label_idx in best_annotation2predict_idx_list:
                    continue
                else:
                    CFunctionToolImageC.RemoveShape(predict_mask,
                                                    np_predict_mask_label_seed_point[predict_mask_label_idx - 1].seed_x,
                                                    np_predict_mask_label_seed_point[predict_mask_label_idx - 1].seed_y)

            new_mask = np.zeros((3, predict_mask.shape[0], predict_mask.shape[1]), dtype=np.uint8)
            CFunctionToolImageC.ReplaceMatrixValue(new_mask[0, :, :], predict_mask)

            del predict_mask
            CFunctionMemory.CleanMemory()

            list_annotation_mask = os.listdir(folder_wsi_instance_mask)
            for file_name in list_annotation_mask:
                if file_name.startswith(wsi_name):
                    CLogging.LogToUser("Get annotation mask ... ")
                    annotation_wsi_instance_mask_RGB = CFunctionFileIO.ReadImage(
                        os.path.join(folder_wsi_instance_mask, file_name))
                    annotation_wsi_instance_mask = CFunctionImageOperator.RGB2Gray(annotation_wsi_instance_mask_RGB)
                    del annotation_wsi_instance_mask_RGB
                    CFunctionMemory.CleanMemory()
                    CLogging.LogToUser("Finish get annotation mask ... ")

                    CFunctionToolImageC.ReplaceMatrixValue(new_mask[1, :, :], annotation_wsi_instance_mask)
                    CFunctionFileIO.WriteTIFByGDAL(folder_output_mask, wsi_name + "_annotation.tiff",
                                                   new_mask.transpose(1, 2, 0))
                    break
            del annotation_wsi_instance_mask
            del new_mask
            CFunctionMemory.CleanMemory()

    with open(path_output, 'w') as f:
        f.write('wsi name\tIdx\tTP\tFP\tFN\tDice\tpredict area\tannotation area\tpredict idx\n')
        for wsi_instance_dice in list_wsi_instance_result:
            f.write(wsi_instance_dice[0])
            f.write('\t')
            if len(wsi_instance_dice[1]) == 1:
                f.write('\n')
            for i in range(1, len(wsi_instance_dice[1])):
                if i > 1:
                    f.write('')
                    f.write('\t')
                f.write(str(i))
                f.write('\t')
                if wsi_instance_dice[1][i] is None:
                    for j in range(6):
                        f.write('0')
                        f.write('\t')
                    f.write('\n')
                else:
                    f.write(str(wsi_instance_dice[1][i]["TP"]))
                    f.write('\t')

                    f.write(str(wsi_instance_dice[1][i]["FP"]))
                    f.write('\t')

                    f.write(str(wsi_instance_dice[1][i]["FN"]))
                    f.write('\t')

                    f.write("%.6f"%wsi_instance_dice[1][i]["InstanceDice"])
                    f.write('\t')

                    f.write(str(wsi_instance_dice[1][i]["annotation_area"]))
                    f.write('\t')

                    f.write(str(wsi_instance_dice[1][i]["predict_area"]))
                    f.write('\t')

                    f.write(str(wsi_instance_dice[1][i]["PredictLabelIdx"]))
                    f.write('\t')

                    f.write('\n')

def main():

    label_name = r"Pos"

    list_evaluate = [
        [
            r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/mask",
            r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/UNIMask2FormerStander/UNIMask2Former/Train/20260623222336/PredictTestDatasetWSI/UNIMask2Former/Predict",
            r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Train",
            r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/image"
        ],
        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/Unet/Unet/Train/20260624055805/PredictTestDatasetWSI/Unet/Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Train",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/image"
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/MaskRCNN/MaskRCNN/Train/20260623222335/PredictTestDataset/MaskRCNN/Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Train",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/image"
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/Unet/Unet/Train/20260610080756/PredictTestDatasetWSI/Unet/Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Train",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/image"
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/MaskRCNN/MaskRCNN/Train/20260610003857/PredictTestDatasetWSI/MaskRCNN/Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Train",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/image"
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/UNIMask2Former/UNIMask2Former/Train/20260611085630/PredictTestDatasetWSI/UNIMask2Former/Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Train",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/image"
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DNHIL/TrainDataset/Start/ModelTrain/MaskRCNN/MaskRCNN/Train/20260619235652/PredictTestDatasetWSI/MaskRCNN/Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DN/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Test/image"
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DNHIL/TrainDataset/Second/ModelTrain/UNIMask2Former/UNIMask2Former/Train/20260624145739/PredictTestDatasetWSI/UNIMask2Former/Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DN/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Test/image"
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Train/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DN/TrainDataset/ModelTrain/UNIMask2Former/UNIMask2Former/Train/20260623094818/PredictTrainDatasetWSI/UNIMask2Former/Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DNHIL/TrainDataset/TrainDatasetForPredict/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Train/image"
        # ],

    ]

    for item in list_evaluate:
        folder_wsi_instance_mask = item[0]
        folder_predict_json = item[1]
        path_dataset = item[2]
        folder_img = item[3]

        folder_output = os.path.join(os.path.dirname(folder_predict_json), r"Evaluate/WSIInstanceDiceMulti/Multi")
        folder_wsi_instance_dice = os.path.join(folder_output, r"pkl")

        RunWSIInstance(folder_wsi_instance_mask, folder_predict_json, path_dataset, folder_output, label_name)
        ExportWSIInstanceResult(folder_wsi_instance_dice, folder_predict_json, folder_wsi_instance_mask, folder_output, folder_img, label_name)

if __name__ == "__main__":

    main()