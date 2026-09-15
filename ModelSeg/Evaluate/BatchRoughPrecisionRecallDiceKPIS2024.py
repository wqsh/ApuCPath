# for the instance targets from wsi by use Multi receptive fields fusion
import copy

from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.Route.MSFunctionDataJsonImageLabel import CFunctionDataJsonImageLabel
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import CDataWSIDatasetBrevity
from Evaluate.MSFunctionSegmentEvaluation import CFunctionWSIInstance

import os
import numpy as np

def GetWSIEvaluate(path_org_mask, path_predict_rough_mask, folder_output):

    img_org_mask_pos = CFunctionImageOperator.RGB2Gray(CFunctionFileIO.ReadImage(path_org_mask))
    CLogging.LogToUser(" Get pos")
    CFunctionMemory.GetCurMemoryInfo()

    img_predict_mask = CFunctionFileIO.ReadNpy(path_predict_rough_mask)
    img_predict_mask = copy.deepcopy(img_predict_mask[:img_org_mask_pos.shape[0], :img_org_mask_pos.shape[1]])
    CLogging.LogToUser("Finish get predict pos")
    CFunctionMemory.GetCurMemoryInfo()

    annotation_h, annotation_w = img_org_mask_pos.shape
    predict_h, predict_w = img_predict_mask.shape
    CLogging.LogToUser(f"annotation h: {str(annotation_h)}, annotation w: {str(annotation_w)}")
    CLogging.LogToUser(f"predict h: {str(predict_h)}, predict w: {str(predict_w)}")
    # check_img = np.zeros((annotation_h, annotation_w, 3), dtype=np.uint8)
    # check_img[:annotation_h, :annotation_w, 0] += img_org_mask_pos[:annotation_h, :annotation_w]
    # check_img[:annotation_h, :annotation_w, 1] += img_predict_mask[:annotation_h, :annotation_w]
    # CFunctionFileIO.WriteTIFByGDAL(folder_output, "check.tiff", check_img)

    count_dict = CFunctionWSIInstance.GetWSICountMatrix(img_predict_mask, img_org_mask_pos, valid_region=False)
    del img_org_mask_pos
    del img_predict_mask
    CFunctionMemory.CleanMemory()

    precision = count_dict['TP'] / (count_dict['FP'] + count_dict["TP"])
    recall = count_dict['TP'] / (count_dict['FN'] + count_dict["TP"])
    dice = 2 * count_dict['TP'] / (2 * count_dict["TP"] + count_dict['FP'] + count_dict['FN'])
    print("precision %.6f"%precision)
    print("recall %.6f"%recall)
    print("dice %.6f"%dice)

    return precision, recall, dice

def RunWSIEvaluate(path_dataset, folder_predict_mask, folder_org_mask, folder_output, label_name:str="label"):

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    path_instance_result = os.path.join(folder_output, "WSIOrg.txt")

    list_org_mask_name = os.listdir(folder_org_mask)

    result_list = []

    CLogging.LogToUser("Start ================ ")

    list_path_wsi_dataset = CFunctionFileIO.ReadPKL(path_dataset)
    for path_wsi_dataset in list_path_wsi_dataset:
        wsi_name = os.path.basename(path_wsi_dataset).split('.')[0]
        CLogging.LogToUser("Predict wsi : " + wsi_name)
        # if wsi_name != "test_116":
        #     continue
        wsi_dataset_info: CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(path_wsi_dataset)
        path_image = wsi_dataset_info.image_path
        path_org_mask = None
        for org_mask_name in list_org_mask_name:
            if org_mask_name.startswith(wsi_name):
                path_org_mask = os.path.join(folder_org_mask, org_mask_name)
                break

        folder_predict_rough_mask = os.path.join(folder_predict_mask, wsi_name, "processed")
        path_predict_rough_mask = None
        for predict_rough_mask_name in os.listdir(folder_predict_rough_mask):
            if predict_rough_mask_name.endswith(".npy"):
                path_predict_rough_mask = os.path.join(folder_predict_rough_mask, predict_rough_mask_name)
        if path_predict_rough_mask is None:
            continue
        precision, recall, dice = GetWSIEvaluate(path_org_mask, path_predict_rough_mask, folder_output)

        CLogging.LogToUser("Instance precision : %.6f" % precision)
        CLogging.LogToUser("Instance recall : %.6f" % recall)
        CLogging.LogToUser("Instance dice : %.6f" % dice)
        result_list.append([wsi_name, "%.6f" % precision, "%.6f" % recall, "%.6f" % dice])

    with open(path_instance_result, 'w') as f:
        f.write("wsi name\tprecision\trecall\tdice\n")
        for result in result_list:
            for item in result:
                f.write(item)
                f.write('\t')
            f.write('\n')

def main():
    list_evaluate = [
        [
            r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Predict",
            r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/mask",
            r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/UNIMask2FormerStander/UNIMask2Former/Train/20260623222336/PredictTestDatasetWSI/UNIMask2Former/Predict",
        ],
        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/Unet/Unet/Train/20260612102707/PredictTestDatasetWSI/Unet/Predict",
        # ],
        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/MaskRCNN/Train/20260518230315/PredictTestDataset/MaskRCNN/Predict",
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/MaskRCNN/Train/20260520105324/PredictTestDataset/MaskRCNN/Predict",
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/Unet/Unet/Train/20260610080756/PredictTestDatasetWSI/Unet/Predict",
        # ],


        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/MaskRCNN/MaskRCNN/Train/20260610003857/PredictTestDatasetWSI/MaskRCNN/Predict",
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/Normal/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/UNIMask2Former/UNIMask2Former/Train/20260611085630/PredictTestDatasetWSI/UNIMask2Former/Predict",
        # ],
    ]

    label_name = "Pos"

    for item in list_evaluate:
        path_dataset = item[0]
        folder_org_mask = item[1]
        folder_predict_mask = item[2]
        folder_output = os.path.join(os.path.dirname(folder_predict_mask), r"EvaluatePatch/WSIPrecisionRecallOrg")

        RunWSIEvaluate(path_dataset, folder_predict_mask, folder_org_mask, folder_output, label_name)

if __name__ == "__main__":

    main()