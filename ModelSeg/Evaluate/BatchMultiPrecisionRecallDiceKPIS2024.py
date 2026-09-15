# for the instance targets from wsi by use Multi receptive fields fusion

from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.Route.MSFunctionDataJsonImageLabel import CFunctionDataJsonImageLabel
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import CDataWSIDatasetBrevity
from Evaluate.MSFunctionSegmentEvaluation import CFunctionWSIInstance

import os

def GetWSIEvaluate(path_image, path_org_mask, path_predict_json, label_name:str="label"):

    img_org_mask_pos = CFunctionImageOperator.RGB2Gray(CFunctionFileIO.ReadImage(path_org_mask))
    CLogging.LogToUser(" Get pos")
    CFunctionMemory.GetCurMemoryInfo()

    img_predict_mask = CFunctionDataJsonImageLabel.GetJson2Mask(path_image, path_predict_json, label_name, 255)

    CLogging.LogToUser("Finish get predict pos")
    CFunctionMemory.GetCurMemoryInfo()

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

def RunWSIEvaluate(path_dataset, folder_predict_json, folder_org_mask, folder_output, label_name:str="label"):

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    path_instance_result = os.path.join(folder_output, "WSIMulti.txt")

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
        path_predict_json = None
        for item in os.listdir(os.path.join(folder_predict_json, wsi_name)):
            if item.endswith("json"):
                path_predict_json = os.path.join(folder_predict_json, wsi_name, item)
                break
        if path_predict_json is None:
            CLogging.LogWarning(wsi_name + ' path_predict_json is not exist! ')
            continue
        precision, recall, dice = GetWSIEvaluate(path_image, path_org_mask, path_predict_json, label_name)

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
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/MaskRCNN/MaskRCNN/Train/20260623222335/PredictTestDatasetWSI/MaskRCNN/Predict",
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/56Nx/Testing/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/MaskRCNN/Train/20260520105324/PredictTestDatasetWSI/MaskRCNN/Predict",
        # ],
        #
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
        #
        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DN/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DNHIL/TrainDataset/Start/ModelTrain/Mask2Former/UNIMask2Former/Train/20260619235652/PredictTestDatasetWSI/UNIMask2Former/Predict",
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DN/TestDataset/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Test/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DNHIL/TrainDataset/Second/ModelTrain/UNIMask2Former/UNIMask2Former/Train/20260624145739/PredictTestDatasetWSI/UNIMask2Former/Predict",
        # ],

        # [
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DNHIL/TrainDataset/TrainDatasetForPredict/ApuCPathDataset.Predict",
        #     r"/media/wqs/SOFT/weiqiushi/Data/KPIs2024/DN/Train/mask",
        #     r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024DN/TrainDataset/ModelTrain/UNIMask2Former/UNIMask2Former/Train/20260623094818/PredictTrainDatasetWSI/UNIMask2Former/Predict",
        # ],

    ]

    label_name = "Pos"

    for item in list_evaluate:
        path_dataset = item[0]
        folder_org_mask = item[1]
        folder_predict_json = item[2]
        folder_output = os.path.join(os.path.dirname(folder_predict_json), r"EvaluatePatch/WSIPrecisionRecallMulti")

        RunWSIEvaluate(path_dataset, folder_predict_json, folder_org_mask, folder_output, label_name)

if __name__ == "__main__":

    main()