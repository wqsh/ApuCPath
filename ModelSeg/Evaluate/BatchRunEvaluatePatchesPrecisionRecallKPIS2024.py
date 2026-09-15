
import os

import cv2
import numpy as np

from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import CDataWSIDatasetBrevity
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import EnumPadType, EnumDatasetType
from ApuCPathCreateDataset.MSSystem import FOLDER_MASK, FOLDER_DATA, DATASET_NAME
from Evaluate.MSFunctionSegmentEvaluation import CFunctionSegmentEvaluation, SegmentationMetric


# count the patches of Precision, Recall, Dice, quantity and the valid region of patch

def RunEvaluatePatch(path_dataset, folder_predict, folder_label, folder_output, filter_score:float=0.99):

    path_evaluate_result_Precision = os.path.join(folder_output, "EvaluatePatch/PrecisionRecall/evaluatePrecision.txt")
    path_evaluate_result_Recall = os.path.join(folder_output, "EvaluatePatch/PrecisionRecall/evaluateRecall.txt")
    path_evaluate_result_Dice = os.path.join(folder_output, "EvaluatePatch/PrecisionRecall/evaluateDice.txt")
    path_image_num = os.path.join(folder_output, "EvaluatePatch/PrecisionRecall/evaluatePrecisionRecallImageNum.txt")

    path_evaluate_result_Mask_dis = os.path.join(folder_output, r"EvaluatePatch/PrecisionRecall/evaluateMaskDist.txt")

    if not os.path.exists(os.path.dirname(path_evaluate_result_Recall)):
        os.makedirs(os.path.dirname(path_evaluate_result_Precision))

    export_result = True

    list_predict_folder_name = os.listdir(folder_predict)
    dataset_info = CFunctionFileIO.ReadPKL(path_dataset)

    path_result = r""

    list_wsi_name = []
    total_Precision = []
    total_Recall = []
    total_Dice = []
    total_mask_pos_percent = []

    list_image_num = []

    wsi_num = 0
    for path_wsi_info in dataset_info:
        wsi_info: CDataWSIDatasetBrevity = CFunctionFileIO.ReadPKL(path_wsi_info)
        image_num = 0
        # if not wsi_info.name.startswith("test_113"):
        #     continue
        total_Recall.append([])
        total_Precision.append([])
        total_Dice.append([])
        total_mask_pos_percent.append([])
        for predict_folder_name in list_predict_folder_name:
            if predict_folder_name != wsi_info.name:
                continue
            # if wsi_num > 2:
            #     break
            print(wsi_info.name)
            folder_predict_mask = os.path.join(folder_predict, predict_folder_name, EnumPadType.Org.name)
            list_predict_mask_name = os.listdir(folder_predict_mask)
            folder_label_mask = os.path.join(folder_label, FOLDER_DATA, wsi_info.name, EnumPadType.Org.name, FOLDER_MASK)
            list_label_mask_name = os.listdir(folder_label_mask)
            for predict_mask_name in list_predict_mask_name:
                for label_mask_name in list_label_mask_name:
                    if os.path.basename(predict_mask_name).split('.')[0] == os.path.basename(label_mask_name).split('.')[0]:
                        if os.path.basename(predict_mask_name).split('.')[-1] == "pkl":
                            predict_mask = CFunctionFileIO.ReadPKL(os.path.join(folder_predict_mask, predict_mask_name))
                        else:
                            predict_mask = CFunctionFileIO.ReadImage(os.path.join(folder_predict_mask, predict_mask_name))
                        if os.path.basename(label_mask_name).split('.')[-1] == "pkl":
                            label_mask = CFunctionFileIO.ReadPKL(os.path.join(folder_label_mask, label_mask_name))
                        else:
                            label_mask = CFunctionFileIO.ReadImage(os.path.join(folder_label_mask, label_mask_name))
                        predict_mask_data = predict_mask[:, :, 0]

                        # filter_int = int(255 * filter_score)
                        # if filter_int > 255:
                        #     filter_int = 255

                        predict_mask_data[predict_mask_data >= filter_score] = 1
                        predict_mask_data[predict_mask_data < filter_score] = 0
                        predict_mask_data = predict_mask_data.astype(np.uint8)

                        label_mask_data = label_mask[:, :, 0]
                        label_mask_data[label_mask_data > 0] = 1
                        label_mask_data[label_mask_data == 0] = 0
                        metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
                        metric.addBatch(predict_mask_data, label_mask_data)

                        mask_value = np.sum(label_mask[:, :, 0])
                        if mask_value == 0:
                            break

                        Precision = metric.Precision()
                        Recall = metric.Recall()
                        Dice = metric.Dice()


                        image_num += 1

                        total_Precision[-1].append(float(Precision))
                        total_Recall[-1].append(float(Recall))
                        total_Dice[-1].append(float(Dice))
                        total_mask_pos_percent[-1].append(mask_value / (label_mask[:, :, 0].shape[0] * label_mask[:, :, 0].shape[1]))

                        break
            list_image_num.append(image_num)
            list_wsi_name.append(wsi_info.name)
            wsi_num += 1
            print("====================")

    with open(path_evaluate_result_Precision, 'w') as f:
        max_num = 0
        for file_idx, file_name in enumerate(list_wsi_name):
            if len(total_Precision[file_idx]) > max_num:
                max_num = len(total_Precision[file_idx])
            f.write(file_name)
            f.write('\t')
        f.write('\n')
        for i in range(max_num):
            for file_idx, file_name in enumerate(list_wsi_name):
                if len(total_Precision[file_idx]) <= i:
                    f.write('')
                    f.write('\t')
                else:
                    f.write("%.6f\t" % total_Precision[file_idx][i])
            f.write('\n')

    with open(path_evaluate_result_Recall, 'w') as f:
        max_num = 0
        for file_idx, file_name in enumerate(list_wsi_name):
            if len(total_Recall[file_idx]) > max_num:
                max_num = len(total_Recall[file_idx])
            f.write(file_name)
            f.write('\t')
        f.write('\n')
        for i in range(max_num):
            for file_idx, file_name in enumerate(list_wsi_name):
                if len(total_Recall[file_idx]) <= i:
                    f.write('')
                    f.write('\t')
                else:
                    f.write("%.6f\t" % total_Recall[file_idx][i])
            f.write('\n')

    with open(path_evaluate_result_Mask_dis, 'w') as f:
        max_num = 0
        for file_idx, file_name in enumerate(list_wsi_name):
            if len(total_mask_pos_percent[file_idx]) > max_num:
                max_num = len(total_mask_pos_percent[file_idx])
            f.write(file_name)
            f.write('\t')
        f.write('\n')
        for i in range(max_num):
            for file_idx, file_name in enumerate(list_wsi_name):
                if len(total_mask_pos_percent[file_idx]) <= i:
                    f.write('')
                    f.write('\t')
                else:
                    f.write("%.6f\t" % total_mask_pos_percent[file_idx][i])
            f.write('\n')

    with open(path_evaluate_result_Dice, 'w') as f:
        max_num = 0
        for file_idx, file_name in enumerate(list_wsi_name):
            if len(total_Dice[file_idx]) > max_num:
                max_num = len(total_Dice[file_idx])
            f.write(file_name)
            f.write('\t')
        f.write('\n')
        for i in range(max_num):
            for file_idx, file_name in enumerate(list_wsi_name):
                if len(total_Dice[file_idx]) <= i:
                    f.write('')
                    f.write('\t')
                else:
                    f.write("%.6f\t" % total_Dice[file_idx][i])
            f.write('\n')

    with open(path_image_num, 'w') as f:
        f.write('Idx\tWsi Name\tImage num\n')
        for idx, wsi_name in enumerate(list_wsi_name):
            f.write(str(idx))
            f.write('\t')
            f.write(wsi_name)
            f.write('\t')
            f.write(str(list_image_num[idx]))
            f.write('\n')

if __name__ == "__main__":


    list_evaluate = [

        [r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Train",
         r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/UNIMask2FormerStander/UNIMask2Former/Train/20260609100735/PredictTestDataset/UNIMask2Former/Predict",
         r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDatasetTrain",
         ],

        # [r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Train",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/Unet/Unet/Train/20260612102707/PredictTestDataset/Unet/Predict",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDatasetTrain",
        #  ],

        # [r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDataset.Train",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ModelTrain/MaskRCNN/MaskRCNN/Train/20260623222335/PredictTestDataset/MaskRCNN/Predict",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TestDataset/ApuCPathDatasetTrain",
        #  ],

        # [r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Train",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/MaskRCNN/MaskRCNN/Train/20260610003857/PredictTestDataset/MaskRCNN/Predict",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDatasetTrain",
        #  ],
        # 
        # [r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Train",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/Unet/Unet/Train/20260610080756/PredictTestDataset/Unet/Predict",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDatasetTrain",
        #  ],

        # [r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDataset.Train",
        #  r"//media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TrainDataset/ModelTrain/UNIMask2Former/UNIMask2Former/Train/20260611085630/PredictTestDataset/UNIMask2Former/Predict",
        #  r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI2024Normal/TestDataset/ApuCPathDatasetTrain",
        #  ],

    ]
    for item in list_evaluate:
        path_dataset = item[0]
        folder_predict = item[1]
        folder_label = item[2]
        folder_output = os.path.dirname(item[1])
        RunEvaluatePatch(path_dataset, folder_predict, folder_label, folder_output, filter_score=0.7)

        folder_path_result = os.path.join(folder_output, "EvaluatePatch/PrecisionRecall")

        path_precision = os.path.join(folder_path_result, "evaluatePrecision.txt")
        path_recall = os.path.join(folder_path_result, "evaluateRecall.txt")
        path_dice = os.path.join(folder_path_result, "evaluateDice.txt")
        path_img_num = os.path.join(folder_path_result, "evaluatePrecisionRecallImageNum.txt")
        path_dist = os.path.join(folder_path_result, "evaluateMaskDist.txt")

        path_output_merge = os.path.join(folder_path_result, "Collect.txt")
        path_output_patch_dist = os.path.join(folder_path_result, "Dist.txt")

        if not os.path.exists(os.path.dirname(path_output_merge)):
            os.makedirs(os.path.dirname(path_output_merge))

        with open(path_img_num, 'r') as f:
            buffer = f.readlines()

        list_img_num = []
        line_num = 0
        for line in buffer:
            if line_num == 0:
                pass
            else:
                list_line = line.strip().split('\t')
                list_img_num.append([list_line[1], int(list_line[2])])
            line_num += 1

        line_num = 0
        list_precision = []
        with open(path_precision, 'r') as f:
            buffer = f.readlines()
        for line in buffer:
            list_line = line[:-1].split('\t')
            if line_num == 0:
                for item in list_line:
                    list_precision.append([item, [], 0])
            else:
                for i, item in enumerate(list_line):
                    if len(item) == 0:
                        pass
                    else:
                        list_precision[i][2] += 1
                        list_precision[i][1].append(float(item))
            line_num += 1

        line_num = 0
        list_recall = []
        with open(path_recall, 'r') as f:
            buffer = f.readlines()
        for line in buffer:
            list_line = line[:-1].split('\t')
            if line_num == 0:
                for item in list_line:
                    list_recall.append([item, [], 0])
            else:
                for i, item in enumerate(list_line):
                    if len(item) == 0:
                        pass
                    else:
                        list_recall[i][2] += 1
                        list_recall[i][1].append(float(item))
            line_num += 1

        line_num = 0
        list_dice = []
        with open(path_dice, 'r') as f:
            buffer = f.readlines()
        for line in buffer:
            list_line = line[:-1].split('\t')
            if line_num == 0:
                for item in list_line:
                    list_dice.append([item, [], 0])
            else:
                for i, item in enumerate(list_line):
                    if len(item) == 0:
                        pass
                    else:
                        list_dice[i][2] += 1
                        list_dice[i][1].append(float(item))
            line_num += 1

        line_num = 0
        list_dist = []
        with open(path_dist, 'r') as f:
            buffer = f.readlines()
        for line in buffer:
            list_line = line[:-1].split('\t')
            if line_num == 0:
                for item in list_line:
                    list_dist.append([item, [], 0])
            else:
                for i, item in enumerate(list_line):
                    if len(item) == 0:
                        pass
                    else:
                        list_dist[i][2] += 1
                        list_dist[i][1].append(float(item))
            line_num += 1

        with open(path_output_merge, 'w') as f:
            f.write('idx\twsi name\timage num\tprecision\trecall\tdice\n')
            for idx, img_info in enumerate(list_img_num):
                f.write(str(idx + 1))
                f.write('\t')
                f.write(img_info[0])
                f.write('\t')
                f.write(str(img_info[1]))
                f.write('\t')
                for j in range(len(list_precision)):
                    if list_precision[j][0] == img_info[0]:
                        f.write("%.6f" % (sum(list_precision[j][1]) / list_precision[j][2]))
                        f.write('\t')
                        break

                for j in range(len(list_recall)):
                    if list_recall[j][0] == img_info[0]:
                        f.write("%.6f" % (sum(list_recall[j][1]) / list_recall[j][2]))
                        f.write('\t')
                        break

                for j in range(len(list_dice)):
                    if list_dice[j][0] == img_info[0]:
                        f.write("%.6f" % (sum(list_dice[j][1]) / list_dice[j][2]))
                        f.write('\t')
                        break
                f.write('\n')

        with open(path_output_patch_dist, 'w') as f:
            f.write('idx\twsi name\tdist\tprecision\trecall\tdice\n')
            count = 0
            for image_idx, image_num_info in enumerate(list_img_num):
                for dist_idx, dist_info in enumerate(list_dist):
                    if dist_info[0] == image_num_info[0]:
                        f.write("%d" % (count + 1))
                        f.write('\t')
                        f.write(image_num_info[0])
                        f.write('\t')
                        for i in range(image_num_info[1]):
                            if i > 0:
                                f.write("%d" % (count + 1))
                                f.write('\t')
                                f.write('')
                                f.write('\t')
                            f.write('%.6f' % list_dist[image_idx][1][i])
                            f.write('\t')
                            f.write('%.6f' % list_precision[image_idx][1][i])
                            f.write('\t')
                            f.write('%.6f' % list_recall[image_idx][1][i])
                            f.write('\t')
                            f.write('%.6f' % list_dice[image_idx][1][i])
                            f.write('\t')
                            f.write('\n')
                            count += 1

                        break
    a = 1