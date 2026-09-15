import os

import numpy as np
import copy

from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

class CFunctionSegmentEvaluation:

    @staticmethod
    def IoU(predict_mask:np.array, true_mask:np.array):
        s = np.zeros(predict_mask.shape)
        s[predict_mask > 0] = 1
        g = np.zeros(true_mask.shape)
        g[true_mask > 0] = 1
        intersecion = np.multiply(s, g)
        # 两者相加，值大于0的部分为交集
        union = np.asarray(s + g > 0, np.float32)
        iou = intersecion.sum() / (union.sum() + 1e-10)
        return iou

    @staticmethod
    def Dice(s, g):
        """
        calculate the Dice score of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
        assert (len(s.shape) == len(g.shape))
        prod = np.multiply(s, g)
        s0 = prod.sum()
        dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
        return dice


"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union # 返回列表，其值为各个类别的IoU
        if np.sum(self.confusionMatrix[0, :]) == 0:
            mIoU = np.nanmean(IoU[1:])
        else:
            mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        self.TN = self.confusionMatrix[0][0]
        self.FN = self.confusionMatrix[1][0]
        self.FP = self.confusionMatrix[0][1]
        self.TP = self.confusionMatrix[1][1]

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def TPR(self):
        if self.TP + self.FN == 0:
            print("TPR error")
        return self.TP / (self.TP + self.FN)

    def FPR(self):
        if self.FP + self.TN == 0:
            print("FPR error")
        return self.FP / (self.FP + self.TN)

    def TNR(self):

        return self.TN / (self.FP + self.TN)

    def FNR(self):

        return self.FN / (self.TP + self.FN)

    def Precision(self):
        if self.TP + self.FP == 0:
            return 0
        else:
            return self.TP / (self.FP + self.TP)

    def Recall(self):

        return self.TP / (self.FN + self.TP)

    def Dice(self):

        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

class CFunctionWSIInstance():

    @staticmethod
    def PixelAcc(wsi_predict_mask:np.array, wsi_label_mask:np.array):

        CLogging.LogToUser("Start pixel acc")
        CFunctionMemory.GetCurMemoryInfo()
        CFunctionToolImageC.SetMatrixValue(wsi_predict_mask, 2)
        CFunctionToolImageC.SetMatrixValue(wsi_label_mask, 1)

        merge_mask = copy.deepcopy(wsi_label_mask)
        CLogging.LogToUser(" Get merge mask ")
        CFunctionMemory.GetCurMemoryInfo()

        CLogging.LogToUser("Check 1")

        CFunctionToolImageC.AddMatrixValue(merge_mask, wsi_predict_mask)
        CLogging.LogToUser("Check 2")

        CLogging.LogToUser("Check 3")

        count_values = [0 for i in range(4)]
        for value in range(len(count_values)):
            count_values[value] = np.count_nonzero(merge_mask == value)
        if (count_values[1] + count_values[3]) == 0:
            instancePA = 0
        else:
            instancePA = count_values[3] / (count_values[1] + count_values[3])
        del merge_mask
        CFunctionMemory.CleanMemory()
        CLogging.LogToUser("Finish pixel acc")
        CFunctionMemory.GetCurMemoryInfo()
        return instancePA

    @staticmethod
    def InstancePixelFNR(wsi_predict_mask:np.array, wsi_label_mask:np.array):
        # 漏报率

        CLogging.LogToUser("Start FNR")
        CFunctionMemory.GetCurMemoryInfo()
        CFunctionToolImageC.SetMatrixValue(wsi_predict_mask, 1)
        CFunctionToolImageC.SetMatrixValue(wsi_label_mask, 2)

        # 获取有效的实例区域mask（与标注示例有交集的部分）
        valid_region_mask = copy.deepcopy(wsi_label_mask)
        CLogging.LogToUser(" Get valid_region_mask ")
        CFunctionMemory.GetCurMemoryInfo()

        CFunctionToolImageC.AddMatrixValue(valid_region_mask, wsi_predict_mask)
        CLogging.LogToUser("Count values ... ")
        CFunctionMemory.GetCurMemoryInfo()

        count_values = CFunctionToolImageC.CountValues(valid_region_mask)

        del valid_region_mask
        CFunctionMemory.CleanMemory()

        CLogging.LogToUser(" Delete valid_region_mask  ")
        CFunctionMemory.GetCurMemoryInfo()

        CLogging.LogToUser("Finish FNR")
        if (count_values[2] + count_values[3] == 0):
            fpr = 0.0
        else:
            fpr = (count_values[2] / (count_values[2] + count_values[3]))
        return fpr

    @staticmethod
    def GetCountVaidRegion(wsi_predict_mask:np.array, wsi_label_mask:np.array):
        CLogging.LogToUser("Start get count valid region matrix")
        CFunctionToolImageC.SetMatrixValue(wsi_predict_mask, 1)
        CFunctionToolImageC.SetMatrixValue(wsi_label_mask, 2)

        # 获取获取标注区域的编号（与标注示例有交集的部分）
        CLogging.LogToUser("Get label mask seed point ... ")
        wsi_label_mask_label, wsi_label_mask_label_num = CFunctionImageOperator.GetImageMask2Label(wsi_label_mask)
        CFunctionMemory.GetCurMemoryInfo()
        np_label_mask_label_seedpoint = CFunctionToolImageC.GetLabelSeedPoint(wsi_label_mask_label, [i + 1 for i in range(wsi_label_mask_label_num)])
        del wsi_label_mask_label
        CFunctionMemory.CleanMemory()
        CLogging.LogToUser("Finish get label mask seed point. ")
        CFunctionMemory.GetCurMemoryInfo()

        CLogging.LogToUser("Start filter valid region ... ")
        valid_region_mask = np.copy(wsi_label_mask)
        CFunctionToolImageC.AddMatrixValue(valid_region_mask, wsi_predict_mask)
        CFunctionToolImageC.SetMatrixValue(valid_region_mask, 255)

        CLogging.LogToUser("label predict and label region ... ")
        valid_region_mask_label, valid_region_mask_label_num = CFunctionImageOperator.GetImageMask2Label(valid_region_mask)

        CFunctionMemory.GetCurMemoryInfo()
        np_valid_region_mask_label_seedpoint = CFunctionToolImageC.GetLabelSeedPoint(valid_region_mask_label, [i + 1 for i in range(valid_region_mask_label_num)])
        retain_region_idx_list = []
        for seedpoint in np_label_mask_label_seedpoint:
            valid_region_mask_label_idx = valid_region_mask_label[seedpoint.seed_y, seedpoint.seed_x]
            retain_region_idx_list.append(valid_region_mask_label_idx)
        del valid_region_mask_label
        CFunctionMemory.CleanMemory()
        CLogging.LogToUser("Get predict and label region. ")
        CFunctionMemory.GetCurMemoryInfo()

        remove_region_idx_list = []
        for i in range(1, valid_region_mask_label_num + 1):
            if not i in retain_region_idx_list:
                remove_region_idx_list.append(i)
        for remove_idx in remove_region_idx_list:
            CFunctionToolImageC.RemoveShape(valid_region_mask, np_valid_region_mask_label_seedpoint[remove_idx - 1].seed_x, np_valid_region_mask_label_seedpoint[remove_idx - 1].seed_y)
        CLogging.LogToUser("Get valid region. ")
        return valid_region_mask

    @staticmethod
    def GetWSICountMatrix(wsi_predict_mask:np.array, wsi_label_mask:np.array, valid_region:bool=False):
        CLogging.LogToUser("Start get count matrix")
        CFunctionMemory.GetCurMemoryInfo()
        CFunctionToolImageC.SetMatrixValue(wsi_predict_mask, 1)
        CFunctionToolImageC.SetMatrixValue(wsi_label_mask, 2)

        if valid_region:
        # 获取有效的实例区域mask（与标注示例有交集的部分）
            valid_region_mask = CFunctionWSIInstance.GetCountVaidRegion(wsi_predict_mask, wsi_label_mask)
            CFunctionMemory.CleanMemory()

            CLogging.LogToUser("Get count matrix .. ")
            CFunctionToolImageC.SetMatrixValue(valid_region_mask, 3)
            CFunctionToolImageC.AddMatrixValue(valid_region_mask, wsi_predict_mask)
            CFunctionToolImageC.AddMatrixValue(valid_region_mask, wsi_label_mask)
            count_mask = valid_region_mask
            count_values = CFunctionToolImageC.CountValues(count_mask)
            count_dict = {
                "TN": count_values[0],
                "FN": count_values[5],
                "FP": count_values[4],
                "TP": count_values[6],
                }

        else:
            count_mask = copy.deepcopy(wsi_label_mask)
            CFunctionToolImageC.AddMatrixValue(count_mask, wsi_predict_mask)

            CLogging.LogToUser("Count values ... ")
            CFunctionMemory.GetCurMemoryInfo()

            count_values = CFunctionToolImageC.CountValues(count_mask)
            count_dict = {
                "TN": count_values[0],
                "FN": count_values[2],
                "FP": count_values[1],
                "TP": count_values[3],
            }
        return count_dict

    # @staticmethod
    # def PixelFNR(wsi_predict_mask:np.array, wsi_
    #     # @staticmethodlabel_mask:np.array):
    #     # 漏报率
    #     CFunctionToolImageC.SetMatrixValue(wsi_predict_mask, 2)
    #     CFunctionToolImageC.SetMatrixValue(wsi_label_mask, 1)

    @staticmethod
    def __GetValidRegion(wsi_predict_mask:np.array, wsi_label_mask:np.array):
        CFunctionToolImageC.SetMatrixValue(wsi_predict_mask, 1)
        CFunctionToolImageC.SetMatrixValue(wsi_label_mask, 2)

        wsi_label_mask_label, wsi_label_mask_label_num = CFunctionImageOperator.GetImageMask2Label(wsi_label_mask)
        wsi_label_mask_label_seed_point = CFunctionToolImageC.GetLabelSeedPoint(wsi_label_mask_label, [i + 1 for i in
                                                                                                       range(
                                                                                                           wsi_label_mask_label_num)])

        valid_region_mask = copy.deepcopy(wsi_label_mask)
        CFunctionToolImageC.AddMatrixValue(valid_region_mask, wsi_predict_mask)
        valid_region_mask_label, valid_region_mask_label_num = CFunctionImageOperator.GetImageMask2Label(
            valid_region_mask)
        valid_region_mask_label_seed_point = CFunctionToolImageC.GetLabelSeedPoint(valid_region_mask_label, [i + 1 for i in range(valid_region_mask_label_num)])

        retain_region_list = []
        for seed_point in wsi_label_mask_label_seed_point:
            retain_region_list.append(valid_region_mask_label[seed_point.seed_y, seed_point.seed_x])
        retain_region_list = list(set(retain_region_list))

        for label_idx in range(wsi_label_mask_label_num):
            if label_idx in retain_region_list:
                continue
            else:
                CFunctionToolImageC.RemoveShape(valid_region_mask, valid_region_mask_label_seed_point[label_idx].seed_x,
                                                valid_region_mask_label_seed_point[label_idx].seed_y)
        return valid_region_mask

if __name__ == '__main__':
    imgPredict = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成预测图片
    imgLabel = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成标注图片
    metric = SegmentationMetric(3)  # 3表示有3个分类，有几个分类就填几
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print('pa is : %f' % pa)
    print('cpa is :')  # 列表
    print(cpa)
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % mIoU)