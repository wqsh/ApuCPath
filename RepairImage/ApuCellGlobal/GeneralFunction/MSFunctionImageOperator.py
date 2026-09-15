import cv2
import numpy as np

from ApuCellGlobal.MSApuCellExternalPackge import *
from ApuCellGlobal.GeneralSystem.MSSystemParameter import UINT_MAX
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.GeneralSystem.MSSystemImage import LIST_COLOR_CHANNEL
from ApuCellGlobal.Route.MSFunctionRouteMove import CFunctionRouteMove

import random

from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory


class CFunctionImageOperator:

    @staticmethod
    def GetImageMaskLine(img_mask:np.array, dilation:bool=True, iterations:int=2):
        # 原始图像的mask只保留边界圈选的
        img_line = np.where((img_mask < 255) & (img_mask > 0), 1, 0)
        max_img_line_value = np.max(img_line)
        if max_img_line_value == 0:
            # 这种情况相当于没有提供边界线
            img_flag = CFunctionImageOperator.GetImageMaskFlag(img_mask)
            if dilation:
                boundary_1 = ndimage.binary_dilation(img_flag, iterations=iterations).astype(img_flag.dtype)
                img_line = boundary_1 - img_flag
            else:
                boundary_1 = ndimage.binary_erosion(img_flag, iterations=iterations).astype(img_flag.dtype)
                img_line = img_flag - boundary_1
        return img_line

    @staticmethod
    def GetImageLabelLineByPath(img_label:np.array, search_list:list):

        img_line = np.zeros(img_label.shape, dtype=np.uint8)
        seed_point_list = CFunctionToolImageC.GetLabelSeedPoint(img_label, search_list)
        img_mask = img_label.astype(np.uint8)
        for seed_point_idx, seed_point in enumerate(seed_point_list):
            CFunctionToolImageC.GetMaskLineBySeedPoint(img_mask, img_line, seed_point.seed_x, seed_point.seed_y)
        return img_line

    @staticmethod
    def GetImageMaskFlag(img_mask:np.array):
        img_flag = copy.deepcopy(img_mask)
        img_flag[img_flag > 0] = 1
        return img_flag

    @staticmethod
    def GetImageMaskLabels(img_mask:np.array, connectivity=2):
        img_labels = measure.label(img_mask, connectivity=connectivity)  # 8连通区域标记
        max_label = np.max(img_labels)
        if max_label > UINT_MAX:
            CLogging.LogWarning("The label numbers may overflow, if you use for the function of MSFunctionToolC. ")
        return img_labels

    @staticmethod
    def GetImageMask2Label(img_mask:np.array):
        img_label = np.zeros(img_mask.shape, dtype=np.uint32)
        label_num = CFunctionToolImageC.Mask2Label(img_label, img_mask)
        return img_label, label_num

    @staticmethod
    def GetImageLabelPositions(img_labels:np.array):

        img_labels_positions = []
        for region in measure.regionprops(img_labels):
            img_labels_positions.append([region.bbox[1], region.bbox[0]])
        return img_labels_positions

    @staticmethod
    def GetImageLabelsMask(img_labels:np.array):
        img_mask = copy.deepcopy(img_labels)
        img_mask.astype(np.uint8)
        for i in range(1, np.max(img_labels) + 1):
            img_mask[img_mask == i] = 255
        return img_mask

    @staticmethod
    def GetImageLabelFlag(img_label:np.array):
        img_flag = copy.deepcopy(img_label)
        img_flag[img_flag > 0] = 1
        return img_flag

    @staticmethod
    def GetImageLineFlag(img_line:np.array):
        img_flag = copy.deepcopy(img_line)
        img_flag[img_flag!=0] = 1
        return img_flag

    @staticmethod
    def SpinImage(image:np.array, n:int, clockwise:bool)->np.array:
        if n > 3:
            n = 0
        if clockwise:
            return np.rot90(image, 4 - n)
        else:
            return np.rot90(image, n)

    @staticmethod
    def VerticalMirrorImage(image:np.array)->np.array:
        return np.flipud(image)

    @staticmethod
    def HorizontalMirrorImage(image:np.array)->np.array:
        return np.fliplr(image)

    @staticmethod
    def HistogramEqualization(image:np.array)->np.array:

        return cv2.equalizeHist(image)

    @staticmethod
    def RGB2Gray(image:np.array)->np.array:
        if len(image.shape) == 2:
            return image
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def Gray2RGB(image_gray:np.array)->np.array:
        if len(image_gray.shape) == 3 and image_gray.shape[2] == 3:
            return image_gray
        elif len(image_gray.shape) == 2:
            return cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        else:
            return None
    @staticmethod
    def EqualizeHist(image:np.array):
        image_shape = image.shape
        if len(image_shape) == 2:
            return cv2.equalizeHist(image)

    @staticmethod
    def Dilate(image:np.array, kernel_size:int):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(image, kernel)

    @staticmethod
    def Erode(image:np.array, kernel_size:int):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.erode(image, kernel)

    @staticmethod
    def Quantization(img:np.array, level_num:int=8, max_level_value:int=256):
        section_length = int(max_level_value / level_num)
        level_list = [i * section_length for i in range(level_num)]
        new_matrix = np.zeros(img.shape, dtype=np.uint8)
        for i in range(level_num):
            if i == level_num - 1:
                new_matrix[img >= level_list[i]] = i
            else:
                new_matrix[(img >= level_list[i]) & (img < level_list[i + 1])] = i
        return new_matrix

    @staticmethod
    def QuantizationMatlab(img:np.array, level_num:int=8, max_level_value:int=256):
        section_length = int(max_level_value / level_num)
        level_list = [i * section_length for i in range(level_num)]
        new_matrix = np.zeros(img.shape, dtype=np.uint8)
        for i in range(level_num):
            if i == level_num - 1:
                new_matrix[img >= level_list[i]] = i + 1
            else:
                new_matrix[(img >= level_list[i]) & (img < level_list[i + 1])] = i + 1
        return new_matrix

    @staticmethod
    def GetMaskLineByValue(img_mask:np.array, value:int):
        img_line = np.zeros(img_mask.shape, dtype=np.uint8)
        CFunctionToolImageC.GetMaskLineByValue(img_mask, img_line, value)
        return img_line


    @staticmethod
    def MatrixDifference(data_matrix: np.array, order: int = 1):
        # 求解差分
        difference_matrix = copy.deepcopy(data_matrix)
        for i in range(order):
            delta_row = np.diff(difference_matrix[:, :], axis=0)  # 上下行差分
            delta_row_reverse = np.diff(difference_matrix[:, :], axis=0) * -1  # 上下行差分
            delta_col = np.diff(difference_matrix[:, :], axis=1)  # 左右行差分
            delta_col_reverse = np.diff(difference_matrix[:, :], axis=1) * -1  # 左右列差分

            delta_row[delta_row <= 0] = 0
            delta_row_reverse[delta_row_reverse <= 0] = 0
            delta_col_reverse[delta_col_reverse <= 0] = 0
            delta_col[delta_col <= 0] = 0

            difference_matrix[1:, :] += delta_row
            difference_matrix[:-1, :] += delta_row_reverse

            difference_matrix[:, 1:] += delta_col
            difference_matrix[:, :-1] += delta_col_reverse
        return difference_matrix

    @staticmethod
    def IsInteriorPoint(mask: np.array, point_x, point_y, top, bottom, left, right, boundary_value):

        ret_top = False
        ret_bottom = False
        ret_left = False
        ret_right = False

        for i in range(point_y, top - 1, -1):
            if mask[i, point_x] == boundary_value:
                ret_top = True
                break
            elif mask[i, point_x] != 0:
                break

        for i in range(point_y, bottom + 1, 1):
            if mask[i, point_x] == boundary_value:
                ret_bottom = True
                break
            elif mask[i, point_x] != 0:
                break

        for i in range(point_x, left - 1, -1):
            if mask[point_y, i] == boundary_value:
                ret_left = True
                break
            elif mask[point_y, i] != 0:
                break

        for i in range(point_x, right + 1, 1):
            if mask[point_y, i] == boundary_value:
                ret_right = True
                break
            elif mask[point_y, i] != 0:
                break

        if (ret_top and ret_bottom and ret_left and ret_right):
            return True
        else:
            return False

    @staticmethod
    def FillPolygons(mask: np.array, threshold_value:int=254, fill_value:int=255):
        mask_label, label_num = CFunctionImageOperator.GetImageMask2Label(mask)
        np_seed_point = CFunctionToolImageC.GetLabelBoundarySeedPoint(mask_label, label_num)
        list_boundary = CFunctionRouteMove.GetLabelBoundary(mask_label, label_num, np_seed_point)
        del mask_label
        CFunctionMemory.CleanMemory()

        polygons_mask = np.zeros(mask.shape, dtype=np.uint8)
        for one_boundary in list_boundary:
            if len(one_boundary) == 1:
                polygons_mask[one_boundary[0][1], one_boundary[0][0]] = 255
            elif len(one_boundary) == 0:
                continue
            else:
                for points_idx in range(1, len(one_boundary)):
                    cv2.line(polygons_mask, one_boundary[points_idx - 1], one_boundary[points_idx], 255, 1)
                cv2.line(polygons_mask, one_boundary[-1], one_boundary[0], 255, 1)

        polygons_mask_label, polygons_label_num = CFunctionImageOperator.GetImageMask2Label(polygons_mask)
        np_seed_point_polygons = CFunctionToolImageC.GetLabelSeedPoint(polygons_mask_label, [i + 1 for i in range(polygons_label_num)])

        # check label num diff between input and cv2 line
        # aa_org = [(np_seed_point[0][i*2], np_seed_point[0][i*2 + 1]) for i in range(1, len(np_seed_point[0]) // 2)]
        # aa_new = [(seed_point.seed_y, seed_point.seed_x) for seed_point in np_seed_point_polygons]
        #
        # for idx, org_seed_point in enumerate(aa_new):
        #     new_seed_point = aa_org[idx]
        #     if (new_seed_point[0] != org_seed_point[0]) or (new_seed_point[1] != org_seed_point[1]):
        #         a = 1

        for label_idx, seed_point in enumerate(np_seed_point_polygons):
            find_zeros = False
            fill_seed_y = 0
            fill_seed_x = 0
            for x in range(seed_point.x_min, seed_point.x_max + 1):
                meet_first_point = False
                for y in range(seed_point.y_min, seed_point.y_max + 1):
                    if not meet_first_point:
                        if (label_idx + 1) == polygons_mask_label[y, x]:
                            meet_first_point = True
                            # 遇到第一个边界点
                            valid_idx_list = [y]
                            for k in range(y + 1, seed_point.y_max + 1):
                                # 找到边界点之间的非0点，如果这个非0点往四个方向碰到了边界，那么它一定是内部点
                                if polygons_mask_label[k, x] == (label_idx + 1):
                                    if k - valid_idx_list[-1] > 1:
                                        if CFunctionImageOperator.IsInteriorPoint(polygons_mask_label, x, valid_idx_list[-1] + 1, seed_point.y_min, seed_point.y_max, seed_point.x_min, seed_point.x_max, label_idx + 1):
                                            find_zeros = True
                                            fill_seed_x = x
                                            fill_seed_y = valid_idx_list[-1] + 1
                                            break
                                    else:
                                        valid_idx_list.append(k)
                                elif polygons_mask_label[k, x] != 0:
                                    break
                                else:
                                    pass
                        else:
                            pass
                    else:
                        break
                if find_zeros:
                    break
            if find_zeros:
                CFunctionToolImageC.FillPolygons(polygons_mask, fill_seed_x, fill_seed_y, threshold_value, fill_value)
        del polygons_mask_label
        CFunctionMemory.CleanMemory()
        CFunctionToolImageC.ReplaceMatrixValue(mask, polygons_mask)
        del polygons_mask
        CFunctionMemory.CleanMemory()

    @staticmethod
    def FillPolygonsCV2(mask:np.array, fill_value:int=255):

        mask_label, label_num = CFunctionImageOperator.GetImageMask2Label(mask)
        np_seed_point = CFunctionToolImageC.GetLabelBoundarySeedPoint(mask_label, label_num)
        list_boundary = CFunctionRouteMove.GetLabelBoundary(mask_label, label_num, np_seed_point)
        del mask_label
        CFunctionMemory.CleanMemory()

        polygons_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(polygons_mask, np.array[list_boundary], 255)
        CFunctionToolImageC.ReplaceMatrixValue(mask, polygons_mask)

class CFunctionImageColorOperator:

    @staticmethod
    def ChannelMatrix2RGBsImage(channel_matrix:np.array):

        row_length, col_length, channel_num = channel_matrix.shape
        if channel_num > len(LIST_COLOR_CHANNEL):
            CLogging.LogError("The number of channels exceeds the number of supported colors. ")
        rgb_image = np.zeros((row_length, col_length, 3),dtype=np.uint8)
        for i in range(channel_num):
            rgb_image[:, :, 0][channel_matrix[:, :, i] > 0] = LIST_COLOR_CHANNEL[i][0]
            rgb_image[:, :, 1][channel_matrix[:, :, i] > 0] = LIST_COLOR_CHANNEL[i][1]
            rgb_image[:, :, 2][channel_matrix[:, :, i] > 0] = LIST_COLOR_CHANNEL[i][2]
        return rgb_image

    @staticmethod
    def GetRGBHex(r:int, g:int, b:int):
        hex_string = '#{:02X}{:02X}{:02X}'.format(r, g, b)
        return hex_string

    @staticmethod
    def GetRandomRGBHex():

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return CFunctionImageColorOperator.GetRGBHex(r, g, b)