import numpy as np
from ApuCellGlobal.MSApuCellExternalPackge import *
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC

class CFunctionMaskMerge:

    @staticmethod
    def GetOverlapSeedPoint(mask1:np.array, mask2:np.array):

        mask1_tmp = copy.deepcopy(mask1)
        mask2_tmp = copy.deepcopy(mask2)

        mask1_label = CFunctionImageOperator.GetImageMaskLabels(mask1_tmp)
        mask2_label = CFunctionImageOperator.GetImageMaskLabels(mask2_tmp)

        mask1_tmp[mask1_tmp > 0] = 1
        mask2_tmp[mask2_tmp > 0] = 1

        merge_mask = np.zeros(mask1.shape, dtype=np.uint8)
        merge_mask += mask1_tmp
        merge_mask += mask2_tmp

        merge_mask[merge_mask <= 1] = 0
        merge_mask[merge_mask > 1] = 1

        merge_mask_label = CFunctionImageOperator.GetImageMaskLabels(merge_mask)
        label_num = np.max(merge_mask_label)
        list_seed_point = CFunctionToolImageC.GetLabelBoundarySeedPoint(merge_mask_label, label_num)

        mask1_remove_label_idx_list = []
        mask2_remove_label_idx_list = []
        for seed_point_idx in range(1, label_num + 1):
            seed_point_x = list_seed_point[0][seed_point_idx * 2 + 1]
            seed_point_y = list_seed_point[0][seed_point_idx * 2]

            mask1_label_idx = mask1_label[seed_point_y, seed_point_x]
            mask2_label_idx = mask2_label[seed_point_y, seed_point_x]

            mask1_remove_label_idx_list.append(mask1_label_idx)
            mask2_remove_label_idx_list.append(mask2_label_idx)

        mask1_remove_seed_point_list = CFunctionToolImageC.GetLabelSeedPoint(mask1_label, mask1_remove_label_idx_list)
        mask2_remove_seed_point_list = CFunctionToolImageC.GetLabelSeedPoint(mask2_label, mask2_remove_label_idx_list)

        return (mask1_remove_seed_point_list, mask2_remove_seed_point_list)

    @staticmethod
    def RemoveOverlapShape(mask1:np.array, mask2:np.array):

        remove_seed_point_list = CFunctionMaskMerge.GetOverlapSeedPoint(mask1, mask2)
        mask1_remove_seed_point_list = remove_seed_point_list[0]
        mask2_remove_seed_point_list = remove_seed_point_list[1]

        for i in range(len(mask1_remove_seed_point_list)):
            mask1_remove_seed_point_info = mask1_remove_seed_point_list[i]
            mask2_remove_seed_point_info = mask2_remove_seed_point_list[i]
            CFunctionToolImageC.RemoveShape(mask1, mask1_remove_seed_point_info.seed_x, mask1_remove_seed_point_info.seed_y)
            CFunctionToolImageC.RemoveShape(mask2, mask2_remove_seed_point_info.seed_x, mask2_remove_seed_point_info.seed_y)

    @staticmethod
    def RemoveOverlapShapeOnlySecond(mask1: np.array, mask2: np.array):

        remove_seed_point_list = CFunctionMaskMerge.GetOverlapSeedPoint(mask1, mask2)
        mask1_remove_seed_point_list = remove_seed_point_list[0]
        mask2_remove_seed_point_list = remove_seed_point_list[1]

        for i in range(len(mask1_remove_seed_point_list)):
            mask2_remove_seed_point_info = mask2_remove_seed_point_list[i]
            CFunctionToolImageC.RemoveShape(mask2, mask2_remove_seed_point_info.seed_x, mask2_remove_seed_point_info.seed_y)