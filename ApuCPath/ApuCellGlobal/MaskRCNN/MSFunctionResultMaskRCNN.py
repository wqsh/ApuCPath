from ApuCellGlobal.MSApuCellExternalPackge import *
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralFunction.MSFunctionImageMerge import CFunctionImageSplitMerge, CFunctionMaskMerge
from ApuCellGlobal.GeneralSystem.MSLogging import CLogProgressPercent
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator


class CFunctionMaskRCNNResultOperator:

    @staticmethod
    def GetWSINonOverlapPredictMask(folder_predict_pkl:str, class_id_num:int, split_symbol:str = "_"):
        data_matrix = CFunctionMaskRCNNResultOperator.GetWSIPredictChannelMatrix(folder_predict_pkl, class_id_num, split_symbol)
        delete_matrix = CFunctionMaskMerge.GetNonOverlapMask(data_matrix)
        delete_matrix[delete_matrix > 0] = 1
        return delete_matrix

    @staticmethod
    def GetWSIPredictChannelMatrix(folder_predict_pkl:str, class_id_num:int, split_symbol:str = "_"):
        image_merge_info = CFunctionImageSplitMerge.GetMergeInfo(folder_predict_pkl)
        list_file = os.listdir(folder_predict_pkl)
        predict_result = CFunctionFileIO.ReadPKL(os.path.join(folder_predict_pkl, list_file[0]))
        y_size = predict_result["masks"].shape[0]
        x_size = predict_result["masks"].shape[1]
        y_full_size, x_full_size = predict_result["masks"].shape[0] * image_merge_info.y_idx_length, x_size * image_merge_info.x_idx_length
        data_matrix = np.zeros((y_full_size, x_full_size, class_id_num), np.uint8)
        progress_percent = CLogProgressPercent(0, len(list_file) - 1, 0.02, "Read predict results %s ... "%folder_predict_pkl)
        for file_idx, file_name in enumerate(list_file):
            progress_percent.ShowProgressPercent(file_idx)
            predict_result = CFunctionFileIO.ReadPKL(os.path.join(folder_predict_pkl, file_name))
            y_idx = int(file_name.split('.')[0].split(split_symbol)[-2])
            x_idx = int(file_name.split('.')[0].split(split_symbol)[-1])
            for class_id_idx, class_id in enumerate(predict_result["class_ids"]):
                detect_mask = predict_result["masks"][:, :, class_id_idx].astype(np.uint8)
                detect_mask[detect_mask > 0] = 1
                data_matrix[y_idx * y_size: (y_idx + 1) * y_size, x_idx * x_size: (x_idx + 1) * x_size, class_id] = detect_mask
        return data_matrix[:, :, 1:]