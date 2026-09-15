from ApuCellGlobal.MSApuCellGlobalImport import *
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory
from ApuCellGlobal.GeneralSystem.MSSystemEnum import LIST_DATA_FILE_TYPE_SUFFIX, EnumDataFileType

from ApuCellGlobal.WSIProcess.MSDataWSIDataset import CDataWSIDatasetBrevity, EnumDatasetType

class CFunctionGenerateWSIDataset():

    @staticmethod
    def GenerateDataset(image_path_list:list, image_mask_path_list:list, out_dataset_path:str, row_size:int=512, col_size:int=512, dataset_type:EnumDatasetType=EnumDatasetType.Predict):
        for idx, image_path in enumerate(image_path_list):
            CLogging.LogToUser("Generate dataset : " + image_path)
            CFunctionGenerateWSIDataset.GenerateDatasetOneImage(image_path, image_mask_path_list[idx], out_dataset_path, row_size, col_size, dataset_type)
            CFunctionMemory.CleanMemory()

    @staticmethod
    def GenerateDatasetOneImage(image_path:str, img_mask_path:str, out_dataset_path:str, row_size:int=512, col_size:int=512, dataset_type:EnumDatasetType=EnumDatasetType.Predict):
        CFunctionMemory.GetCurMemoryInfo()

        CLogging.LogToUser("Start generate dataset index. ")
        wsi_dataset = CDataWSIDatasetBrevity(os.path.basename(image_path).split('.')[0], image_path, img_mask_path, row_size, col_size, dataset_type)
        out_name = os.path.basename(image_path).split('.')[0]
        out_file_path = os.path.join(out_dataset_path, out_name + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value])
        CFunctionFileIO.WritePKL(out_dataset_path, out_name + LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value], wsi_dataset)
        del wsi_dataset
        CLogging.LogToUser("Finish generate dataset index. ")
        return out_file_path