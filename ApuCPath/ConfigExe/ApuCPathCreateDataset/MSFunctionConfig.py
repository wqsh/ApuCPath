from ConfigExe.ApuCPathCreateDataset.MSDataPack import CConfigCreateDataset, EnumTypeExportPad
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import EnumTypeExportImage, EnumDatasetType

import os

class CFunctionConfig:

    @staticmethod
    def Config2File(config:CConfigCreateDataset, path:str, name:str):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, name), 'w') as f:
            f.write("A1_TYPE_DATASET" + '=' + str(config.A1_TYPE_DATASET.value) + '\n')
            f.write("A2_LIST_IMAGE_PATH" + '=' + str(config.A2_LIST_IMAGE_PATH) + "\n")
            f.write("A3_LIST_MASK_PATH" + '=' + str(config.A3_LIST_MASK_PATH) + '\n')
            f.write("A4_FOLDER_EXPORT" + '=' + str(config.A4_FOLDER_EXPORT) + '\n')
            f.write("A5_NAME_DATASET" + "=" + str(config.A5_NAME_DATASET) + "\n")
            f.write('\n')

            f.write("B1_IMAGE_DIM" + '=' + str(config.B1_IMAGE_DIM) + '\n')
            f.write("B2_MASK_RATIO_MIN" + '=' + "%.7f"%(config.B2_MASK_RATIO_MIN) + '\n')
            f.write("B3_MASK_RATIO_MAX" + '=' + "%.7f"%(config.B3_MASK_RATIO_MAX) + '\n')
            f.write("B4_EXPORT_TYPE" + '=' + "%d"%(config.B4_EXPORT_TYPE.value) + '\n')
            f.write("B5_EXPORT_PAD_TYPE" + '=' + str(config.B5_EXPORT_PAD_TYPE.value) + '\n')
            f.write('\n')

    @staticmethod
    def File2Config(path_config:str, config:CConfigCreateDataset):

        with open(path_config, 'r', encoding="utf-8") as f:
            buffer = f.readlines()

        for line in buffer:
            line = line.strip()
            if line.find('=') < 0:
                continue
            if line.find('#') >= 0:
                line = line[:line.find('#')].strip()
            tmp = line.split('=')
            if len(tmp) < 2:
                continue
            name = tmp[0]
            value = tmp[1]
            print(name, ":", value)
            try:
                if "A1_TYPE_DATASET" == name:
                    config.A1_TYPE_DATASET = EnumDatasetType(int(value))
                elif "A2_LIST_IMAGE_PATH" == name:
                    config.A2_LIST_IMAGE_PATH = value
                elif "A3_LIST_MASK_PATH" == name:
                    config.A3_LIST_MASK_PATH = value
                elif "A4_FOLDER_EXPORT" == name:
                    if not os.path.exists(value):
                        os.makedirs(value)
                    config.A4_FOLDER_EXPORT = value
                elif "A5_NAME_DATASET" == name:
                    config.A5_NAME_DATASET = value

                elif "B1_IMAGE_DIM" == name:
                    config.B1_IMAGE_DIM = int(value)
                elif "B2_MASK_RATIO_MIN" == name:
                    config.B2_MASK_RATIO_MIN = float(value)
                elif "B3_MASK_RATIO_MAX" == name:
                    config.B3_MASK_RATIO_MAX = float(value)
                elif "B4_EXPORT_TYPE" == name:
                    value_int = int(value)
                    if value_int == EnumTypeExportImage.Pkl.value:
                        config.B4_EXPORT_TYPE = EnumTypeExportImage.Pkl
                    elif value_int == EnumTypeExportImage.Tif.value:
                        config.B4_EXPORT_TYPE = EnumTypeExportImage.Tif
                    else:
                        config.B4_EXPORT_TYPE = EnumTypeExportImage.Pkl
                elif "B5_EXPORT_PAD_TYPE" == name:
                    value_int = int(value)
                    if value_int == EnumTypeExportPad.All.value:
                        config.B5_EXPORT_PAD_TYPE = EnumTypeExportPad.All
                    elif value_int == EnumTypeExportPad.Org.value:
                        config.B5_EXPORT_PAD_TYPE = EnumTypeExportPad.Org
                    else:
                        config.B5_EXPORT_PAD_TYPE = EnumTypeExportPad.Org

            except Exception as err:
                CLogging.LogError("Occurred in " + path_config + "." + str(err))