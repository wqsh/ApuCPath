from ConfigExe.WSISegModel.MSDataPack import CConfigWSISegModel, EnumTypeModelInit, EnumTypeModelType, EnumTypeNegSample
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

import os

class CFunctionConfig:

    @staticmethod
    def Config2File(config:CConfigWSISegModel, path:str, name:str):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, name), 'w') as f:
            f.write("A1_TYPE_FLOW" + '=' + str(config.A1_TYPE_FLOW) + '\n')
            f.write("A2_PATH_DATASET_NORMAL" + '=' + str(config.A2_PATH_DATASET_NORMAL) + '\n')
            f.write("A3_PATH_DATASET_TRAIN" + '=' + str(config.A3_PATH_DATASET_TRAIN) + '\n')
            f.write("A4_PATH_DATASET_VALID" + '=' + str(config.A4_PATH_DATASET_VALID) + '\n')
            f.write("A5_PATH_DATASET_PREDICT" + '=' + str(config.A5_PATH_DATASET_PREDICT) + '\n')
            f.write("A6_FOLDER_OUTPUT" + '=' + str(config.A6_FOLDER_OUTPUT) + '\n')
            f.write("A7_SKIP_CHECK_MASK" + '=' + str(int(config.A7_SKIP_CHECK_MASK)) + '\n')
            f.write("A8_PREDICT_SKIP_EXIST" + '=' + str(int(config.A8_PREDICT_SKIP_EXIST)) + '\n')
            f.write('\n')

            f.write("B1_IMAGE_DIM" + '=' + str(config.B1_IMAGE_DIM) + '\n')
            # f.write("B2_IMAGE_HEIGHT" + '=' + str(config.B2_IMAGE_HEIGHT) + '\n')
            f.write("B3_CLASSES_NUM" + '=' + str(config.B3_CLASSES_NUM) + '\n')
            f.write("B4_BATCH_SIZE" + '=' + str(config.B4_BATCH_SIZE) + '\n')
            f.write("B5_EPOCH_NUM" + '=' + str(config.B5_EPOCH_NUM) + '\n')
            f.write("B6_LEARNING_RATE" + '=' + "%.8f"%config.B6_LEARNING_RATE + '\n')
            f.write("B7_TRAIN_TYPE" + '=' + str(config.B7_TRAIN_TYPE) + '\n')
            f.write("B8_K_FOLD" + '=' + str(config.B8_K_FOLD) + '\n')
            f.write("B9_VALID_PERCENT" + '=' + str(config.B9_VALID_PERCENT) + '\n')

            f.write('\n')
            f.write("C1_MODEL_INIT" + '=' + str(config.C1_MODEL_INIT.value) + '\n')
            f.write("C2_MODEL_PATH" + '=' + str(config.C2_MODEL_PATH) + '\n')
            f.write("C3_MODEL_TYPE" + '=' + str(config.C3_MODEL_TYPE.value) + '\n')
            f.write("C4_MODEL_ADD_NEGATIVE" + '=' + str(int(config.C4_MODEL_ADD_NEGATIVE)) + '\n')
            f.write("C5_MODEL_NEGATIVE_SAMPLE" + '=' + str(config.C5_MODEL_NEGATIVE_SAMPLE.value) + '\n')
            f.write("C5_MODEL_NEGATIVE_SAMPLE_CLUSTER" + '=' + str(int(config.C5_MODEL_NEGATIVE_SAMPLE_CLUSTER)) + '\n')
            f.write("C6_BALANCE_SAMPLE" + '=' + str(int(config.C6_BALANCE_SAMPLE)) + '\n')

            f.write('\n')
            f.write("D1_1_FILTER_MASK_SCORE" + "=" + "%.4f" % config.D1_1_FILTER_MASK_SCORE + "\n")
            f.write("D2_EXPORT_PREDICT_HEATMAP" + "=" + str(int(config.D2_EXPORT_PREDICT_HEATMAP)) + "\n")
            f.write("D3_MIN_AREA" + "=" + str(config.D3_MIN_AREA) + "\n")
            f.write("D4_MERGE_SCORE" + "=" + "%.4f"%config.D4_MERGE_SCORE + "\n")
            f.write("D5_MERGE_OVERLAP" + "=" + "%d"%int(config.D5_MERGE_OVERLAP) + "\n")
            f.write("D6_EXPORT_PAD_MASK" + "=" + "%d"%int(config.D6_EXPORT_PAD_MASK) + "\n")

            f.write('\n')
            f.write("E1_TRAIN_IMAGE_LIST_FILE" + "=" + str(config.E1_TRAIN_IMAGE_LIST_FILE) + "\n")
            f.write("E2_VALID_IMAGE_LIST_FILE" + "=" + str(config.E2_VALID_IMAGE_LIST_FILE) + "\n")
            f.write("E3_PREDICT_IMAGE_LIST_FILE" + "=" + str(config.E3_PREDICT_IMAGE_LIST_FILE) + "\n")

            f.write('\n')
            f.write("F1_REPLY_TRAIN" + "=" + str(config.F1_REPLY_TRAIN) + "\n")
            f.write("F2_FOLDER_REPLY_TRAIN" + "=" + str(config.F2_FOLDER_REPLY_TRAIN) + '\n')
            f.write("F3_PATH_REPLY_TRAIN_MODEL" + "=" + str(config.F3_PATH_REPLY_TRAIN_MODEL) + '\n')

            f.write('\n')
            f.write("G1_BACKBONE_1_RPN_PRE_NMS_TOP_N_TRAIN" + "=" + str(config.G1_BACKBONE_1_RPN_PRE_NMS_TOP_N_TRAIN) + '\n')
            f.write("G1_BACKBONE_2_RPN_POST_NMS_TOP_N_TRAIN" + "=" + str(config.G1_BACKBONE_2_RPN_POST_NMS_TOP_N_TRAIN) + '\n')
            f.write("G1_BACKBONE_3_RPN_FG_IOU_THRESH" + "=" + str(config.G1_BACKBONE_3_RPN_FG_IOU_THRESH) + '\n')
            f.write("G1_BACKBONE_4_RPN_BG_IOU_THRESH" + "=" + str(config.G1_BACKBONE_4_RPN_BG_IOU_THRESH) + '\n')

            f.write("G2_BACKBONE_1_RPN_NMS_THRESH" + "=" + str(config.G2_BACKBONE_1_RPN_NMS_THRESH) + '\n')
            f.write("G2_BACKBONE_2_MASK_ROI_POOL_SIZE" + "=" + str(config.G2_BACKBONE_2_MASK_ROI_POOL_SIZE) + '\n')

            f.write("G3_BACKBONE_1_RPN_PRE_NMS_TOP_N_TEST" + "=" + str(config.G3_BACKBONE_1_RPN_PRE_NMS_TOP_N_TEST) + '\n')
            f.write("G3_BACKBONE_2_RPN_POST_NMS_TOP_N_TEST" + "=" + str(config.G3_BACKBONE_2_RPN_POST_NMS_TOP_N_TEST) + '\n')
            f.write("G3_BACKBONE_3_BOX_SCORE_THRESH" + "=" + str(config.G3_BACKBONE_3_BOX_SCORE_THRESH) + '\n')
            f.write("G3_BACKBONE_4_BOX_NMS_THRESH" + "=" + str(config.G3_BACKBONE_4_BOX_NMS_THRESH) + '\n')
            f.write("G3_BACKBONE_5_BOX_DETECTIONS_PER_IMG" + "=" + str(config.G3_BACKBONE_5_BOX_DETECTIONS_PER_IMG) + '\n')

            f.write("G4_PATH_UNI_CHECKPOINT" + "=" + str(config.G4_PATH_UNI_CHECKPOINT) + '\n')

            f.write('\n')
            f.write("Z1_USE_GPU" + '=' + str(int(config.Z1_USE_GPU)) + '\n')

    @staticmethod
    def File2Config(path_config:str, config:CConfigWSISegModel):

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
                if "A1_TYPE_FLOW" == name:
                    config.A1_TYPE_FLOW = int(value)
                elif "A2_PATH_DATASET_NORMAL" == name:
                    config.A2_PATH_DATASET_NORMAL = value
                elif "A3_PATH_DATASET_TRAIN" == name:
                    config.A3_PATH_DATASET_TRAIN = value
                elif "A4_PATH_DATASET_VALID" == name:
                    config.A4_PATH_DATASET_VALID = value
                elif "A5_PATH_DATASET_PREDICT" == name:
                    config.A5_PATH_DATASET_PREDICT = value
                elif "A6_FOLDER_OUTPUT" == name:
                    if not os.path.exists(value):
                        os.makedirs(value)
                    config.A6_FOLDER_OUTPUT = value
                elif "A7_SKIP_CHECK_MASK" == name:
                    config.A7_SKIP_CHECK_MASK = bool(int(value))
                elif "A8_PREDICT_SKIP_EXIST" == name:
                    config.A8_PREDICT_SKIP_EXIST = bool(int(value))

                elif "B1_IMAGE_DIM" == name:
                    config.B1_IMAGE_DIM = int(value)
                # elif "B2_IMAGE_HEIGHT" == name:
                #     config.B2_IMAGE_HEIGHT = int(value)
                elif "B3_CLASSES_NUM" == name:
                    config.B3_CLASSES_NUM = int(value)
                elif "B4_BATCH_SIZE" == name:
                    config.B4_BATCH_SIZE = int(value)
                elif "B5_EPOCH_NUM" == name:
                    config.B5_EPOCH_NUM = int(value)
                elif "B6_LEARNING_RATE" == name:
                    config.B6_LEARNING_RATE = float(value)
                elif "B7_TRAIN_TYPE" == name:
                    config.B7_TRAIN_TYPE = int(value)
                elif "B8_K_FOLD" == name:
                    config.B8_K_FOLD = int(value)
                elif "B9_VALID_PERCENT" == name:
                    config.B9_VALID_PERCENT = float(value)

                elif "C1_MODEL_INIT" == name:
                    config.C1_MODEL_INIT = EnumTypeModelInit(int(value))
                elif "C2_MODEL_PATH" == name:
                    config.C2_MODEL_PATH = value
                elif "C3_MODEL_TYPE" == name:
                    config.C3_MODEL_TYPE = EnumTypeModelType(int(value))
                elif "C4_MODEL_ADD_NEGATIVE" == name:
                    config.C4_MODEL_ADD_NEGATIVE = bool(int(value))
                elif "C5_MODEL_NEGATIVE_SAMPLE" == name:
                    config.C5_MODEL_NEGATIVE_SAMPLE = EnumTypeNegSample(int(value))
                elif "C5_MODEL_NEGATIVE_SAMPLE_CLUSTER" == name:
                    config.C5_MODEL_NEGATIVE_SAMPLE_CLUSTER = int(value)
                elif "C6_BALANCE_SAMPLE" == name:
                    config.C6_BALANCE_SAMPLE = bool(int(value))

                elif "D1_1_FILTER_MASK_SCORE" == name:
                    config.D1_1_FILTER_MASK_SCORE = float(value)
                elif "D2_EXPORT_PREDICT_HEATMAP" == name:
                    config.D2_EXPORT_PREDICT_HEATMAP = bool(int(value))
                elif "D3_MIN_AREA" == name:
                    config.D3_MIN_AREA = int(value)
                elif "D4_MERGE_SCORE" == name:
                    config.D4_MERGE_SCORE = float(value)
                elif "D5_MERGE_OVERLAP" == name:
                    config.D5_MERGE_OVERLAP = bool(int(value))
                elif "D6_EXPORT_PAD_MASK" == name:
                    config.D6_EXPORT_PAD_MASK = bool(int(value))

                elif "E1_TRAIN_IMAGE_LIST_FILE" == name:
                    config.E1_TRAIN_IMAGE_LIST_FILE = value
                elif "E2_VALID_IMAGE_LIST_FILE" == name:
                    config.E2_VALID_IMAGE_LIST_FILE = value
                elif "E3_PREDICT_IMAGE_LIST_FILE" == name:
                    config.E3_PREDICT_IMAGE_LIST_FILE = value

                elif "F1_REPLY_TRAIN" == name:
                    config.F1_REPLY_TRAIN = int(value)
                elif "F2_FOLDER_REPLY_TRAIN" == name:
                    config.F2_FOLDER_REPLY_TRAIN = value
                elif "F3_PATH_REPLY_TRAIN_MODEL" == name:
                    config.F3_PATH_REPLY_TRAIN_MODEL = value

                elif "G1_BACKBONE_1_RPN_PRE_NMS_TOP_N_TRAIN" == name:
                    config.G1_BACKBONE_1_RPN_PRE_NMS_TOP_N_TRAIN = int(value)
                elif "G1_BACKBONE_2_RPN_POST_NMS_TOP_N_TRAIN" == name:
                    config.G1_BACKBONE_2_RPN_POST_NMS_TOP_N_TRAIN = int(value)
                elif "G1_BACKBONE_3_RPN_FG_IOU_THRESH" == name:
                    config.G1_BACKBONE_3_RPN_FG_IOU_THRESH = float(value)
                elif "G1_BACKBONE_4_RPN_BG_IOU_THRESH" == name:
                    config.G1_BACKBONE_4_RPN_BG_IOU_THRESH = float(value)

                elif "G2_BACKBONE_1_RPN_NMS_THRESH" == name:
                    config.G2_BACKBONE_1_RPN_NMS_THRESH = float(value)
                elif "G2_BACKBONE_2_MASK_ROI_POOL_SIZE" == name:
                    config.G2_BACKBONE_2_MASK_ROI_POOL_SIZE = int(value)

                elif "G3_BACKBONE_1_RPN_PRE_NMS_TOP_N_TEST" == name:
                    config.G3_BACKBONE_1_RPN_PRE_NMS_TOP_N_TEST = int(value)
                elif "G3_BACKBONE_2_RPN_POST_NMS_TOP_N_TEST" == name:
                    config.G3_BACKBONE_2_RPN_POST_NMS_TOP_N_TEST = int(value)
                elif "G3_BACKBONE_3_BOX_SCORE_THRESH" == name:
                    config.G3_BACKBONE_3_BOX_SCORE_THRESH = float(value)
                elif "G3_BACKBONE_4_BOX_NMS_THRESH" == name:
                    config.G3_BACKBONE_4_BOX_NMS_THRESH = float(value)
                elif "G3_BACKBONE_5_BOX_DETECTIONS_PER_IMG" == name:
                    config.G3_BACKBONE_5_BOX_DETECTIONS_PER_IMG = int(value)

                elif "G4_PATH_UNI_CHECKPOINT" == name:
                    config.G4_PATH_UNI_CHECKPOINT = value

                elif "Z1_USE_GPU" == name:
                    config.Z1_USE_GPU = int(value)

            except Exception as err:
                CLogging.LogError("Occurred in " + path_config + "." + str(err))