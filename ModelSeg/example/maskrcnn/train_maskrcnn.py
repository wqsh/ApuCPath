import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.maskrcnn.dataset import SyntheticShapesDataset
from Model.maskrcnn.config import CConfigParameterMaskRCNN
from Model.maskrcnn.maskrcnn_model import ModelMaskRCNN

import os

def train(folder_export:str=None):

    parameter_mask_rcnn = CConfigParameterMaskRCNN()
    if folder_export is None:
        parameter_mask_rcnn.OUTPUT_ROOT = os.path.join(os.getcwd(), "maskrcnn")
    else:
        parameter_mask_rcnn.OUTPUT_ROOT = folder_export
    # 数据集
    train_dataset = SyntheticShapesDataset(num_samples=201, img_size=256)
    valid_dataset = SyntheticShapesDataset(num_samples=50, img_size=256)
    parameter_mask_rcnn.NUM_CLASSES = len(train_dataset.class_names)

    model = ModelMaskRCNN("train", parameter_mask_rcnn)
    # model = ModelMaskRCNNBGClassifier("train", parameter_mask_rcnn)
    model.Train(train_dataset, valid_dataset)

    # model = ModelMaskRCNNWithEdge("train", parameter_mask_rcnn)
    # model.Train(train_dataset, valid_dataset)

if __name__ == '__main__':

    train(r"/media/wqs/SOFT/weiqiushi/MaskRCNN/TorchvisionMaskRCNN/ModelTrain/maskrcnn")