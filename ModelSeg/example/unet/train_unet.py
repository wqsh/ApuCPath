import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import os

from Model.unet.dataset import SyntheticShapesDataset
from Model.unet.unet_model import ModelUnet
from Model.unet.config import CConfigParameterUnet

def train(folder_export:str=None):

    parameter_unet = CConfigParameterUnet()
    if folder_export is None:
        parameter_unet.OUTPUT_ROOT = os.path.join(os.getcwd(), "unet")
    else:
        parameter_unet.OUTPUT_ROOT = folder_export
    # 数据集
    train_dataset = SyntheticShapesDataset(num_samples=201, img_size=256)
    valid_dataset = SyntheticShapesDataset(num_samples=50, img_size=256)
    parameter_unet.NUM_CLASSES = len(train_dataset.class_names)

    model = ModelUnet("train", parameter_unet)
    model.Train(train_dataset, valid_dataset)

if __name__ == "__main__":

    train(r"/media/wqs/SOFT/weiqiushi/MaskRCNN/TorchvisionMaskRCNN/ModelTrain/unet")