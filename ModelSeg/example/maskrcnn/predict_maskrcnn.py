import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.maskrcnn.maskrcnn_model import ModelMaskRCNN, get_checkpoint_num_class
from Model.maskrcnn.config import CConfigParameterMaskRCNN

import os
import numpy as np
import matplotlib.pyplot as plt
import copy

def ExportPredictMask(folder_export, images, images_name, predictions, list_color, num_classes):

    for idx, prediction in enumerate(predictions):
        predict_mask = np.zeros((images[idx].shape[0], images[idx].shape[1], num_classes - 1), dtype=np.float32)
        if prediction['masks'].shape[0] > 0:
            masks = prediction['masks'].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()
            for i, label in enumerate(labels):
                mask = copy.deepcopy(masks[i, 0, :, :])
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = scores[i]
                predict_mask[:, :, label - 1] += mask
        final_mask = np.zeros((images[idx].shape[0], images[idx].shape[1], 3), dtype=np.uint16)
        for class_idx in range(1, num_classes):
            final_mask[:, :, 0][predict_mask[:, :, class_idx - 1] > 0] += list_color[class_idx - 1][0]
            final_mask[:, :, 1][predict_mask[:, :, class_idx - 1] > 0] += list_color[class_idx - 1][1]
            final_mask[:, :, 2][predict_mask[:, :, class_idx - 1] > 0] += list_color[class_idx - 1][2]
        final_mask[final_mask > 255] = 255
        plt.imsave(os.path.join(folder_export, images_name[idx]), final_mask.astype(np.uint8))

def Predict(folder_image, path_checkpoint:str, folder_export:str, score_thresh=0.5, iou_thresh=0.5):

    parameter_mask_rcnn = CConfigParameterMaskRCNN()
    num_classes = get_checkpoint_num_class(path_checkpoint)
    parameter_mask_rcnn.NUM_CLASSES = num_classes

    if not os.path.exists(folder_export):
        os.makedirs(folder_export)

    list_color = [ np.array(np.random.rand(3,) * 255).astype(np.uint8) for i in range(parameter_mask_rcnn.NUM_CLASSES)]

    model = ModelMaskRCNN("inference", parameter_mask_rcnn, path_checkpoint)
    # model = ModelMaskRCNNBGClassifier("inference", parameter_mask_rcnn, path_checkpoint)

    list_image = os.listdir(folder_image)
    images = []
    images_name = []
    for image_name in list_image:
        if len(images) == parameter_mask_rcnn.BATCH_SIZE:
            predictions = model.Predict(images)
            ExportPredictMask(folder_export, images, images_name, predictions, list_color, parameter_mask_rcnn.NUM_CLASSES)
            images = []
            images_name = []
        images.append(plt.imread(os.path.join(folder_image, image_name)))
        images_name.append(image_name)
    if len(images) > 0:
        predictions = model.Predict(images)
        ExportPredictMask(folder_export, images, images_name, predictions, list_color, parameter_mask_rcnn.NUM_CLASSES)

if __name__ == "__main__":
    
    folder_image = r"/mnt/sdb/weiqiushi/MaskRCNN/TorchvisionMaskRCNN/TestImage/20260507223115"
    path_checkpoint = r"/ModelTrain/maskrcnn/20260520014636/maskrcnn_epoch_12_best.pth"
    #     # path_checkpoint = r"/media/wqs/SOFT/weiqiushi/MaskRCN/TorchvisionMaskRCNN/example/maskrcnn_edge/20260515225950/maskrcnn_epoch_11_best.pth"
    folder_export = folder_image + "_predict_maskrcnn"

    Predict(folder_image, path_checkpoint, folder_export)