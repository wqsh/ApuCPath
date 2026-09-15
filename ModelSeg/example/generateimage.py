import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

from Model.maskrcnn.dataset import SyntheticShapesDataset, tensor_to_img_predict

if __name__ == "__main__":

    path_img_export = os.path.join(os.getcwd(), "TestImage")
    num_samples = 4

    path_data = os.path.join(path_img_export, CLogging.GetLocTimeStrForName())
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    # 数据集
    dataset = SyntheticShapesDataset(num_samples, img_size=256)
    for image_idx in range(num_samples):
        image_tensor, target = dataset[image_idx]
        image = tensor_to_img_predict(image_tensor)
        CFunctionFileIO.WriteImage(path_data, "%d.png"%(image_idx + 1), image)
        image_idx += 1