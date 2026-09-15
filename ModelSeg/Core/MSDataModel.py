from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator

from ApuCellGlobal.GeneralSystem.MSSystemEnum import LIST_DATA_FILE_TYPE, EnumDataFileType, LIST_VISUAL_FILE_TYPE, EnumVisualFileType

import math
from enum import Enum, unique
import os
import numpy as np
import skimage.io
import torch
from scipy import ndimage
from torchvision import transforms as T
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from Core.MSDataPack import EnumTypeModelType

@unique
class EnumModelDatasetType(Enum):

    Normal = 0
    Train = 1
    Valid = 2
    Test = 3
    Predict = 4

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = []
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        CLogging.LogWarning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

class CDataModelDatasetInfo:


    def __init__(self, dataset_type:EnumModelDatasetType=EnumModelDatasetType.Normal):

        self.__path_image_list = []
        self.__path_mask_list = []
        self.__dataset_type:EnumModelDatasetType = dataset_type

    def AddPath(self, path_image:str, path_mask:str=None):

        if not os.path.exists(path_image):
            CLogging.LogError(path_image)

        if self.__dataset_type == EnumModelDatasetType.Predict:
            pass
        else:
            if not os.path.exists(path_mask):
                CLogging.LogError(path_mask)

        self.__path_image_list.append(path_image)
        self.__path_mask_list.append(path_mask)

    @property
    def dataset_type(self):
        return self.__dataset_type

    def GetImageNum(self):
        return len(self.__path_image_list)

    def GetImagePath(self, idx:int):

        return self.__path_image_list[idx]

    def GetMaskPath(self, idx:int):

        return self.__path_mask_list[idx]

class CDataModelDataset(Dataset):

    def __init__(self, image_dim:int, class_name_list:list, wsi_name:str, pad_name:str=None, model_type:EnumTypeModelType=EnumTypeModelType.MaskRCNN):
        super(CDataModelDataset, self).__init__()
        self.image_dim = image_dim
        self.wsi_name:str = wsi_name
        self.pad_name:str = pad_name
        if pad_name is None:
            self.dataset_name = wsi_name
        else:
            self.dataset_name = wsi_name + "-" + pad_name

        for class_idx, class_name in enumerate(class_name_list):
            self.add_class(wsi_name, class_idx, class_name)
        self.model_type:EnumTypeModelType = model_type

        # 图像的数值变换（归一化 + ToTensor）
        self.image_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def SetModleType(self, model_type:EnumTypeModelType=EnumTypeModelType.MaskRCNN):

        self.model_type: EnumTypeModelType = model_type

    def CreateTrainDatasets(self, path_image_list:list, path_mask_list:list, source:str=None):
        if source is None:
            source = self.dataset_name
        start_idx = len(self.image_info)
        for img_idx, path_img in enumerate(path_image_list):
            if path_mask_list is None:
                self.add_image(source, image_id=img_idx + start_idx, path=path_img, width=self.image_dim, height=self.image_dim, classes=[self.class_info[0]["name"]])
            else:
                path_mask = path_mask_list[img_idx]
                if path_mask is None:
                    class_names = self.class_info[0]["name"]
                else:
                    if path_mask.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIFF.value]) or path_mask.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIF.value]):
                        mask = CFunctionFileIO.ReadImage(path_mask)
                    elif path_mask.endswith(LIST_DATA_FILE_TYPE[EnumDataFileType.PKL.value]):
                        mask = CFunctionFileIO.ReadPKL(path_mask)
                    else:
                        CLogging.LogError("Unsupport mask type! ")
                    class_names = []
                    for class_idx in range(mask.shape[2]):
                        if np.sum(mask[:, :, class_idx]) > 0:
                            class_names.append(self.class_info[class_idx + 1]["name"])
                self.add_image(source, image_id=img_idx + start_idx, path=path_img, width=self.image_dim, height=self.image_dim, mask=path_mask, classes=class_names)

    def CreatePredictDatasets(self, path_image_list:list, source:str=None):
        if source is None:
            source = self.dataset_name
        for img_idx, path_img in enumerate(path_image_list):
            self.add_image(source, image_id=img_idx, path=path_img, width=self.image_dim, height=self.image_dim)

    def GetImageAndMaskPathList(self):
        list_path_image_list = []
        list_path_mask_list = []
        for id in self.image_ids:
            list_path_image_list.append(self.image_info[id]["path"])
            if "mask" not in self.image_info[id]:
                list_path_mask_list.append(None)
            else:
                list_path_mask_list.append(self.image_info[id]["mask"])
        return [list_path_image_list, list_path_mask_list]

    def load_image(self, image_id):
        # 根据id返回图片的数组
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        if info['path'].split('.')[-1] == LIST_DATA_FILE_TYPE[EnumDataFileType.PKL.value]:
            image = CFunctionFileIO.ReadPKL(info['path'])
        elif info['path'].split('.')[-1] == LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIF.value]:
            image = CFunctionFileIO.ReadImage(info['path'])
        else:
            CLogging.LogError('Unknown file type ! ')
        # image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        # image = image * bg_color.astype(np.uint8)
        # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        # for shape, color, dims in info['shapes']:
        #     image = self.draw_shape(image, shape, dims, color)
        return image

    def load_mask(self, image_id):
        # mask数组是的维度对应有多少个类别，class_ids长度和mask维度一致
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        classes = info['classes']
        count = len(classes)
        # return none mask
        if info["mask"] is None:
            if info['path'].split('.')[-1] == LIST_DATA_FILE_TYPE[EnumDataFileType.PKL.value]:
                img = CFunctionFileIO.ReadPKL(info["path"])
                h = img.shape[0]
                w = img.shape[1]
            elif info['path'].split('.')[-1] == LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIF.value]:
                img = CFunctionFileIO.ReadImage(info["path"])
                h = img.shape[0]
                w = img.shape[1]
            else:
                CLogging.LogError('Unknown file type ! ')
            mask = np.zeros((h, w, 1), dtype=np.uint8)
            class_ids = np.array([], dtype=np.int32)
        else:
            if info['mask'].split('.')[-1] == LIST_DATA_FILE_TYPE[EnumDataFileType.PKL.value]:
                mask = CFunctionFileIO.ReadPKL(info["mask"])
            elif info['mask'].split('.')[-1] == LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIF.value]:
                mask = CFunctionFileIO.ReadImage(info["mask"])
            else:
                CLogging.LogError('Unknown file type ! ')
            class_ids = np.array([self.class_names[1:].index(classes[i]) for i in range(count)])
        # mask = mask[:, :, 2:]
        # class_ids = class_ids[2:]
        return mask.astype(np.bool_), class_ids.astype(np.int32)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # 图像（RGB）和配套的mask
        img = self.load_image(idx)
        dataset_masks, class_ids = self.load_mask(idx)
        if ((self.model_type == EnumTypeModelType.MaskRCNN) or
            (self.model_type == EnumTypeModelType.MaskRCNNBGClassifier)
            ):
            boxes = []
            masks = []
            labels = []
            for class_id in class_ids:
                class_mask = dataset_masks[:, :, class_id]
                labeled_array, num_objects = ndimage.label(class_mask)  # 默认4连通
                for object_idx in range(1, num_objects + 1):
                    ys, xs = np.where(labeled_array == object_idx)
                    x1, x2 = np.min(xs), np.max(xs)
                    y1, y2 = np.min(ys), np.max(ys)

                    if x1 > 0:
                        x1 =  x1 - 1
                    if x2 < self.image_dim - 1:
                        x2 = x2 + 1
                    if y1 > 0:
                        y1 = y1 - 1
                    if y2 < self.image_dim - 1:
                        y2 = y2 + 1

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # 类别固定为1
                    # 将 PIL mask 转为 numpy 数组，值自然为 0 或 1（dtype=np.uint8）
                    mask = np.zeros(class_mask.shape, dtype=np.uint8)
                    mask[ys, xs] = 1
                    masks.append(mask)

            if len(boxes) == 0:
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "masks": torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.uint8),
                    'image_id': torch.tensor([idx]),  # 原样保留
                    'area': torch.zeros((0,), dtype=torch.float32),
                    'iscrowd': torch.zeros((0,), dtype=torch.int64)
                }
            else:
                # 转换为 tensor
                boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
                masks_tensor = torch.as_tensor(np.array(masks), dtype=torch.uint8)  # [N, H, W]
                area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])

                iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

                target = {
                    'boxes': boxes_tensor,
                    'labels': labels_tensor,
                    'masks': masks_tensor,
                    'image_id': torch.tensor([idx]),
                    'area': area,
                    'iscrowd': iscrowd
                }

            # 图像转为 tensor 并归一化到 [0,1]
            img_tensor = T.functional.to_tensor(img)
            return img_tensor, target
        elif self.model_type == EnumTypeModelType.Unet:
            # 转换为 tensor
            img_tensor = F.to_tensor(img)  # [3, H, W], 归一化到 [0,1]
            mask = np.zeros((dataset_masks.shape[0], dataset_masks.shape[1]), dtype=np.uint32)
            for class_id in class_ids:
                mask[dataset_masks[:, :, class_id] > 0] = class_id + 1
            mask_tensor = torch.as_tensor(mask, dtype=torch.long)
            return img_tensor, mask_tensor
        elif self.model_type == EnumTypeModelType.UNIMask2Former:

            # 将 mask 转换为 (H, W, num_classes) 以便与 albumentations 空间变换兼容
            mask_hwc = dataset_masks # (H, W, num_classes)
            # 分别转换图像和 mask
            # 图像：归一化 + ToTensor
            image_tensor = self.image_transform(image=img)['image']  # (3, H, W)
            # mask：直接转为 torch.LongTensor，形状 (num_classes, H, W)
            mask_tensor = torch.from_numpy(np.transpose(mask_hwc, (2, 0, 1))).long()  # (num_classes, H, W)
            # 提取实例 masks 和 labels
            masks_list = []
            labels_list = []
            num_classes = mask_tensor.shape[0]
            for c in range(num_classes):
                channel = mask_tensor[c]  # (H, W)
                unique_ids = torch.unique(channel)
                unique_ids = unique_ids[unique_ids != 0]
                for inst_id in unique_ids:
                    binary_mask = (channel == inst_id).float()
                    masks_list.append(binary_mask)
                    labels_list.append(c + 1)  # 类别 ID 从 1 开始

            if len(masks_list) == 0:
                # 无实例（理论上很少），返回空
                masks_tensor = torch.zeros((0, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)
                labels_tensor = torch.zeros(0, dtype=torch.int64)
            else:
                masks_tensor = torch.stack(masks_list, dim=0)
                labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

            return {
                'image': image_tensor,
                'masks': masks_tensor,
                'labels': labels_tensor
            }
        else:
            return None, None

