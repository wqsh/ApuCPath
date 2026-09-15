from ApuCellGlobal.MSApuCellGlobalImport import *
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory


import math

class CFunctionWSIImage:

    __WORK_DICTIONARY = os.path.join(os.getcwd(), 'tmp', 'WSI')
    __SPLIT_IMAGE = 5
    @staticmethod
    def PadImage(image_path:str, row_size:int=512, col_size:int=512):
        img = CFunctionFileIO.ReadImage(image_path)

        half_row_size = int(row_size / 2)
        half_col_size = int(col_size / 2)

        x_min = 0
        x_max = img.shape[1]
        y_min = 0
        y_max = img.shape[0]

        row_length = y_max - y_min
        col_length = x_max - x_min

        row_num = math.ceil((y_max - y_min) / row_size)
        col_num = math.ceil((x_max - x_min) / col_size)

        col_num += 1
        row_num += 1

        CLogging.LogToUser("finish load image")
        CFunctionMemory.GetCurMemoryInfo()

        new_img = np.zeros((row_size * row_num, col_size * col_num, 3), dtype=np.uint8)
        new_img[y_min + half_row_size: y_min + half_row_size + row_length,
        x_min + half_col_size: half_col_size + col_length + x_min, :] += img[y_min: y_max, x_min: x_max, :]
        return new_img

    @staticmethod
    def PadMask(image_path:str, row_size:int=512, col_size:int=512):
        CLogging.LogToUser('Start pad mask ...')
        img_mask = CFunctionImageOperator.RGB2Gray(CFunctionFileIO.ReadImage(image_path))

        half_row_size = int(row_size / 2)
        half_col_size = int(col_size / 2)

        x_min = 0
        x_max = img_mask.shape[1]
        y_min = 0
        y_max = img_mask.shape[0]

        row_length = y_max - y_min
        col_length = x_max - x_min

        row_num = math.ceil((y_max - y_min) / row_size)
        col_num = math.ceil((x_max - x_min) / col_size)

        col_num += 1
        row_num += 1

        CFunctionMemory.GetCurMemoryInfo()

        new_mask = np.zeros((row_size * row_num, col_size * col_num), dtype=np.uint8)
        new_mask[y_min + half_row_size: y_min + half_row_size + row_length, x_min + half_col_size: half_col_size + col_length + x_min] += img_mask[y_min: y_max, x_min: x_max]
        CLogging.LogToUser('Finish pad mask ...')
        return new_mask

    @staticmethod
    def PadImageLowMemory(image_path:str, row_size:int=512, col_size:int=512):
        CFunctionWSIImage.__captainCreateWorkDictionary()
        CLogging.LogToUser('Start pad img ' + image_path + ' ...')
        img = CFunctionFileIO.ReadImage(image_path)
        CFunctionMemory.GetCurMemoryInfo()
        name = os.path.basename(image_path).split('.')[0]
        half_row_size = int(row_size / 2)
        half_col_size = int(col_size / 2)

        x_min = 0
        x_max = img.shape[1]
        y_min = 0
        y_max = img.shape[0]

        row_length = y_max - y_min
        col_length = x_max - x_min

        row_num = math.ceil((y_max - y_min) / row_size)
        col_num = math.ceil((x_max - x_min) / col_size)

        col_num += 1
        row_num += 1

        tmp_file_list = []
        if row_length % CFunctionWSIImage.__SPLIT_IMAGE == 0:
            split_size_row = row_length // CFunctionWSIImage.__SPLIT_IMAGE
        else:
            split_size_row = row_length // CFunctionWSIImage.__SPLIT_IMAGE + 1
        if col_length % CFunctionWSIImage.__SPLIT_IMAGE == 0:
            split_size_col = col_length // CFunctionWSIImage.__SPLIT_IMAGE
        else:
            split_size_col = col_length// CFunctionWSIImage.__SPLIT_IMAGE + 1
        CLogging.LogToUser("Generating WSI image index ...")
        for i in range(CFunctionWSIImage.__SPLIT_IMAGE):
            for j in range(CFunctionWSIImage.__SPLIT_IMAGE):
                out_name = name + '_%d_%d.pkl'%(i, j)
                CFunctionFileIO.WritePKL(CFunctionWSIImage.__WORK_DICTIONARY, out_name, img[i * split_size_row: (i + 1) * split_size_row, j * split_size_col: (j + 1) * split_size_col, :])
                tmp_file_list.append(os.path.join(CFunctionWSIImage.__WORK_DICTIONARY, out_name))
                CLogging.LogToUser('Output ' + os.path.join(CFunctionWSIImage.__WORK_DICTIONARY, out_name))
        del img
        CFunctionMemory.CleanMemory()
        CLogging.LogToUser("Finish generate WSI image index. ")
        new_img = np.zeros((row_size * row_num, col_size * col_num, 3), dtype=np.uint8)
        for item in tmp_file_list:
            CLogging.LogToUser(item)
            CFunctionWSIImage.__captainPadImage(new_img, item, split_size_row, split_size_col, half_row_size, half_col_size)
            CFunctionMemory.CleanMemory()
        CFunctionWSIImage.__catainCleanWorkDictionary()
        CLogging.LogToUser("Finish pad image. ")
        return new_img

    @staticmethod
    def PadMaskLowMemory(mask_path:str, row_size:int = 512, col_size:int=512):

        CFunctionWSIImage.__captainCreateWorkDictionary()
        CLogging.LogToUser('Start pad mask ' + mask_path + ' ...')
        mask = CFunctionImageOperator.RGB2Gray(CFunctionFileIO.ReadImage(mask_path))
        CFunctionMemory.GetCurMemoryInfo()
        name = os.path.basename(mask_path).split('.')[0]
        half_row_size = int(row_size / 2)
        half_col_size = int(col_size / 2)

        x_min = 0
        x_max = mask.shape[1]
        y_min = 0
        y_max = mask.shape[0]

        row_length = y_max - y_min
        col_length = x_max - x_min

        row_num = math.ceil((y_max - y_min) / row_size)
        col_num = math.ceil((x_max - x_min) / col_size)

        col_num += 1
        row_num += 1

        tmp_file_list = []
        if row_length % CFunctionWSIImage.__SPLIT_IMAGE == 0:
            split_size_row = row_length // CFunctionWSIImage.__SPLIT_IMAGE
        else:
            split_size_row = row_length // CFunctionWSIImage.__SPLIT_IMAGE + 1
        if col_length % CFunctionWSIImage.__SPLIT_IMAGE == 0:
            split_size_col = col_length // CFunctionWSIImage.__SPLIT_IMAGE
        else:
            split_size_col = col_length // CFunctionWSIImage.__SPLIT_IMAGE + 1
        CLogging.LogToUser("Generating WSI mask index ...")
        for i in range(CFunctionWSIImage.__SPLIT_IMAGE):
            for j in range(CFunctionWSIImage.__SPLIT_IMAGE):
                out_name = name + '_%d_%d.pkl' % (i, j)
                CFunctionFileIO.WritePKL(CFunctionWSIImage.__WORK_DICTIONARY, out_name, mask[i * split_size_row: (i + 1) * split_size_row, j * split_size_col: (j + 1) * split_size_col])
                tmp_file_list.append(os.path.join(CFunctionWSIImage.__WORK_DICTIONARY, out_name))
                CLogging.LogToUser('Output ' + os.path.join(CFunctionWSIImage.__WORK_DICTIONARY, out_name))
        del mask
        CFunctionMemory.CleanMemory()
        CLogging.LogToUser("Finish generate WSI mask index. ")
        new_mask = np.zeros((row_size * row_num, col_size * col_num), dtype=np.uint8)
        for item in tmp_file_list:
            CLogging.LogToUser(item)
            CFunctionWSIImage.__captainPadMask(new_mask, item, split_size_row, split_size_col, half_row_size, half_col_size)
            CFunctionMemory.CleanMemory()
        CFunctionWSIImage.__catainCleanWorkDictionary()
        CLogging.LogToUser("Finish pad mask. ")
        return new_mask

    @staticmethod
    def __captainPadImage(pad_img:np.array, path_part_img:np.array, split_size_row, split_size_col, half_row_size, half_col_size):
        part_img = CFunctionFileIO.ReadPKL(path_part_img)
        file_name = os.path.basename(path_part_img).split('.')[0]
        part_img_row_idx = int(file_name.split('_')[-2])
        part_img_col_idx = int(file_name.split('_')[-1])

        new_img_start_row = part_img_row_idx * split_size_row + half_row_size
        new_img_end_row = part_img_row_idx * split_size_row + part_img.shape[0] + half_row_size

        new_img_start_col = part_img_col_idx * split_size_col + half_col_size
        new_img_end_col = part_img_col_idx * split_size_col + part_img.shape[1] + half_col_size

        pad_img[new_img_start_row: new_img_end_row, new_img_start_col: new_img_end_col, :] += part_img[:, :, :]
        del part_img
        CFunctionMemory.CleanMemory()

    @staticmethod
    def __captainPadMask(pad_mask:np.array, path_part_mask:np.array, split_size_row, split_size_col, half_row_size, half_col_size):
        part_mask = CFunctionFileIO.ReadPKL(path_part_mask)
        file_name = os.path.basename(path_part_mask).split('.')[0]
        part_img_row_idx = int(file_name.split('_')[-2])
        part_img_col_idx = int(file_name.split('_')[-1])

        new_img_start_row = part_img_row_idx * split_size_row + half_row_size
        new_img_end_row = part_img_row_idx * split_size_row + part_mask.shape[0] + half_row_size

        new_img_start_col = part_img_col_idx * split_size_col + half_col_size
        new_img_end_col = part_img_col_idx * split_size_col + part_mask.shape[1] + half_col_size

        pad_mask[new_img_start_row: new_img_end_row, new_img_start_col: new_img_end_col] += part_mask[:, :]
        del part_mask
        CFunctionMemory.CleanMemory()

    @staticmethod
    def __captainCreateWorkDictionary():

        if not os.path.exists(CFunctionWSIImage.__WORK_DICTIONARY):
            os.makedirs(CFunctionWSIImage.__WORK_DICTIONARY)
        else:
            list_file = os.listdir(CFunctionWSIImage.__WORK_DICTIONARY)
            for item in list_file:
                os.remove(os.path.join(CFunctionWSIImage.__WORK_DICTIONARY, item))

    @staticmethod
    def __catainCleanWorkDictionary():

        file_list = os.listdir(CFunctionWSIImage.__WORK_DICTIONARY)
        for item in file_list:
            os.remove(os.path.join(CFunctionWSIImage.__WORK_DICTIONARY, item))

    @staticmethod
    def SetWorkDictionary(path:str):

        CFunctionWSIImage.__WORK_DICTIONARY = os.path.join(path, 'tmp', 'WSI')

    @staticmethod
    def GetWorkDictionary():
        return CFunctionWSIImage.__WORK_DICTIONARY