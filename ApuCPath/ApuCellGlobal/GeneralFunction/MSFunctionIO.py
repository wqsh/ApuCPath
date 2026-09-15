from ApuCellGlobal.MSApuCellExternalPackge import *

from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC

from ApuCellGlobal.GeneralSystem.MSSystemEnum import EnumVisualFileType, EnumDataFileType, LIST_VISUAL_FILE_TYPE, LIST_DATA_FILE_TYPE, LIST_VISUAL_FILE_TYPE_SUFFIX
from ApuCellGlobal.GeneralSystem.MSLoggingInfo import ERROR_TO_USER
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging, Assert
from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory

from osgeo import gdal
from multiprocessing import Queue

IMAGE_SIZE = 5 * 1e4

@unique
class EnumModeIOImage(Enum):

    OpenCV = 1
    PIL = 2
    Matplotlib = 3
    Scipy = 4

class CFunctionFileIO:

    @staticmethod
    def ReadImage(path_image:str, mode:int=1, slide_level:int=0):
        '''
        :param path_image:str
        :param mode:EnumVisualFileType
        :return: np.array
        '''
        image_type = path_image.split('.')[-1]
        image_type = str.lower(image_type)
        if image_type in LIST_VISUAL_FILE_TYPE:
            w, h = CFunctionFileIO.GetImageFileSize(path_image)
            if (w > IMAGE_SIZE) or (h > IMAGE_SIZE):
                # 超过这个尺寸就用pyvips
                CLogging.LogToUser(path_image + " height : %d width : %d" % (h, w))
                CLogging.LogToUser("Use pyvips for large image ... ")
                img = CFunctionFileIO.ReadImageLowMemory(path_image)
                CLogging.LogToUser("Finished load image . ")
                return img
            else:
                image_type_idx = LIST_VISUAL_FILE_TYPE.index(image_type)
                if EnumVisualFileType.PNG.value == image_type_idx:
                    return CFunctionFileIO.__captainLoadImagePNG(path_image, mode)
                elif EnumVisualFileType.JPG.value == image_type_idx:
                    return CFunctionFileIO.__captainLoadImageJPG(path_image, mode)
                elif EnumVisualFileType.TIF.value == image_type_idx:
                    return CFunctionFileIO.__captainLoadImageTif(path_image)
                elif EnumVisualFileType.TIFF.value == image_type_idx:
                    return CFunctionFileIO.__captainLoadImageTif(path_image)
                elif EnumVisualFileType.SVS.value == image_type_idx:
                    return CFunctionFileIO.__captainLoadSVS(path_image, slide_level)
                else:
                    CLogging.LogError(ERROR_TO_USER.NONSUPPORT_FILE + path_image)
        else:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_FILE + path_image)

    @staticmethod
    def __captainLoadImagePNG(path_image: str, mode:int=1):
        Assert(os.path.exists(path_image))
        return CFunctionFileIO.__soldierReadImage(path_image, mode)

    @staticmethod
    def __captainLoadImageJPG(path_image: str, mode:int=1):

        Assert(os.path.exists(path_image))
        return CFunctionFileIO.__soldierReadImage(path_image, mode)

    @staticmethod
    def __captainLoadImageTif(path_image:str):
        try:
            img = tf.imread(path_image)
        except:
            CLogging.LogWarning("Use tifffile failed . ")
        else:
            return img
        try:
            img = CFunctionFileIO.__soldierReadImage(path_image, EnumModeIOImage.OpenCV.value)
        except:
            CLogging.LogWarning("Use opencv failed . ")
        else:
            return img
        CLogging.LogWarning("Use PIL failed . This function may be very slow . ")
        img = CFunctionFileIO.__soldierReadImage(path_image, EnumModeIOImage.PIL.value)
        return img

    @staticmethod
    def __captainLoadSVS(path_image:str, slide_level:int=0):
        slide = openslide.open_slide(path_image)
        img = np.array(slide.read_region((0, 0), level=slide_level, size=slide.level_dimensions[slide_level]))[:, :, 0:3]
        return img
    @staticmethod
    def __soldierReadImage(path_image:str, mode:int=0):
        if mode == EnumModeIOImage.OpenCV.value:
            img = cv2.imread(path_image)
            if (len(img.shape) == 3):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == EnumModeIOImage.PIL.value:
            img = np.array(Image.open(path_image))
        elif mode == EnumModeIOImage.Matplotlib.value:
            img = plt.imread(path_image)
            img = copy.deepcopy(img)
        elif mode == EnumModeIOImage.Scipy.value:
            img = imageio.imread(path_image)
        else:
            return None
        return img

    @staticmethod
    def WriteImage(path_folder:str, image_name:str, image_array:np.array, mode=EnumModeIOImage.OpenCV.value):
        if not os.path.exists(path_folder):
            CLogging.LogForLackPath(path_folder)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        w, h = image_array.shape[1], image_array.shape[0]
        if (w > IMAGE_SIZE) or (h > IMAGE_SIZE):
            # 超过这个尺寸就用pyvips
            CLogging.LogToUser(os.path.join(path_folder, image_name) + " height : %d width : %d" % (h, w))
            CFunctionFileIO.WriteImageLowMemory(path_folder, image_name, image_array)
            CFunctionFileIO.RepairOverview(os.path.join(path_folder, image_name))
        else:
            if mode == EnumModeIOImage.OpenCV.value:
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(path_folder, image_name), image_array)
            elif mode == EnumModeIOImage.PIL.value:
                im = Image.fromarray(image_array)
                im.save(os.path.join(path_folder, image_name))
            elif mode == EnumModeIOImage.Matplotlib.value:
                plt.imsave(os.path.join(path_folder, image_name), image_array)
            elif mode == EnumModeIOImage.Scipy.value:
                io.imsave(os.path.join(path_folder, image_name), image_array)

    @staticmethod
    def ReadMat(mat_path:str):
        if not os.path.exists(mat_path):
            CLogging.LogForLackPath(mat_path)
        return sio.loadmat(mat_path)

    @staticmethod
    def WriteMat(folder:str, file_name:str, data:dict):
        if not os.path.exists(folder):
            os.makedirs(folder)
        sio.savemat(os.path.join(folder, file_name), data)

    @staticmethod
    def ReadMask(mat_path:str, print_Flag:bool=False):

        if not os.path.exists(mat_path):
            CLogging.LogForLackPath(mat_path)

        mask = sio.loadmat(mat_path)

        mask = (mask['mask']).astype("int32")
        num = np.max(mask)
        # num=100
        if print_Flag:
            CLogging.LogToUser("Nucleus number " + str(num))

    @staticmethod
    def ReadImageGrayscale(mask_png_path:str, mode:int=1, print_Flag:bool=False):
        if not os.path.exists(mask_png_path):
            CLogging.LogForLackPath(mask_png_path)
        # 按照灰度读取
        if mode == EnumModeIOImage.OpenCV.value:
            img = cv2.imread(mask_png_path, cv2.IMREAD_GRAYSCALE)
        elif mode == EnumModeIOImage.Matplotlib.value:
            img = CFunctionFileIO.__soldierReadImage(mask_png_path, mode)
            img = CFunctionImageOperator.RGB2Gray(img)
        else:
            img = CFunctionFileIO.__soldierReadImage(mask_png_path, mode)
            img = CFunctionImageOperator.RGB2Gray(img)
        # 自动将读到的矩阵灰度值归一化为0-1
        img = img.astype(np.uint8)
        return img

    @staticmethod
    def GetImageFileSize(path_image:str, mode:int=1, slide_level:int=0):
        path_image_suffix = path_image.split('.')[-1].lower()
        if path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.SVS.value]):
            img_array = CFunctionFileIO.ReadImage(path_image, mode, slide_level)
            return (img_array.shape[0], img_array.shape[1])
        elif (path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.JPG.value]) or
            path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.PNG.value]) or
            path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIF.value]) or
            path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIFF.value])
            ):
            # w, h
            return imagesize.get(path_image)
        else:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_FILE)

    @staticmethod
    def GetImageChannelNum(path_image:str, mode:int=1, slide_level:int=0):
        path_image_suffix = path_image.split('.')[-1].lower()
        if path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.SVS.value]):
            img_array = CFunctionFileIO.ReadImage(path_image, mode, slide_level)
            return img_array.shape[2]
        elif (path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.JPG.value]) or
              path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.PNG.value]) or
              path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIF.value]) or
              path_image_suffix.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIFF.value])
              ):
            # 获取图像的模式，比如 'RGB', 'RGBA', 'L', 等等
            with Image.open(path_image) as img:
                mode = img.mode

                # 根据模式确定通道数
                if mode == 'L':  # 单通道灰度图像
                    return 1
                elif mode == 'RGB':  # 三通道彩色图像
                    return 3
                elif mode == 'RGBA':  # 四通道图像（RGB + Alpha）
                    return 4
                elif mode == 'CMYK':  # 四通道印刷图像
                    return 4
                elif mode == 'P':  # 调色板图像
                    return 1  # 通常视为单通道
                else:
                    CLogging.LogError(f"Unsupported image mode: {mode}")
        else:
            CLogging.LogError(ERROR_TO_USER.NONSUPPORT_FILE)

    @staticmethod
    def ReadPKL(path:str):
        if not os.path.exists(path):
            CLogging.LogForLackPath(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def WritePKL(path:str, name:str, data):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, name), "wb") as f:
            pickle.dump(data, f, protocol=4)

    @staticmethod
    def ReadFile(path:str):
        file_type:str = path.split(".")[-1]
        if file_type in LIST_DATA_FILE_TYPE:
            file_type_idx = LIST_DATA_FILE_TYPE.index(file_type)
            if file_type_idx == EnumDataFileType.IMAGE.value:
                return CFunctionFileIO.ReadPKL(path)
                # elif file_type_idx == EnumDataFileType.LABEL.value:
            else:
                return None
        elif file_type in LIST_VISUAL_FILE_TYPE:
            file_type_idx = LIST_VISUAL_FILE_TYPE.index(file_type)
            if file_type_idx == EnumVisualFileType.PNG.value or file_type_idx == EnumVisualFileType.TIF.value or file_type_idx == EnumVisualFileType.JPG.value:
                return CFunctionFileIO.ReadImage(path)
            else:
                return None

    @staticmethod
    def WriteJson(path:str, name:str, data:dict):

        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, name), 'w') as f:
            json.dump(data, f)

    @staticmethod
    def ReadJson(path:str):
        if not os.path.exists(path):
            CLogging.LogForLackPath(path)
        with open(os.path.join(path), 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def ReadNpy(path:str):
        if not os.path.exists(path):
            CLogging.LogForLackPath(path)
        return np.load(path)

    @staticmethod
    def WriteNpy(path:str, name:str, data:np.array):

        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, name), data)

    @staticmethod
    def WriteImageLowMemory(path:str, name:str, image:np.array):
        img_pyvips = pyvips.Image.new_from_array(image)
        img_pyvips.write_to_file(os.path.join(path, name))

    @staticmethod
    def ReadImageLowMemory(path_image:str, remove_alpha:bool=True):
        if remove_alpha:
            if pyvips.Image.new_from_file(path_image).bands == 4:
                pyvips_img = pyvips.Image.new_from_file(path_image).flatten()
            else:
                pyvips_img = pyvips.Image.new_from_file(path_image)
        else:
            pyvips_img = pyvips.Image.new_from_file(path_image)
        CFunctionMemory.CleanMemory()
        CFunctionMemory.GetCurMemoryInfo()
        try:
            img = pyvips_img.numpy()
        except Exception as msg:
            CLogging.LogWarning(str(msg))
            CLogging.LogToUser("Using gdal reading image ... " + path_image)
            img = CFunctionFileIO.ReadTIFByGDAl(path_image)
        CFunctionMemory.GetCurMemoryInfo()
        return img

    @staticmethod
    def ReadTIFByGDAl(path_img:str):

        image = gdal.Open(path_img)
        band = image.RasterCount
        img_width = image.RasterXSize
        img_height = image.RasterYSize

        image_np = np.zeros((img_height, img_width, band), dtype=np.uint8)

        for i in range(band):
            image_band = image.GetRasterBand(i + 1)
            data = image_band.ReadAsArray(0, 0, img_width, img_height)
            image_np[:, :, i] = data[:, :]
            del data
            CFunctionMemory.CleanMemory()
        return image_np

    @staticmethod
    def WriteTIFByGDAL(image_path:str, image_name:str, image:np.array, overview:bool=True):
        if image_name.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIFF.value]):
            target_path = os.path.join(image_path, image_name)
        else:
            if "." in image_name:
                target_path = os.path.join(image_path, image_name)
            else:
                target_path = os.path.join(image_path, image_name + LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value])

        band = 3
        if len(image.shape) == 2:
            band = 3
        elif len(image.shape) == 3:
            band = image.shape[2]
        else:
            CLogging.LogError("Image channel must be 3 ! ")

        width = image.shape[1]
        height = image.shape[0]

        queue = Queue()

        gdal.UseExceptions()
        gdal.SetConfigOption(
            "GDAL_NUM_THREADS", str((psutil.cpu_count(logical=False) - 1) // 3)
        )
        gdal.SetConfigOption(
            "GDAL_CACHEMAX",
            str((psutil.virtual_memory().available * 0.2) // (1024 * 1024)),
        )
        task_size = 4096

        target = gdal.GetDriverByName("GTiff").Create(
            target_path,
            width,
            height,
            band,
            gdal.GDT_Byte,
            options=[
                "BIGTIFF=YES",
                "COMPRESS=LZW",
                f"PHOTOMETRIC={'MINISBLACK' if band == 1 else 'RGB'}",
                # "JPEG_QUALITY=80",
                "INTERLEAVE=PIXEL",
                # "JPEGTABLESMODE=1",
                "TILED=YES",
                f"BLOCKXSIZE=512",
                f"BLOCKYSIZE=512",
                f"NUM_THREADS={(psutil.cpu_count(logical=False) - 1) // 3}",
            ],
        )
        total = math.ceil(height / task_size) * math.ceil(width / task_size)
        count = 0
        for y in range(0, height, task_size):
            for x in range(0, width, task_size):
                tile_width = min(task_size, width - x)
                tile_height = min(task_size, height - y)
                if tile_height != task_size:
                    copy_y_length = height - y
                else:
                    copy_y_length = task_size
                if tile_width != task_size:
                    copy_x_length = width - x
                else:
                    copy_x_length = task_size
                if len(image.shape) == 2:
                    data = np.zeros((tile_height, tile_width), dtype=np.uint8)
                    data[:copy_y_length, :copy_x_length] = image[y: y + copy_y_length, x: x + copy_x_length]
                    for band_idx in range(band):
                        target.GetRasterBand(band_idx + 1).WriteArray(data[:, :], x, y)
                else:
                    data = np.zeros((tile_height, tile_width, band), dtype=np.uint8)
                    for band_idx in range(band):
                        data[:copy_y_length, :copy_x_length, band_idx] = image[y: y + copy_y_length, x : x + copy_x_length, band_idx]
                        target.GetRasterBand(band_idx + 1).WriteArray(data[:, :, band_idx], x, y)
                target.FlushCache()
                count = count + 1
                queue.put(int(count * 100 / total))

        target = None
        del target

        if overview:
            CFunctionFileIO.RepairOverview(target_path)

    @staticmethod
    def WriteTIFByGDALCompress(image_path:str, image_name:str, image:np.array, overview:bool=True):
        if image_name.endswith(LIST_VISUAL_FILE_TYPE[EnumVisualFileType.TIFF.value]):
            target_path = os.path.join(image_path, image_name)
        else:
            if "." in image_name:
                target_path = os.path.join(image_path, image_name)
            else:
                target_path = os.path.join(image_path, image_name + LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value])

        band = 3
        if len(image.shape) == 2:
            band = 3
        elif len(image.shape) == 3:
            band = image.shape[2]
        else:
            CLogging.LogError("Image channel must be 3 ! ")

        width = image.shape[1]
        height = image.shape[0]

        queue = Queue()

        gdal.UseExceptions()
        gdal.SetConfigOption(
            "GDAL_NUM_THREADS", str((psutil.cpu_count(logical=False) - 1) // 3)
        )
        gdal.SetConfigOption(
            "GDAL_CACHEMAX",
            str((psutil.virtual_memory().available * 0.2) // (1024 * 1024)),
        )
        task_size = 4096

        target = gdal.GetDriverByName("GTiff").Create(
            target_path,
            width,
            height,
            band,
            gdal.GDT_Byte,
            options=[
                "BIGTIFF=YES",
                "COMPRESS=JPEG",
                f"PHOTOMETRIC={'MINISBLACK' if band == 1 else 'YCBCR'}",
                "JPEG_QUALITY=80",
                "INTERLEAVE=PIXEL",
                "JPEGTABLESMODE=1",
                "TILED=YES",
                f"BLOCKXSIZE=512",
                f"BLOCKYSIZE=512",
                f"NUM_THREADS={(psutil.cpu_count(logical=False) - 1) // 3}",
            ],
        )
        total = math.ceil(height / task_size) * math.ceil(width / task_size)
        count = 0
        for y in range(0, height, task_size):
            for x in range(0, width, task_size):
                tile_width = min(task_size, width - x)
                tile_height = min(task_size, height - y)
                if tile_height != task_size:
                    copy_y_length = height - y
                else:
                    copy_y_length = task_size
                if tile_width != task_size:
                    copy_x_length = width - x
                else:
                    copy_x_length = task_size
                if len(image.shape) == 2:
                    data = np.zeros((tile_height, tile_width), dtype=np.uint8)
                    data[:copy_y_length, :copy_x_length] = image[y: y + copy_y_length, x: x + copy_x_length]
                    for band_idx in range(band):
                        target.GetRasterBand(band_idx + 1).WriteArray(data[:, :], x, y)
                else:
                    data = np.zeros((tile_height, tile_width, band), dtype=np.uint8)
                    for band_idx in range(band):
                        data[:copy_y_length, :copy_x_length, band_idx] = image[y: y + copy_y_length, x : x + copy_x_length, band_idx]
                        target.GetRasterBand(band_idx + 1).WriteArray(data[:, :, band_idx], x, y)
                target.FlushCache()
                count = count + 1
                queue.put(int(count * 100 / total))

        target = None
        del target

        if overview:
            CFunctionFileIO.RepairOverview(target_path)

    @staticmethod
    def RepairOverview(path_tiff:str):

        dataset = gdal.Open(path_tiff, gdal.GA_Update)

        gdal.SetConfigOption("COMPRESS_OVERVIEW", "JPEG")
        gdal.SetConfigOption("JPEG_QUALITY_OVERVIEW", "80")
        gdal.SetConfigOption(
            "PHOTOMETRIC_OVERVIEW",
            "MINISBLACK" if dataset.RasterCount == 1 else "YCBCR",
        )
        gdal.SetConfigOption(
            "GDAL_NUM_THREADS", str((psutil.cpu_count(logical=False) - 1) // 3)
        )
        gdal.SetConfigOption(
            "GDAL_CACHEMAX",
            str((psutil.virtual_memory().available * 0.2) // (1024 * 1024)),
        )
        gdal.SetConfigOption("BLOCKXSIZE_OVERVIEW", "512")
        gdal.SetConfigOption("BLOCKYSIZE_OVERVIEW", "512")
        gdal.SetConfigOption("JPEGTABLESMODE", "1")
        gdal.SetConfigOption("BIGTIFF_OVERVIEW", "IF_SAFER")
        gdal.SetConfigOption("COPY_SRC_OVERVIEWS", "NO")

        dataset.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64, 128])
        dataset.FlushCache()
        dataset = None
        del dataset
