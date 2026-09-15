# 用于调试查找值的一些函数
from ApuCellGlobal.MSApuCellExternalPackge import *

class CFunctionDebug:
    @staticmethod
    def NumpyArray2Clipboard(data:np.array):
        df = pd.DataFrame(data)
        df.to_clipboard(index=False, header=False)

    @staticmethod
    def WatchImage(img:np.array, x_start:int, x_end:int, y_start:int, y_end:int):

        return_img = None
        if len(img.shape) == 2:
            return_img = img[y_start: y_end, x_start: x_end]
        elif len(img.shape) == 3:
            return_img = img[y_start: y_end, x_start: x_end, :]
        return return_img

    @staticmethod
    def WatchImageAddMask(img:np.array, mask :np.array):

        assert len(img.shape) == 3
        assert len(mask.shape) == 2
        assert img.shape[2] == 3
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]

        img[:, :, 1][mask > 0] = 255