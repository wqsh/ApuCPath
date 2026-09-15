from ApuCellGlobal.MSApuCellExternalPackge import *

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

class CFunctionConv2d:

    @staticmethod
    def Conv2d(input_data:np.array, kernel:np.array, stride=1, padding=0):


        # 填充输入数据（根据填充数量添加额外的行和列）
        if padding > 0:
            input_data = np.pad(input_data, ((padding, padding), (padding, padding)), mode='constant')

        # 获取输入数据的尺寸
        input_height, input_width = input_data.shape
        # 获取卷积核的尺寸
        kernel_height, kernel_width = kernel.shape
        if kernel_height != kernel_width:
            CLogging.LogError("kernel_height != kernel_width")
        # 计算输出图像的尺寸
        output_height = (input_height - kernel_height) // stride + 1
        output_width = (input_width - kernel_width) // stride + 1

        # 初始化输出图像s
        output_data = np.zeros((output_height, output_width))

        # 执行卷积操作
        for i in range(0, input_height - kernel_height + 1, stride):
            for j in range(0, input_width - kernel_width + 1, stride):
                output_data[i // stride, j // stride] = np.sum(input_data[i:i + kernel_height, j:j + kernel_width] * kernel)

        return output_data

    @staticmethod
    def FFTConv2d(input_data:np.array, kernel:np.array):
        # 执行卷积操作
        fft_kernel = np.fft.fft2(kernel, kernel.shape)
        print("fft_kernel")
        out_data = np.real(np.fft.ifft2(np.fft.fft2(input_data) * fft_kernel))
        return out_data

    @staticmethod
    def PaddingForConv2dFFT(input_data:np.array, padding_height:int, padding_width:int):
        # 填充输入数据（根据填充数量添加额外的行和列）
        out_data = np.pad(input_data, ((0, padding_height), (0, padding_width)), mode='constant')
        out_data = out_data.astype(np.float32)
        return out_data