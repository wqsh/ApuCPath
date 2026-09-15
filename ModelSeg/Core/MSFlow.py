from Core.MSDataPack import CDataPack
from Core.MSTask import CTaskTrain, CTaskTrainDatasetValidDatasetForTrain, CTaskPredict

class CFlow:

    def __init__(self, inputDP:CDataPack, flow_enum:int=0):
        self.dp = inputDP
        self.__FlowEnum:int = flow_enum

    def GetFlowEnum(self):

        return self.__FlowEnum

class CFlowTrain(CFlow):

    def __init__(self, inputDP:CDataPack, flow_enum:int=1):
        super(CFlowTrain, self).__init__(inputDP, flow_enum)

    def run(self):
        # 开始
        taskTrain = CTaskTrain(self.dp)
        taskTrain.TrainModel()

class CFlowTrainDatasetValidDatasetForTrain(CFlow):

    def __init__(self, inputDP:CDataPack, flow_enum:int = 1):

        super(CFlowTrainDatasetValidDatasetForTrain, self).__init__(inputDP, flow_enum)

    def run(self):

        taskTrainDatasetValidDatasetForTrain = CTaskTrainDatasetValidDatasetForTrain(self.dp)
        taskTrainDatasetValidDatasetForTrain.TrainModel()

class CFlowPredict(CFlow):

    def __init__(self, inputDP:CDataPack, flow_enum:int=1):
        super(CFlowPredict, self).__init__(inputDP, flow_enum)

    def run(self):
        # 开始
        # taskPredict = CTaskPredict(self.dp)
        taskPredict = CTaskPredict(self.dp)
        taskPredict.Predict()

class CFlowCheckGPU(CFlow):

    def __init__(self):

        pass

    def run(self):

        import sys
        import torch
        import subprocess
        import re

        def get_system_cuda_version():
            """获取系统 nvcc 的 CUDA 版本，返回字符串如 '11.8'，失败返回 None"""
            try:
                output = subprocess.check_output(["nvcc", "--version"], text=True)
                match = re.search(r"release (\d+\.\d+)", output)
                return match.group(1) if match else None
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        cuda_available = False
        # 1. 检查 PyTorch 是否编译了 CUDA（通过 version.cuda 是否非 None）
        cuda_version = torch.version.cuda
        if cuda_version is not None:
            print(f"PyTorch built with CUDA: Yes (CUDA version {cuda_version})")
            print("CUDA installation successful")
            cuda_available = True
        else:
            print("PyTorch built with CUDA: No")
            print("CUDA not installed or PyTorch not compiled with CUDA support")

        # 2. 获取系统 CUDA 版本
        system_cuda = get_system_cuda_version()
        if system_cuda is None:
            print("Can't not find nvcc，please check CUDA Toolkit.")
        else:
            print(f"System CUDA 版本: {system_cuda}")

        # 3. 比较版本（仅比较 major.minor，忽略补丁号）
        if cuda_version and system_cuda:
            # 提取主次版本号（例如 "11.8"）
            pytorch_ver = ".".join(cuda_version.split(".")[:2])
            system_ver = ".".join(system_cuda.split(".")[:2])
            if pytorch_ver == system_ver:
                print(f"Versions match: both are {pytorch_ver}")
                cuda_version = True
            else:
                cuda_version = False
                print(f"Version mismatch: PyTorch is {pytorch_ver}, system is {system_ver}")

        # 2. 列出可用的 GPU 设备
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            devices = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            print(f"Detected GPU devices: {devices}")
            print(f"Found {gpu_count} GPU device(s)")
        else:
            print("Detected GPU devices: []")
            print("No GPU device detected")

        # 3. 检查运行时 GPU 是否可用（综合了编译+驱动+硬件）
        gpu_available = torch.cuda.is_available()
        print(f"GPU available: {gpu_available}")
        if gpu_available and cuda_version:
            print("GPU is available, PyTorch will use GPU acceleration")
        else:
            print("GPU not available, will fall back to CPU")

        # 4. 设置退出码
        if gpu_available and cuda_available:
            sys.exit(0)  # success
        else:
            sys.exit(-1)  # failure

if __name__ == "__main__":
    CFlowCheckGPU().run()