import sys
import torch

if __name__ == "__main__":
    # 1. 检查 PyTorch 是否编译了 CUDA（通过 version.cuda 是否非 None）
    cuda_version = torch.version.cuda
    if cuda_version is not None:
        print(f"PyTorch built with CUDA: Yes (CUDA version {cuda_version})")
        print("CUDA installation successful")
    else:
        print("PyTorch built with CUDA: No")
        print("CUDA not installed or PyTorch not compiled with CUDA support")

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
    if gpu_available:
        print("GPU is available, PyTorch will use GPU acceleration")
    else:
        print("GPU not available, will fall back to CPU")

    # 4. 设置退出码
    if gpu_available:
        sys.exit(0)   # success
    else:
        sys.exit(-1)  # failure