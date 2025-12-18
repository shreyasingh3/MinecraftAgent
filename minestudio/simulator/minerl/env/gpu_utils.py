'''
Date: 2024-11-29 11:05:35

LastEditTime: 2024-11-29 11:07:37
FilePath: /MineStudio/minestudio/simulator/minerl/env/gpu_utils.py
'''
# https://nvidia.github.io/cuda-python/
# CUDA is optional - not available on macOS (MPS) or CPU-only systems
try:
    from cuda import cuda, cudart
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None
    cudart = None

import argparse
import os
import platform

def call_and_check_error(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        assert result[0] == 0, f"cuda-python error, {result[0]}"
        if len(result) == 2:
            return result[1]
        else:
            assert len(result) == 1, "Unsupported function call"
            return None
    return wrapper

def getCudaDeviceCount():
    if not CUDA_AVAILABLE:
        return 0
    return call_and_check_error(cudart.cudaGetDeviceCount)()

def getPCIBusIdByCudaDeviceOrdinal(cuda_device_id):
    '''
    cuda_device_id 在 0 ~ getCudaDeviceCount() - 1 之间取值，受到 CUDA_VISIBLE_DEVICES 影响
    '''
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    device = call_and_check_error(cuda.cuDeviceGet)(cuda_device_id)
    result = call_and_check_error(cuda.cuDeviceGetPCIBusId)(100, device)
    return result.decode("ascii").split('\0')[0]

if __name__ == "__main__":
    # macOS doesn't support /dev/dri paths - use CPU
    if platform.system() == "Darwin":
        print("cpu")
        exit(0)
    
    if os.environ.get("MINESTUDIO_GPU_RENDER", 0) != '1':
        print("cpu")
        exit(0)
    
    # If CUDA is not available, default to CPU
    if not CUDA_AVAILABLE:
        print("cpu")
        exit(0)
    
    try:
        call_and_check_error(cuda.cuInit)(0)
    except:
        print("cpu")
        exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=str)
    args = parser.parse_args()
    index = int (args.index)
    num_cuda_devices = getCudaDeviceCount()
    if num_cuda_devices == 0:
        device = "cpu"
    else:
        cuda_device_id = index % num_cuda_devices
        pci_bus_id = getPCIBusIdByCudaDeviceOrdinal(cuda_device_id)
        # /dev/dri is Linux-specific - check if path exists before using
        device_path = f"/dev/dri/by-path/pci-{pci_bus_id.lower()}-card"
        if os.path.exists(device_path):
            device = os.path.realpath(device_path)
        else:
            # Fallback to CPU if device path doesn't exist
            device = "cpu"
    print(device)