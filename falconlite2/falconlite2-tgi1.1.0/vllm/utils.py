import enum
import uuid
import ctypes
from platform import uname

import psutil
import torch

class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        id = self.counter
        self.counter += 1
        return id

    def reset(self) -> None:
        self.counter = 0

def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    # """Returns the maximum shared memory per thread block in bytes."""
    # # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    # cudaDevAttrMaxSharedMemoryPerBlockOptin = 97  # pylint: disable=invalid-name
    # max_shared_mem = cuda_utils.get_device_attribute(
    #     cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu)
    # return int(max_shared_mem)

    libnames = ("libcuda.so", "libcuda.dylib", "nvcuda.dll", "cuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError(f"Could not load any of {' '.join(libnames)}")

    smem_size = ctypes.c_size_t()
    device = ctypes.c_size_t()

    cuda.cuDeviceGet(ctypes.byref(device), torch.cuda.current_device())
    cuda.cuInit(0)
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97  # pylint: disable=invalid-name
    assert not cuda.cuDeviceGetAttribute(
        ctypes.byref(smem_size), cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device)
    return smem_size.value


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()
