import ctypes
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]


class LlaisysQwen2Model(ctypes.Structure):
    pass


class LlaisysQwen2Weights(ctypes.Structure):
    pass


def load_qwen2(LIB):
    LIB.llaisysQwen2ModelCreate.restype = ctypes.POINTER(LlaisysQwen2Model)
    LIB.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

    LIB.llaisysQwen2ModelDestroy.restype = None
    LIB.llaisysQwen2ModelDestroy.argtypes = [ctypes.POINTER(LlaisysQwen2Model)]

    LIB.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)
    LIB.llaisysQwen2ModelWeights.argtypes = [ctypes.POINTER(LlaisysQwen2Model)]

    LIB.llaisysQwen2ModelInfer.restype = ctypes.c_int64
    LIB.llaisysQwen2ModelInfer.argtypes = [
        ctypes.POINTER(LlaisysQwen2Model),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
    ]

    return LIB


__all__ = [
    "LlaisysQwen2Meta",
    "LlaisysQwen2Model",
    "LlaisysQwen2Weights",
    "load_qwen2",
]


