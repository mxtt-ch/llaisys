#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void rearrange_(std::byte *out, const std::byte *in, size_t size) {
    const T* in_ptr = reinterpret_cast<const T*>(in);
    T* out_ptr = reinterpret_cast<T*>(out);

    for (size_t i = 0; i < size; i++) {
        out_ptr[i] = in_ptr[i];
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_<float>(out, in, size);
    case LLAISYS_DTYPE_BF16:
        return rearrange_<llaisys::bf16_t>(out, in, size);
    case LLAISYS_DTYPE_F16:
        return rearrange_<llaisys::fp16_t>(out, in, size);
    case LLAISYS_DTYPE_I64:
        return rearrange_<int64_t>(out, in, size);
    case LLAISYS_DTYPE_I32:
        return rearrange_<int32_t>(out, in, size);
    case LLAISYS_DTYPE_I16:
        return rearrange_<int16_t>(out, in, size);
    case LLAISYS_DTYPE_I8:
        return rearrange_<int8_t>(out, in, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
