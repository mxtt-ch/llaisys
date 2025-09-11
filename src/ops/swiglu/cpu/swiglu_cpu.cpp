#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(std::byte *out, const std::byte *gate, const std::byte *up, size_t size) {
    const T* gate_ptr = reinterpret_cast<const T*>(gate);
    const T* up_ptr = reinterpret_cast<const T*>(up);
    T* out_ptr = reinterpret_cast<T*>(out);
    
    for (size_t i = 0; i < size; i++) {
        T gate_val = gate_ptr[i];
        T up_val = up_ptr[i];
        
        float gate_float = llaisys::utils::cast<float>(gate_val);
        float up_float = llaisys::utils::cast<float>(up_val);
        
        float denominator = 1.0f + std::exp(-gate_float);
        float result = up_float * (gate_float / denominator);
        
        out_ptr[i] = llaisys::utils::cast<T>(result);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(out, gate, up, size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(out, gate, up, size);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(out, gate, up, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
