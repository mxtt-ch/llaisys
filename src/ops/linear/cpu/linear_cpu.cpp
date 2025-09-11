#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void linear_(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
             size_t batch_size, size_t in_features, size_t out_features, bool has_bias) {
    const T* in_ptr = reinterpret_cast<const T*>(in);
    const T* weight_ptr = reinterpret_cast<const T*>(weight);
    const T* bias_ptr = has_bias ? reinterpret_cast<const T*>(bias) : nullptr;
    T* out_ptr = reinterpret_cast<T*>(out);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float accumulator = 0.0f;
            if (has_bias && bias_ptr != nullptr) {
                accumulator = llaisys::utils::cast<float>(bias_ptr[o]);
            }
            
            for (size_t i = 0; i < in_features; i++) {
                T in_val = in_ptr[b * in_features + i];
                T weight_val = weight_ptr[o * in_features + i];
                
                float in_float = llaisys::utils::cast<float>(in_val);
                float weight_float = llaisys::utils::cast<float>(weight_val);
                accumulator += in_float * weight_float;
            }
            
            out_ptr[b * out_features + o] = llaisys::utils::cast<T>(accumulator);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features, bool has_bias) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(out, in, weight, bias, batch_size, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(out, in, weight, bias, batch_size, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(out, in, weight, bias, batch_size, in_features, out_features, has_bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
