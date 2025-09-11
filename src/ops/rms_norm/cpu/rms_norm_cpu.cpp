#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(std::byte *out, const std::byte *in, const std::byte *weight,
               size_t batch_size, size_t hidden_size, float eps) {
    const T* in_ptr = reinterpret_cast<const T*>(in);
    const T* weight_ptr = reinterpret_cast<const T*>(weight);
    T* out_ptr = reinterpret_cast<T*>(out);
    
    for (size_t b = 0; b < batch_size; b++) {
        float sum_squares = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            T val = in_ptr[b * hidden_size + i];
            float val_float = llaisys::utils::cast<float>(val);
            sum_squares += val_float * val_float;
        }
        
        float rms = std::sqrt(sum_squares / hidden_size + eps);
        float rms_inv = 1.0f / rms;
        
        for (size_t i = 0; i < hidden_size; i++) {
            T in_val = in_ptr[b * hidden_size + i];
            T weight_val = weight_ptr[i];
            
            float in_float = llaisys::utils::cast<float>(in_val);
            float weight_float = llaisys::utils::cast<float>(weight_val);
            float normalized = in_float * rms_inv;
            float result = weight_float * normalized;
            
            out_ptr[b * hidden_size + i] = llaisys::utils::cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t batch_size, size_t hidden_size, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(out, in, weight, batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(out, in, weight, batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(out, in, weight, batch_size, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
