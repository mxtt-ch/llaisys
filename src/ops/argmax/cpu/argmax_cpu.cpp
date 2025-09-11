#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>

template <typename T>
void argmax_(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t size) {
    const T* vals_ptr = reinterpret_cast<const T*>(vals);
    T* max_val_ptr = reinterpret_cast<T*>(max_val);
    int64_t* max_idx_ptr = reinterpret_cast<int64_t*>(max_idx);
    
    if (size == 0) {
        *max_val_ptr = llaisys::utils::cast<T>(0.0f);
        *max_idx_ptr = 0;
        return;
    }
    
    T max_value = vals_ptr[0];
    int64_t max_index = 0;
    
    for (size_t i = 1; i < size; i++) {
        T current_val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            current_val = vals_ptr[i];
            float current_float = llaisys::utils::cast<float>(current_val);
            float max_float = llaisys::utils::cast<float>(max_value);
            if (current_float > max_float) {
                max_value = current_val;
                max_index = static_cast<int64_t>(i);
            }
        } else {
            current_val = vals_ptr[i];
            if (current_val > max_value) {
                max_value = current_val;
                max_index = static_cast<int64_t>(i);
            }
        }
    }
    
    *max_val_ptr = max_value;
    *max_idx_ptr = max_index;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_BF16:
        return argmax_<llaisys::bf16_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_F16:
        return argmax_<llaisys::fp16_t>(max_idx, max_val, vals, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
