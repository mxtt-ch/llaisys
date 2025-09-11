#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(std::byte *out, const std::byte *in, const std::byte *pos_ids,
           size_t seq_len, size_t nhead, size_t d, float theta) {
    const T* in_ptr = reinterpret_cast<const T*>(in);
    const int64_t* pos_ids_ptr = reinterpret_cast<const int64_t*>(pos_ids);
    T* out_ptr = reinterpret_cast<T*>(out);
    
    size_t d_half = d / 2;
    
    for (size_t seq = 0; seq < seq_len; seq++) {
        int64_t pos = pos_ids_ptr[seq];
        
        for (size_t head = 0; head < nhead; head++) {
            for (size_t j = 0; j < d_half; j++) {
                float angle = pos / std::pow(theta, 2.0f * j / d);
                float cos_angle = std::cos(angle);
                float sin_angle = std::sin(angle);
                
                size_t a_idx = seq * nhead * d + head * d + j;
                size_t b_idx = seq * nhead * d + head * d + j + d_half;
                
                T a_val = in_ptr[a_idx];
                T b_val = in_ptr[b_idx];
                
                float a_float = llaisys::utils::cast<float>(a_val);
                float b_float = llaisys::utils::cast<float>(b_val);
                
                float a_new = a_float * cos_angle - b_float * sin_angle;
                float b_new = b_float * cos_angle + a_float * sin_angle;
                
                out_ptr[a_idx] = llaisys::utils::cast<T>(a_new);
                out_ptr[b_idx] = llaisys::utils::cast<T>(b_new);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t nhead, size_t d, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(out, in, pos_ids, seq_len, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(out, in, pos_ids, seq_len, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(out, in, pos_ids, seq_len, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
