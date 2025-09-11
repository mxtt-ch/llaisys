#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename OutT, typename WeightT>
void embedding_(std::byte *out, const std::byte *index, const std::byte *weight,
                size_t batch_size, size_t seq_len, size_t hidden_size) {
    const int64_t* index_ptr = reinterpret_cast<const int64_t*>(index);
    const WeightT* weight_ptr = reinterpret_cast<const WeightT*>(weight);
    OutT* out_ptr = reinterpret_cast<OutT*>(out);
    
    for (size_t i = 0; i < batch_size * seq_len; i++) {
        int64_t idx = index_ptr[i];
        if (idx < 0) {
            throw std::runtime_error("Embedding: negative index not supported");
        }
        
        const WeightT* src_row = weight_ptr + idx * hidden_size;
        OutT* dst_row = out_ptr + i * hidden_size;
        
        for (size_t j = 0; j < hidden_size; j++) {
            if constexpr (std::is_same_v<OutT, WeightT>) {
                dst_row[j] = src_row[j];
            } else if constexpr (std::is_same_v<OutT, llaisys::bf16_t> || std::is_same_v<OutT, llaisys::fp16_t> ||
                                std::is_same_v<WeightT, llaisys::bf16_t> || std::is_same_v<WeightT, llaisys::fp16_t>) {
                dst_row[j] = llaisys::utils::cast<OutT>(llaisys::utils::cast<float>(src_row[j]));
            } else {
                dst_row[j] = static_cast<OutT>(src_row[j]);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t out_type, llaisysDataType_t weight_type,
               size_t batch_size, size_t seq_len, size_t hidden_size) {
    if (out_type == LLAISYS_DTYPE_F32 && weight_type == LLAISYS_DTYPE_F32) {
        return embedding_<float, float>(out, index, weight, batch_size, seq_len, hidden_size);
    } else if (out_type == LLAISYS_DTYPE_F32 && weight_type == LLAISYS_DTYPE_BF16) {
        return embedding_<float, llaisys::bf16_t>(out, index, weight, batch_size, seq_len, hidden_size);
    } else if (out_type == LLAISYS_DTYPE_F32 && weight_type == LLAISYS_DTYPE_F16) {
        return embedding_<float, llaisys::fp16_t>(out, index, weight, batch_size, seq_len, hidden_size);
    } else if (out_type == LLAISYS_DTYPE_BF16 && weight_type == LLAISYS_DTYPE_F32) {
        return embedding_<llaisys::bf16_t, float>(out, index, weight, batch_size, seq_len, hidden_size);
    } else if (out_type == LLAISYS_DTYPE_BF16 && weight_type == LLAISYS_DTYPE_BF16) {
        return embedding_<llaisys::bf16_t, llaisys::bf16_t>(out, index, weight, batch_size, seq_len, hidden_size);
    } else if (out_type == LLAISYS_DTYPE_BF16 && weight_type == LLAISYS_DTYPE_F16) {
        return embedding_<llaisys::bf16_t, llaisys::fp16_t>(out, index, weight, batch_size, seq_len, hidden_size);
    } else if (out_type == LLAISYS_DTYPE_F16 && weight_type == LLAISYS_DTYPE_F32) {
        return embedding_<llaisys::fp16_t, float>(out, index, weight, batch_size, seq_len, hidden_size);
    } else if (out_type == LLAISYS_DTYPE_F16 && weight_type == LLAISYS_DTYPE_BF16) {
        return embedding_<llaisys::fp16_t, llaisys::bf16_t>(out, index, weight, batch_size, seq_len, hidden_size);
    } else if (out_type == LLAISYS_DTYPE_F16 && weight_type == LLAISYS_DTYPE_F16) {
        return embedding_<llaisys::fp16_t, llaisys::fp16_t>(out, index, weight, batch_size, seq_len, hidden_size);
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(out_type);
    }
}
} // namespace llaisys::ops::cpu
