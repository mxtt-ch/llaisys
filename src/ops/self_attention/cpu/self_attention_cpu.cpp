#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

template <typename T>
void self_attention_(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     size_t seq_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, size_t total_len, float scale) {
    const T* q_ptr = reinterpret_cast<const T*>(q);
    const T* k_ptr = reinterpret_cast<const T*>(k);
    const T* v_ptr = reinterpret_cast<const T*>(v);
    T* attn_val_ptr = reinterpret_cast<T*>(attn_val);
    
    // For each sequence position and attention head
    for (size_t seq = 0; seq < seq_len; seq++) {
        for (size_t head = 0; head < nhead; head++) {
            // Precompute kv head mapping for grouped-query attention
            const size_t heads_per_kv_head = nhead / nkvhead;
            const size_t kv_head = head / heads_per_kv_head;
            // Compute attention scores: Q * K^T * scale
            std::vector<float> attention_scores(total_len);
            
            for (size_t kv_seq = 0; kv_seq < total_len; kv_seq++) {
                float score = 0.0f;
                for (size_t dim = 0; dim < d; dim++) {
                    const T q_val = q_ptr[seq * nhead * d + head * d + dim];
                    const T k_val = k_ptr[kv_seq * nkvhead * d + kv_head * d + dim];
                    const float q_float = llaisys::utils::cast<float>(q_val);
                    const float k_float = llaisys::utils::cast<float>(k_val);
                    score += q_float * k_float;
                }
                attention_scores[kv_seq] = score * scale;
            }
            
            // Apply causal mask matching PyTorch's tril(diagonal=S-L) behavior
            // L = seq_len, S = total_len
            // tril(diagonal=S-L) creates a lower triangular mask with diagonal offset S-L
            // The mask allows attention to positions kv_seq <= seq + (S-L)
            const int diagonal_offset = static_cast<int>(total_len) - static_cast<int>(seq_len);
            for (size_t kv_seq = 0; kv_seq < total_len; kv_seq++) {
                // Mask positions that are not allowed by the causal mask
                // Allow kv_seq <= seq + diagonal_offset
                if (static_cast<int>(kv_seq) > static_cast<int>(seq) + diagonal_offset) {
                    attention_scores[kv_seq] = -std::numeric_limits<float>::infinity();
                }
            }
            
            // Softmax with numerical stability
            float max_score = *std::max_element(attention_scores.begin(), attention_scores.end());
            float sum_exp = 0.0f;
            for (size_t kv_seq = 0; kv_seq < total_len; kv_seq++) {
                attention_scores[kv_seq] = std::exp(attention_scores[kv_seq] - max_score);
                sum_exp += attention_scores[kv_seq];
            }
            
            for (size_t kv_seq = 0; kv_seq < total_len; kv_seq++) {
                attention_scores[kv_seq] /= sum_exp;
            }
            
            // Apply attention to values
            for (size_t dim = 0; dim < dv; dim++) {
                float weighted_sum = 0.0f;
                for (size_t kv_seq = 0; kv_seq < total_len; kv_seq++) {
                    const T v_val = v_ptr[kv_seq * nkvhead * dv + kv_head * dv + dim];
                    const float v_float = llaisys::utils::cast<float>(v_val);
                    weighted_sum += attention_scores[kv_seq] * v_float;
                }
                const size_t out_idx = seq * nhead * dv + head * dv + dim;
                attn_val_ptr[out_idx] = llaisys::utils::cast<T>(weighted_sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
    void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                        llaisysDataType_t type, size_t seq_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, size_t total_len, float scale) {
        switch (type) {
            case LLAISYS_DTYPE_F32:
                return self_attention_<float>(attn_val, q, k, v, seq_len, nhead, nkvhead, d, dv, total_len, scale);
            case LLAISYS_DTYPE_BF16:
                return self_attention_<llaisys::bf16_t>(attn_val, q, k, v, seq_len, nhead, nkvhead, d, dv, total_len, scale);
            case LLAISYS_DTYPE_F16:
                return self_attention_<llaisys::fp16_t>(attn_val, q, k, v, seq_len, nhead, nkvhead, d, dv, total_len, scale);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
} // namespace llaisys::ops::cpu
