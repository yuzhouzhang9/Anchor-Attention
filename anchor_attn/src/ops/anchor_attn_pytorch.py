import torch
import math
from .anchor import get_anchors
from .fast_gather import get_compress_attn_weight


def anchor_attn_torch_ops(Q, K, V, 
                          softmax_scale=None, 
                          block_size=128, 
                          step=16, 
                          diff_anchor=12):
    """
    Anchor Attention with sparse windowed FlashAttention.
    
    Q, K, V: [B, H, T, D] --> will be transposed to [B, T, H, D] for flash_attention
    """
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(Q.shape[-1])
    
    B, H, T, D = Q.shape
    pad = T % block_size
    T_ = T - pad

    # Output tensor [B, H, T, D]
    O1 = torch.zeros_like(Q)

    # Transpose to [B, T, H, D] for flash_attention
    Q_t = Q.transpose(1, 2).contiguous()  # [B, T, H, D]
    K_t = K.transpose(1, 2).contiguous()
    V_t = V.transpose(1, 2).contiguous()

    from .flash_attn_triton import triton_flash_attention
    # Handle the first block
    O1[:, :, :block_size, :] = triton_flash_attention(
        Q_t[:, :block_size], K_t[:, :block_size], V_t[:, :block_size]
    ).transpose(1, 2)  # back to [B, H, T, D]

    # Handle the padded part
    if pad:
        from .flash_attn_triton import triton_flash_attention
        O1[:, :, -pad:, :] = triton_flash_attention(
            Q_t[:, -pad:], K_t, V_t,
            attention_mask=None, query_length=pad, is_causal=True
        ).transpose(1, 2)

    if pad:
        Q = Q[:, :, :-pad, :]
        K = K[:, :, :-pad, :]
        V = V[:, :, :-pad, :]
        Q_t = Q_t[:, :-pad]
        K_t = K_t[:, :-pad]
        V_t = V_t[:, :-pad]
        T_ = Q.shape[2]

    # Anchor attention: [B, H, num_block, 1]
    Q = Q.contiguous()
    K = K.contiguous()
    anchor_attention = get_anchors(Q, K) * scale  # placeholder
    anchor_attention = torch.mean(anchor_attention.view(B, H, -1, block_size, 1), dim=-2)

    tot = 0
    used = 0

    for i in range(B):
        for j in range(H):
            for k in range(1, T_ // block_size, step):
                l, r = k, min(k + step, T_ // block_size)

                compress_attn_weight = get_compress_attn_weight(
                    Q[i, j, k * block_size:r * block_size], 
                    K[i, j, :k * block_size], 
                    block_size
                )  # [step * block, past_block * block]

                mask = (compress_attn_weight + diff_anchor) >= anchor_attention[i, j, l:r]
                mask = torch.sum(mask, dim=0) > 0  # [tokens]

                tot += k * block_size
                used += mask.sum()

                t_key = K[i, j, :k * block_size][mask]
                t_value = V[i, j, :k * block_size][mask]
                t_key = torch.cat([t_key, K[i, j, k * block_size:k * block_size + (r - l) * block_size]], dim=0)
                t_value = torch.cat([t_value, V[i, j, k * block_size:k * block_size + (r - l) * block_size]], dim=0)

                q_block = Q[i:i+1, j:j+1, k * block_size:k * block_size + (r - l) * block_size]  # [1, 1, T, D]
                t_key = t_key.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
                t_value = t_value.unsqueeze(0).unsqueeze(0)

                q_block = q_block.transpose(1, 2)
                t_key = t_key.transpose(1, 2)
                t_value = t_value.transpose(1, 2)
                from .flash_attn_triton import triton_flash_attention
                O_block = triton_flash_attention(
                    q_block, t_key, t_value,
                    attention_mask=None, query_length=(r - l) * block_size, is_causal=True
                ).transpose(1, 2)  # [1, 1, T, D]

                O1[i:i+1, j:j+1, k * block_size:k * block_size + (r - l) * block_size] = O_block

    return O1


if __name__ == "__main__":
    B, H, T, D = 2, 4, 256, 64
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)

    output = anchor_attn_torch_ops(Q, K, V)
    print("Output shape:", output.shape)
    print("Output sample:", output[0, 0, :5, :5])
