import torch
import triton
import triton.language as tl
import time


@triton.jit
def get_q_mask_triton(
    Q, K, anchors, true_coords, true_counts, diff,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_az, stride_ah, stride_an, stride_ad,
    stride_cz, stride_ch, stride_cn, stride_cd,
    stride_tcz, stride_tch, stride_tcn, stride_tcd,
    B, H, N: tl.constexpr, D: tl.constexpr,
    BLOCK_SZ: tl.constexpr, step: tl.constexpr,
    qk_scale: tl.constexpr,
    dtype: tl.constexpr,
):
    # Get thread-block indices
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Compute tensor offsets
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    anchors_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    coords_offset = off_z.to(tl.int64) * stride_cz + off_h.to(tl.int64) * stride_ch
    counts_offset = off_z.to(tl.int64) * stride_tcz + off_h.to(tl.int64) * stride_tch

    # Create block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_SZ * step, 0),
        block_shape=(BLOCK_SZ * step, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_SZ),
        order=(0, 1),
    )
    anchors_block_ptr = tl.make_block_ptr(
        base=anchors + anchors_offset,
        shape=(N, 1),
        strides=(stride_an, stride_ad),
        offsets=(start_m * BLOCK_SZ * step, 0),
        block_shape=(BLOCK_SZ * step, 1),
        order=(1, 0),
    )
    coords_block_ptr = tl.make_block_ptr(
        base=true_coords + coords_offset,
        shape=(N // BLOCK_SZ // step, N),
        strides=(stride_cn, stride_cd),
        offsets=(start_m, 0),
        block_shape=(1, BLOCK_SZ),
        order=(1, 0),
    )
    counts_block_ptr = tl.make_block_ptr(
        base=true_counts + counts_offset,
        shape=(N // BLOCK_SZ // step, 1),
        strides=(stride_tcn, stride_tcd),
        offsets=(start_m, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )

    # Load and process queries
    q = tl.load(Q_block_ptr)
    q = tl.reshape(q, (step, BLOCK_SZ, D))          # [step, BS, D]
    q = tl.sum(q, axis=1, keep_dims=False) / BLOCK_SZ  # [step, D]
    q = q.to(dtype)

    # Load and process anchors
    anchors = tl.load(anchors_block_ptr)
    anchors = tl.reshape(anchors, (step, BLOCK_SZ, 1))  # [step, BS, 1]
    anchors = tl.sum(anchors, axis=1, keep_dims=False) / BLOCK_SZ  # [step, 1]
    anchors = anchors.to(dtype)

    qk_scale = qk_scale * 1.44269504

    # Initialize indices and counters
    count = tl.zeros((1, 1), dtype=tl.int32)           # Record the number of valid indices
    cur_mask = tl.zeros((step, BLOCK_SZ), dtype=tl.int32)
    f_mask = tl.zeros((1, BLOCK_SZ), dtype=tl.int1)

    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SZ))
    for i in range(BLOCK_SZ, start_m * BLOCK_SZ * step - BLOCK_SZ, BLOCK_SZ):
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * qk_scale                  # [step, BS]
        qk = qk.to(dtype)

        # Compute mask
        cur_mask = ((qk + diff) >= anchors).to(tl.int32)             # [step, BS]
        f_mask = (tl.sum(cur_mask, axis=0, keep_dims=True).to(tl.int32) > 0)  # [1, BS]

        # Generate global indices
        global_indices = tl.arange(0, BLOCK_SZ)[None, :] + i         # [1, BS]

        # Compute valid mask and count
        valid_mask = f_mask > 0                                      # [1, BS]
        valid_count = tl.sum(valid_mask.to(tl.int32))                # scalar

        # Create valid positions: keep global_indices where valid_mask is True, else set to N
        valid_positions = tl.where(valid_mask, global_indices, N)    # [1, BS]

        # Sort valid_positions along the last dimension
        valid_positions = tl.sort(valid_positions)

        count = count + valid_count

        # Store results
        tl.store(coords_block_ptr, valid_positions)
        coords_block_ptr = tl.advance(coords_block_ptr, (0, valid_count))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SZ))

    # Store indices and counts
    tl.store(counts_block_ptr, count)


def get_q_mask(
    Q, K, anchors, block_size, step, diff,
    B, H, seq, h_dim, qk_scale
):
    assert Q.is_contiguous(), "Q should be contiguous"
    assert K.is_contiguous(), "K should be contiguous"
    assert block_size in [16, 32, 64, 128, 256], "BLOCK_SIZE must be a power of 2 and ≥16"
    assert step in [16, 32], "step must be ≥16"
    assert seq % (block_size * step) == 0

    grid = (triton.cdiv(seq, block_size * step), B * H, 1)
    true_coords = torch.zeros(
        (B, H, seq // block_size // step, seq),
        dtype=torch.int32, device=Q.device
    ).contiguous()
    true_counts = torch.zeros(
        (B, H, seq // block_size // step, 1),
        dtype=torch.int32, device=Q.device
    ).contiguous()

    get_q_mask_triton[grid](
        Q, K, anchors, true_coords, true_counts, diff,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        anchors.stride(0), anchors.stride(1), anchors.stride(2), anchors.stride(3),
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        B, H, seq, h_dim,
        BLOCK_SZ=block_size, step=step,
        qk_scale=qk_scale,
        dtype=tl.float16 if Q.dtype == torch.float16 else tl.bfloat16
    )
    return true_coords, true_counts


def get_mask_block(
    Q, K, anchors,
    block_size_N, block_size_M, step, diff,
    B, H, seq, h_dim, sm_scale,
    device, return_sparsity_ratio: bool = False
):
    torch.cuda.synchronize()
    computer_mask_time = time.time()

    c_n = (seq // block_size_N + step - 1) // step
    padding_size = (step - ((seq // block_size_N) % step)) % step

    true_coords = torch.zeros(B, H, c_n, seq, device=device, dtype=torch.int32)
    true_counts = torch.zeros(B, H, c_n, 1, device=device, dtype=torch.int32)

    mean_Q = torch.mean(Q.view(B, H, -1, block_size_N, h_dim), dim=-2)         # (B, H, seq//block_size_N, h_dim)
    anchors = anchors.unsqueeze(-1)                                            # (B, H, seq, 1)
    mean_anchors = torch.mean(
        anchors.view(B, H, -1, block_size_N, 1), dim=-2
    )                                                                          # (B, H, seq//block_size_N, 1)

    compress_q_attn = (
        torch.matmul(mean_Q, K.transpose(-1, -2) * sm_scale) + diff
    ) >= mean_anchors                                                           # (B, H, seq//block_size_N, seq)

    if padding_size > 0:
        compress_q_attn = torch.nn.functional.pad(
            compress_q_attn, (0, 0, 0, padding_size)
        ).contiguous()

    compress_q_mask = torch.sum(
        compress_q_attn.view(B, H, -1, step, seq),
        dim=-2,
        keepdim=False
    ) > 0                                                                       # (B, H, seq//block_size_N//step, seq)

    torch.cuda.synchronize()
    computer_mask_time_cost = (time.time() - computer_mask_time) * 1000

    fz_time = time.time()
    total_valid_positions = 0
    selected_positions = 0

    for b in range(B):
        for h in range(H):
            for i in range(1, c_n):
                end_idx = i * block_size_N * step - block_size_N
                true_mask = compress_q_mask[b, h, i, block_size_N:end_idx]
                true_indices = torch.where(true_mask)[0] + block_size_N
                count = true_indices.size(0)
                selected_positions += count
                total_valid_positions += end_idx - block_size_N
                if count > 0:
                    true_coords[b, h, i, :count] = true_indices[:count]
                    true_counts[b, h, i, 0] = count

    torch.cuda.synchronize()
    fz_time_cost = (time.time() - fz_time) * 1000

    if total_valid_positions == 0:
        print("anchor ratio: 0")
        sparsity_ratio = 0.0
    else:
        print(f"anchor ratio: {selected_positions / total_valid_positions}")
        sparsity_ratio = selected_positions / total_valid_positions
    print(f"compute time: {computer_mask_time_cost} ms")
    print(f"assign time: {fz_time_cost} ms")

    if return_sparsity_ratio:
        return true_coords, true_counts, sparsity_ratio
    return true_coords, true_counts


def get_q_mask_test():
    B, H, N, D = 1, 32, 1024 * 128, 128
    BLOCK_SZ = 64
    step = 16
    diff = 0
    MAX_TRUE = BLOCK_SZ
    sm_scale = 1
    torch.cuda.manual_seed_all(342)

    Q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16).contiguous()
    K = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16).contiguous()
    anchors = torch.randn((B, H, N, 1), device="cuda", dtype=torch.float16).contiguous() + 1.3
    sm_scale = 1 / (D ** 0.5)

    for i in range(3):
        torch.cuda.synchronize()
        triton_get_mask = time.time()
        true_coords, true_counts = get_q_mask(Q, K, anchors, BLOCK_SZ, step, diff, B, H, N, D, MAX_TRUE, sm_scale)
        torch.cuda.synchronize()
        triton_get_mask_cost = (time.time() - triton_get_mask) * 1000
        print(f"triton_get_mask cost: {triton_get_mask_cost} ms")
        print(f"true_coords shape: {true_coords.shape}")
        print(f"true_counts shape: {true_counts.shape}")
        torch.cuda.synchronize()


if __name__ == "__main__":
    import numpy as np
    get_q_mask_test()
