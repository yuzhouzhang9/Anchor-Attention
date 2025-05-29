import torch
import triton
import triton.language as tl
import time 
@triton.jit
def get_q_mask_triton(Q, K, anchors, c_q_mask, diff,
                      stride_qz, stride_qh, stride_qm, stride_qd,
                      stride_kz, stride_kh, stride_kn, stride_kd, 
                      stride_az, stride_ah, stride_an, stride_ad,
                      stride_mz, stride_mh, stride_mn, stride_md,
                      B, H, N: tl.constexpr, D: tl.constexpr,
                      BLOCK_SZ: tl.constexpr, step: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    anchors_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    mask_offset = off_z.to(tl.int64) * stride_mz + off_h.to(tl.int64) * stride_mh

    # Query block pointer
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_SZ * step, 0),
        block_shape=(BLOCK_SZ * step, D),
        order=(1, 0),
    )
    # Key block pointer
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_SZ),
        order=(0, 1),
    )
    # Anchors block pointer
    anchors_block_ptr = tl.make_block_ptr(
        base=anchors + anchors_offset,
        shape=(N, 1),
        strides=(stride_an, stride_ad),
        offsets=(start_m * BLOCK_SZ * step, 0),
        block_shape=(BLOCK_SZ * step, 1),
        order=(1, 0),
    )
    # Mask block pointer
    mask_block_ptr = tl.make_block_ptr(
        base=c_q_mask + mask_offset,
        shape=(N // BLOCK_SZ // step, N),
        strides=(stride_mn, stride_md),
        offsets=(start_m, 0),
        block_shape=(1, BLOCK_SZ),
        order=(1, 0),
    )

    # Load and process queries
    q = tl.load(Q_block_ptr)
    q = tl.reshape(q, (step, BLOCK_SZ, D))  # [step, BS, D]
    q = tl.sum(q, axis=1, keep_dims=False) / BLOCK_SZ  # [step, D]
    q = q.to(tl.float16)

    # Load and process anchors
    anchors = tl.load(anchors_block_ptr)
    anchors = tl.reshape(anchors, (step, BLOCK_SZ, 1))  # [step, BS, 1]
    anchors = tl.sum(anchors, axis=1, keep_dims=False) / BLOCK_SZ  # [step, 1]
    anchors = anchors.to(tl.float16)

    # Query block start index
    query_idx = start_m * BLOCK_SZ * step
    
    for i in range(0, N, BLOCK_SZ):
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)  # [step, BS]
        qk = qk.to(tl.float16)
        cur_mask = (qk + diff) > anchors  # [step, BS]
        f_mask = tl.sum(cur_mask, axis=0, keep_dims=True) > 0  # [1, BS]
        f_mask = f_mask.to(tl.int8)
        tl.store(mask_block_ptr, f_mask)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SZ))
        mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_SZ))
        
    for i in range(0, N, BLOCK_SZ):
        # Key block start index
        key_idx = i

        # Compute positional mask: |query_idx - key_idx| in (block_size, step * block_size - block_size)
        dist = tl.abs(query_idx - key_idx)
        pos_mask = (dist > BLOCK_SZ) & (dist < (step * BLOCK_SZ - BLOCK_SZ))  # [1]

        # Load keys
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)  # [step, BS]
        qk = qk.to(tl.float16)

        # Combine content-based and positional mask
        cur_mask = (qk + diff) > anchors  # [step, BS]
        cur_mask = cur_mask & pos_mask  # Broadcast pos_mask to [step, BS]
        f_mask = tl.sum(cur_mask, axis=0, keep_dims=True) > 0  # [1, BS]
        f_mask = f_mask.to(tl.int8)

        # Store mask
        tl.store(mask_block_ptr, f_mask)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SZ))
        mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_SZ))

def get_q_mask(Q, K, anchors, block_size, step, diff, B, H, seq, h_dim):
    assert Q.is_contiguous(), "Q should be contiguous"
    assert K.is_contiguous(), "K should be contiguous"
    assert block_size in [16, 32, 64, 128, 256], "BLOCK_SIZE should be the power of 2 and larger than 16"
    assert step >= 16, "step should be larger than 16"
    grid = (triton.cdiv(seq, block_size * step), B * H, 1)
    mask = torch.empty((B, H, seq // block_size // step, seq), dtype=torch.int8, device=Q.device).contiguous()
    get_q_mask_triton[grid](Q, K, anchors, mask, diff,
                            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                            anchors.stride(0), anchors.stride(1), anchors.stride(2), anchors.stride(3),
                            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
                            B, H, seq, h_dim, BLOCK_SZ=block_size, step=step)
    return mask.to(torch.bool)


def get_mask_block(Q, K, 
                   anchors, 
                   block_size_N, 
                   block_size_M, 
                   step, diff, 
                   B,
                   H, 
                   seq,
                   h_dim,
                   sm_scale,
                   device,
                   return_sparsity_ratio: bool = False):
    
    torch.cuda.synchronize()
    computer_mask_time = time.time()
    c_n = (seq//block_size_N + step - 1) // step
    padding_size = (step - ((seq//block_size_N) % step)) % step
    
    true_coords = torch.zeros(B,H,c_n,seq,device=device,dtype=torch.int32)
    true_counts = torch.zeros(B,H,c_n,1,device=device,dtype=torch.int32)
    # diff = 3
    # # q shape: (B, H, seq, h_dim)
    mean_Q = torch.mean(Q.view(B, H, -1, block_size_N, h_dim), dim=-2)  # Shape: (B, H, seq//block_size_N, h_dim)
    anchors = anchors.unsqueeze(-1)  # Shape: (B, H, seq, 1)
    mean_anchors = torch.mean(anchors.view(B, H, -1, block_size_N, 1), dim=-2)  # Shape: (B, H, seq//block_size_N, 1)

    # q_length/block, k_length
    compress_q_attn = (torch.matmul(mean_Q, K.transpose(-1, -2)*sm_scale) + diff) > mean_anchors  # Shape: (B, H, seq//block_size_N, seq)
    
    # print(compress_q_attn.max())
    # print(mean_anchors.max())
    
    # # q_length, k_length/block
    
    if padding_size > 0 :
        compress_q_mask = torch.nn.functional.pad(compress_q_attn, (0, 0, 0, padding_size)).contiguous()
    
    # # Compute masks
    compress_q_mask = torch.sum(
        compress_q_attn.view(B, H, -1, step, seq),
        dim=-2,
        keepdim=False
    ) > 0  # Shape: (B, H, seq//block_size_N//step, seq)
    torch.cuda.synchronize()
    computer_mask_time_cost = (time.time() - computer_mask_time) * 1000
    # print(f"ratio:{compress_q_mask.sum()/compress_q_mask.numel()}")
    
    # if True: # use key
    #     mean_K = torch.mean(K.view(B, H, -1, block_size_M, h_dim), dim=-2)  # Shape: (B, H, seq//block_size_M, h_dim)
    #     compress_k_attn = torch.matmul(Q, mean_K.transpose(-1, -2))  # Shape: (B, H, seq, seq//block_size_M)

    #     compress_k_mask = (torch.sum(
    #         ((compress_k_attn + diff) > anchors).view(B, H, -1, step * block_size_N, compress_k_attn.size(-1)),
    #         dim=-2,
    #         keepdim=False
    #     )>0).repeat_interleave(block_size_M, dim=-1)  # Shape: (B, H, seq//block_size_N, seq)
    #     print(compress_k_mask.numel() )
    #     print(compress_k_mask.sum())
    #     compress_q_mask = compress_q_mask | compress_k_mask 
    fz_time = time.time()
    total_valid_positions = 0
    selected_positions = 0
    for b in range(B):
        for h in range(H):
            for i in range(1,c_n):
                end_idx = i * block_size_N * step - block_size_N 
                # true_mask = torch.ones_like(compress_q_mask[b, h, i, block_size_N:end_idx], dtype=torch.bool) # Shape: (seq,)
                true_mask = compress_q_mask[b, h, i, block_size_N:end_idx]
                true_indices = torch.where(true_mask)[0] + block_size_N  # 获取 True 的坐标
                count = true_indices.size(0)  # True 的数量
                selected_positions += count
                total_valid_positions += end_idx - block_size_N
                # print(f"i:{i}")
                # print(f"count:{count}")
                # print(f"tot:{end_idx - block_size_N}")
                if count > 0:
                    true_coords[b, h, i, :count] = true_indices[:count]
                    true_counts[b, h, i, 0] = count
    torch.cuda.synchronize()
    fz_time_cost = (time.time() - fz_time) * 1000
    if total_valid_positions == 0:
        print(f"anchor ratio:{0}")
        sparsity_ratio = 0.0
    else:
        print(f"anchor ratio:{selected_positions / total_valid_positions}")
        sparsity_ratio = selected_positions / total_valid_positions
    print(f"computer time :{computer_mask_time_cost}ms")
    print(f"fuzhi time :{fz_time_cost}ms")
    if return_sparsity_ratio:
        return true_coords, true_counts, sparsity_ratio
    return true_coords, true_counts

def get_q_mask_test():
    B, H, N, D = 1, 32, 1024 * 64, 128  # Reduced N for testing
    BLOCK_SZ = 128
    step = 16
    diff = 0
    Q = torch.randn((B, H, N, D), device='cuda', dtype=torch.float16).contiguous()
    K = torch.randn((B, H, N, D), device='cuda', dtype=torch.float16).contiguous()
    anchors = torch.randn((B, H, N, 1), device='cuda', dtype=torch.float16).contiguous()
    get_q_mask(Q, K, anchors, BLOCK_SZ, step, diff, B, H, N, D)
    
    import time
    for i in range(3):
        torch.cuda.synchronize()
        triton_get_mask = time.time()
        mask = get_q_mask(Q, K, anchors, BLOCK_SZ, step, diff, B, H, N, D)
        torch.cuda.synchronize()
        triton_get_mask_cost = (time.time() - triton_get_mask) * 1000
        print(f"triton_get_mask cost:{triton_get_mask_cost}ms")
        
        
        # 
        torch.cuda.synchronize()
        pytorch_get_mask = time.time()
        get_mask_block(Q, K, anchors, BLOCK_SZ,BLOCK_SZ, step, diff,
                        B, H, N, D,
                        sm_scale = 10,
                        device = Q.device,
                       )
        torch.cuda.synchronize()
        pytorch_get_mask_cost = (time.time() - pytorch_get_mask) * 1000
        print(f"pytorch_get_mask cost:{pytorch_get_mask_cost}ms")
    # print(mask)
    # print(mask.shape)

if __name__ == "__main__":
    get_q_mask_test()