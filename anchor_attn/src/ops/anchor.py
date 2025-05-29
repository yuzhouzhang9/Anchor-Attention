import triton  # Import Triton library for writing efficient GPU kernels
import triton.language as tl  # Import Triton language module for low-level operations (e.g., tensor loads, dot products)
import torch  # Import PyTorch for tensor operations and GPU computation

@triton.jit  # Use Triton's just-in-time compilation decorator to compile the function into a GPU kernel
def get_anchors_triton(Q, K, Anchors,  # Input tensors: Query (Q), Key (K), and output Anchors
                       stride_qz, stride_qh, stride_qm, stride_qd,  # Strides for Q: batch (z), head (h), sequence (m), dimension (d)
                       stride_kz, stride_kh, stride_kn, stride_kd,  # Strides for K: batch (z), head (h), sequence (n), dimension (d)
                       stride_az, stride_ah, stride_an, stride_ad,  # Strides for Anchors: batch (z), head (h), sequence (n), dimension (d)
                       B, H, N, D: tl.constexpr,  # Constant parameters: batch size (B), number of heads (H), sequence length (N), head dimension (D)
                       BLOCK_SZ: tl.constexpr):  # Constant parameter: block size (BLOCK_SZ) for block-wise computation
    # Compute the row-block index in Q's sequence dimension
    start_m = tl.program_id(0)
    # Compute the combined batch * head index
    off_hz = tl.program_id(1)
    # Derive batch index from off_hz
    off_z = off_hz // H
    # Derive head index from off_hz
    off_h = off_hz % H

    # Compute memory offset for Q and K based on batch and head
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    # Compute memory offset for Anchors based on batch and head
    anchor_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah

    # Define the range of Q row indices for this block
    offs_m = start_m * BLOCK_SZ + tl.arange(0, BLOCK_SZ)
    # Define the range of K column indices for a full block
    offs_n = tl.arange(0, BLOCK_SZ)

    # Create a block pointer for loading the current block of Q
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_SZ, 0),
        block_shape=(BLOCK_SZ, D),
        order=(1, 0),  # Row-major order
    )
    
    # Create a block pointer for loading the initial block of K (first BLOCK_SZ columns)
    K_init_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(D, N),  # Note K is in transposed form
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_SZ),
        order=(0, 1),  # Column-major order
    )

    # Create a block pointer for storing the output anchors
    Anchors_block_ptr = tl.make_block_ptr(
        base=Anchors + anchor_offset,
        shape=(N, 1),
        strides=(stride_an, stride_ad),
        offsets=(start_m * BLOCK_SZ, 0),
        block_shape=(BLOCK_SZ, 1),
        order=(1, 0),  # Row-major order
    )

    # Load the current Q block into registers
    q = tl.load(Q_block_ptr)
    # Load the initial K block into registers
    k = tl.load(K_init_block_ptr)

    # Compute dot-product between q and k: shape (BLOCK_SZ, BLOCK_SZ)
    qk = tl.dot(q, k)

    # Build causal mask to prevent attending to future tokens (upper-triangle mask)
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    # Apply mask: future positions get a large negative value so they are zero in softmax
    qk += tl.where(causal_mask, 0, -1.0e6)

    # Compute the maximum score along each row of qk
    qk_max = tl.max(qk, axis=1)

    # Create a block pointer for loading the windowed K block for this Q block
    K_window_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, start_m * BLOCK_SZ),
        block_shape=(D, BLOCK_SZ),
        order=(0, 1),
    )

    # Load the windowed K block and compute dot-product with q
    k_window = tl.load(K_window_block_ptr)
    qk2 = tl.dot(q, k_window)

    # Build causal mask for the windowed block
    causal_mask = offs_m[:, None] >= (start_m * BLOCK_SZ + offs_n[None, :])
    # Apply mask
    qk2 += tl.where(causal_mask, 0, -1.0e6)

    # Compute the maximum score along rows of qk2
    qk2_max = tl.max(qk2, axis=1)

    # Combine the initial block max and window block max as the anchor
    anchors = tl.maximum(qk_max, qk2_max)
    anchors = tl.reshape(anchors, (BLOCK_SZ, 1))

    # Store the computed anchors back to output
    tl.store(Anchors_block_ptr, anchors.to(Anchors.type.element_ty))

@triton.jit  # Compile into a second kernel with identical logic but named differently
def get_anchors_triton_with_attn_score(Q, K, Anchors,  # Input tensors
                       stride_qz, stride_qh, stride_qm, stride_qd,
                       stride_kz, stride_kh, stride_kn, stride_kd,
                       stride_az, stride_ah, stride_an, stride_ad,
                       B, H, N, D: tl.constexpr,
                       BLOCK_SZ: tl.constexpr):
    # Implementation identical to above get_anchors_triton
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    anchor_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    offs_m = start_m * BLOCK_SZ + tl.arange(0, BLOCK_SZ)
    offs_n = tl.arange(0, BLOCK_SZ)
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N, D), strides=(stride_qm, stride_qd), offsets=(start_m * BLOCK_SZ, 0), block_shape=(BLOCK_SZ, D), order=(1, 0))
    K_init_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(D, N), strides=(stride_kd, stride_kn), offsets=(0, 0), block_shape=(D, BLOCK_SZ), order=(0, 1))
    Anchors_block_ptr = tl.make_block_ptr(base=Anchors + anchor_offset, shape=(N, 1), strides=(stride_an, stride_ad), offsets=(start_m * BLOCK_SZ, 0), block_shape=(BLOCK_SZ, 1), order=(1, 0))
    q = tl.load(Q_block_ptr)
    k = tl.load(K_init_block_ptr)
    qk = tl.dot(q, k)
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    qk += tl.where(causal_mask, 0, -1.0e6)
    qk_max = tl.max(qk, axis=1)
    K_window_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(D, N), strides=(stride_kd, stride_kn), offsets=(0, start_m * BLOCK_SZ), block_shape=(D, BLOCK_SZ), order=(0, 1))
    k_window = tl.load(K_window_block_ptr)
    qk2 = tl.dot(q, k_window)
    causal_mask = offs_m[:, None] >= (start_m * BLOCK_SZ + offs_n[None, :])
    qk2 += tl.where(causal_mask, 0, -1.0e6)
    qk2_max = tl.max(qk2, axis=1)
    anchors = tl.maximum(qk_max, qk2_max)
    anchors = tl.reshape(anchors, (BLOCK_SZ, 1))
    tl.store(Anchors_block_ptr, anchors.to(Anchors.type.element_ty))

# Python interface for Triton kernel invocation
def get_anchors(Q, K, BLOCK_SIZE=256):
    assert Q.is_contiguous(), "Q must be contiguous for performance"
    assert K.is_contiguous(), "K must be contiguous for performance"
    assert type(Q) == type(K), "Q and K must have the same dtype"
    B, H, N, D = Q.shape
    num_stages = 3  # Triton kernel pipeline stages
    num_warps = 8   # Number of warps per Triton kernel

    # Allocate output anchors tensor
    anchors = torch.empty((B, H, N, 1), dtype=Q.dtype, device=Q.device)
    # Define Triton launch grid: (number of row-blocks, batch*head, 1)
    grid = (((N + BLOCK_SIZE - 1) // BLOCK_SIZE), B * H, 1)

    # Launch the Triton kernel
    get_anchors_triton[grid](
        Q, K, anchors,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        anchors.stride(0), anchors.stride(1), anchors.stride(2), anchors.stride(3),
        B, H, N, D,
        BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return anchors

# PyTorch fallback implementation for comparison
def create_causal_mask(seq_length):
    """Generate a causal mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
    return mask

def get_anchors_pytorch(Q, K, BLOCK_SIZE=256):
    # Compute full attention scores
    qk = torch.matmul(Q, K.transpose(-1, -2))
    # Apply causal mask
    causal_mask = create_causal_mask(Q.size(-2)).to(Q.device)
    qk += causal_mask

    # Initial block max for first BLOCK_SIZE columns
    qk_init = qk[..., :BLOCK_SIZE]
    qk_max, _ = torch.max(qk_init, dim=-1, keepdim=True)

    # Sliding window max over blocks
    window_qk = torch.full_like(qk_max, -1e6)
    for i in range(0, Q.size(-2), BLOCK_SIZE):
        block_scores = qk[..., i:i+BLOCK_SIZE, i:i+BLOCK_SIZE]
        cur_max, _ = torch.max(block_scores, dim=-1, keepdim=True)
        window_qk[..., i:i+BLOCK_SIZE, :] = cur_max

    # Combine initial block and window max as anchors
    return torch.maximum(qk_max, window_qk)

if __name__ == "__main__":
    torch.set_printoptions(profile="full")  # Print full tensor contents
    torch.manual_seed(123)
    B, H, N, D = 1, 1, 128*100, 128
    BS = 128

    Q = torch.randint(0, 2, (B, H, N, D), dtype=torch.float16, device="cuda")
    K = torch.randint(0, 2, (B, H, N, D), dtype=torch.float16, device="cuda")

    # Compute anchors via PyTorch and Triton
    a1 = get_anchors_pytorch(Q, K, BS)
    a2 = get_anchors(Q, K, BS)
    print("[info] precision diff is", torch.max(torch.abs(a1 - a2)))

    # Benchmark Triton vs PyTorch implementations
    min_time_triton = float('inf')
    min_time_torch = float('inf')
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        start_evt.record()
        get_anchors(Q, K, BS)
        end_evt.record()
        torch.cuda.synchronize()
        min_time_triton = min(min_time_triton, start_evt.elapsed_time(end_evt))

    torch.cuda.synchronize()
    for _ in range(10):
        start_evt.record()
        get_anchors_pytorch(Q, K, BS)
        end_evt.record()
        torch.cuda.synchronize()
        min_time_torch = min(min_time_torch, start_evt.elapsed_time(end_evt))

    print("[info] speedup is", min_time_torch / min_time_triton)
