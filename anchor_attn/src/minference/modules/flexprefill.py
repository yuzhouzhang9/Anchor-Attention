# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# 下面这行引用了相关论文中的代码实现：https://openreview.net/forum?id=OfjIlbelrT

import math
from typing import List, Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

# 这里尝试导入 flash_attn_func，如果导入失败，则从本地(../ops/flash_attn_triton)导入闪电注意力算子
try:
    from flash_attn import flash_attn_func
except ImportError:
    from ..ops.flash_attn_triton import _flash_attn_triton_decoding as flash_attn_func


def torch_block_wise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: torch.Tensor,
    block_size: int,
    grid_offset: int = 0,
):
    """
    使用 PyTorch 实现的块状稀疏注意力(解码模式)，主要用于调试或参考。
    
    参数说明：
        q, k, v: 形状均为 [batch_size, seq_len, num_heads, head_dim]
                 其中 seq_len 在这里假设是一致的
        block_idx: 每个批次、每个头上被激活的块的索引
        block_size: 块大小
        grid_offset: 用于指定网格在序列中偏移量的起始位置

    返回：
        形状 [batch_size, seq_len, num_heads, head_dim] 的注意力结果
    """
    b, n, h, d = q.shape
    assert k.shape == q.shape
    assert v.shape == k.shape
    # 计算激活块的总数
    num_block = math.ceil(grid_offset / block_size) + math.ceil(
        (n - grid_offset) / block_size
    )
    # 根据 block_idx 构建掩码 mask
    mask = torch.zeros(b, h, num_block, num_block, dtype=torch.bool, device=q.device)
    # block_idx 是按 num_block 行列展开的二维网格的扁平索引，因此需要 //, % 操作转换到 2D
    mask[
        torch.arange(b).view(b, 1, 1).expand(b, h, block_idx.shape[-1]),
        torch.arange(h).view(1, h, 1).expand(b, h, block_idx.shape[-1]),
        block_idx // num_block,
        block_idx % num_block,
    ] = 1
    # 只使用下三角区域：act_blocks_per_row 是用于统计块的个数
    act_blocks_per_row = torch.tril(mask).sum(-1)
    # 恢复到原序列维度，把块再 repeat 回去
    mask = mask.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
    # 切片到序列实际长度
    mask = mask[..., grid_offset : grid_offset + n, grid_offset : grid_offset + n]
    # 再次只取下三角，确保因果性
    mask = torch.tril(mask)
    # 计算注意力分数 attn_weight = q * k^T / sqrt(d)
    attn_weight = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(d)
    attn_weight.masked_fill_(~mask, float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # 最后得到输出
    o = torch.einsum("bhij,bjhd->bhid", attn_weight, v)
    o = o.transpose(1, 2)
    return o


@triton.jit
def block_wise_decode_attention_kernel(
    q_ptr,  # [batch_size, seq_len, num_heads, head_dim]
    k_ptr,
    v_ptr,
    o_ptr,
    block_idx_ptr,  # [batch_size, num_heads, num_activated_block]
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    GQA_GROUPS,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCK,
    # softmax_scale
    softmax_scale,
    # gqa
    gqa_interleave: tl.constexpr,
    # stride
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    stride_bb,
    stride_bh,
    stride_bt,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q 的 block 大小
    BLOCK_SIZE_K: tl.constexpr,  # k 的 block 大小
):
    """
    这个 Triton kernel 在解码模式中实现块稀疏注意力的核心计算。
    通过将查询 q 拆成小块(BLOCK_SIZE_Q)，并迭代获取对应激活块中的 key, value 来做注意力计算。
    最终将注意力结果写到输出缓存 o_ptr。
    """
    # 这里的 pid_b, pid_h 表示当前核线程所属的 batch 维度下标和 head 下标
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # 如果是 gqa_interleave 为 True，则使用另一种索引方式
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // GQA_GROUPS

    # 取当前行的 block_idx_ptr 指针
    block_idx_ptr = block_idx_ptr + pid_b * stride_bb + pid_h * stride_bh

    # 构造 q 的块指针（Block Pointer）
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(1, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    # 构造 k 的块指针
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    # 构造 v 的块指针
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )

    # 加载 q 的数据到寄存器
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    # 这里初始化存储注意力统计的变量
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)

    # 开始循环处理 NUM_BLOCK 个块
    for i in range(0, NUM_BLOCK):
        # 加载当前块的起始列号 c
        c = tl.load(block_idx_ptr).to(tl.int32) * BLOCK_SIZE_K
        block_idx_ptr = block_idx_ptr + stride_bt

        # 加载对应的 k
        k = tl.load(
            tl.advance(k_ptrs, (0, c)), boundary_check=(1,), padding_option="zero"
        )
        # 计算 q * k^T
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where((off_n < K_LEN - c)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * softmax_scale
        # 计算本批次的局部最大值 m_ij 和 log-sum-exp
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])  # 2 的指数，提升数值稳定性
        l_ij = tl.sum(p, axis=1)
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]

        # 加载当前块的 v，然后更新 o
        v = tl.load(
            tl.advance(v_ptrs, (c, 0)), boundary_check=(0,), padding_option="zero"
        )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # 更新统计量
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)

    # 最后的缩放
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    # 写回输出
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(1, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0,))


def triton_block_wise_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: torch.Tensor,
    block_size: int,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """
    Triton 实现的解码模式下的块稀疏注意力。
    
    参数：
        q, k, v: 形状分别是 [batch_size, 1, num_heads, head_dim], 
                  [batch_size, seq_len, num_heads, head_dim], 与 block_idx 配合使用
        block_idx: 已激活块的索引
        block_size: 块大小
        softmax_scale: 注意力分数缩放因子，可自定义
        gqa_interleave: 是否使用 gqa 交织模式

    返回：
        [batch_size, 1, num_heads, head_dim]
    """
    batch_size, q_len, num_q_heads, head_dim = q.shape
    # 断言 q_len == 1，是解码模式
    assert q_len == 1
    batch_size, k_len, num_kv_heads, head_dim = k.shape
    batch_size, num_q_heads, num_blocks = block_idx.shape
    assert q.dtype == torch.bfloat16
    # 只支持特定大小
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    assert block_size in {
        16,
        32,
        64,
        128,
    }, "only support block size in {16, 32, 64, 128}"
    # block 数量不能超过 seq_len / block_size
    assert num_blocks <= triton.cdiv(k_len, block_size)
    # 确保 GQA 维度匹配
    assert num_q_heads % num_kv_heads == 0
    gqa_groups = num_q_heads // num_kv_heads

    # softmax_scale 默认值
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)

    # 对 block_idx 排序，便于 kernel 中处理
    block_idx = block_idx.sort(-1).values

    # 准备输出
    o = torch.empty_like(q)
    num_warps = 8
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = block_size

    # 调用底层 Triton kernel
    block_wise_decode_attention_kernel[(batch_size, num_q_heads)](
        q,
        k,
        v,
        o,
        block_idx,
        batch_size,
        num_q_heads,
        num_q_heads,
        num_kv_heads,
        gqa_groups,
        k_len,
        head_dim,
        num_blocks,
        softmax_scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return o


@triton.jit
def count_kernel(
    x_ptr,
    y_ptr,
    k,
    r,
    stride_xb,
    stride_xh,
    stride_xk,
    stride_yb,
    stride_yh,
    stride_yr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    """
    count_kernel 用于统计 x_ptr 中的 block 索引落在哪些 bin(列)里，并将结果做 cumsum 累加。
    其中 x_ptr 是一维索引，r 为列数(或 bin 数)，最后结果存到 y_ptr。
    
    x_ptr: [batch_size, num_heads, activated_block_num]
    y_ptr: [batch_size, num_heads, r+1]
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    # 将 x_ptr 指向对应 batch 和 head 的起始地址
    x_ptr = x_ptr + pid_b * stride_xb + pid_h * stride_xh

    off_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + off_k * stride_xk

    y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)

    # 每次批处理 BLOCK_SIZE_K 个元素
    for i in range(0, k, BLOCK_SIZE_K):
        # 读取 x
        x = tl.load(x_ptrs, off_k < k - i, -1)
        x = x // r
        x = tl.where(off_k < k - i, x, -1)
        # 进行直方图统计
        y += tl.histogram(x, BLOCK_SIZE_R)
        # 移动指针处理下一个BLOCK_SIZE_K
        x_ptrs = x_ptrs + BLOCK_SIZE_K * stride_xk

    # 对结果做 cumsum
    y = tl.cumsum(y, axis=0)

    # 写回 y_ptr
    y_ptr = y_ptr + pid_b * stride_yb + pid_h * stride_yh + stride_yr
    off_r = tl.arange(0, BLOCK_SIZE_R)
    tl.store(y_ptr + off_r * stride_yr, y, off_r < r)


def triton_column_count_cumsum(x: torch.Tensor, num_columns: int) -> torch.Tensor:
    """
    统计并做列方向的 cumsum，返回每行、每 batch、每 head 的块计数累加结果。
    x 形状: [batch_size, num_heads, activated_block_num]
    num_columns: 网格列数
    
    返回形状: [batch_size, num_heads, num_columns + 1]
    """
    x = x.to(torch.int32)
    b, h, k = x.shape
    r = num_columns
    block_size_k = min(triton.next_power_of_2(k), 4096)
    # 为了避免 Triton 在 histogram 上的 bug，这里要 + 2
    block_size_r = triton.next_power_of_2(r + 2)
    y = torch.zeros(b, h, r + 1, device=x.device, dtype=torch.int32)

    # 启动 kernel
    count_kernel[(b, h)](
        x,
        y,
        k,
        r,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        block_size_k,
        block_size_r,
    )
    return y


@triton.jit
def block_wise_prefill_attention_kernel(
    q_ptr,  # [batch_size, seq_len, num_heads, head_dim]
    k_ptr,
    v_ptr,
    o_ptr,
    block_idx_ptr,  # [batch_size, num_heads, num_all_block]
    idx_bin_ptr,    # [batch_size, num_heads, seq_len / block_size + 1]
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    GQA_GROUPS,
    Q_LEN,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCK,
    grid_offset,
    # softmax_scale
    softmax_scale,
    # gqa
    gqa_interleave: tl.constexpr,
    # stride
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    stride_bb,
    stride_bh,
    stride_bt,
    stride_ib,
    stride_ih,
    stride_it,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    预填充(Prefill)模式下的块稀疏注意力核心 Triton kernel，用于 batch_size, seq_lens > 1 的情况。
    与 decode_attention_kernel 最大的区别在于，这里对 Q (以及可能的 K, V) 进行多块的并行处理。
    """

    # 声明：BLOCK_SIZE_Q == BLOCK_SIZE_K
    tl.static_assert(BLOCK_SIZE_Q == BLOCK_SIZE_K)

    # 通过 pid_b, pid_h, pid_q 确定当前 kernel 的处理单元
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // GQA_GROUPS
    pid_q = tl.program_id(2)

    # 读取索引边界：决定哪些块是激活的
    idx_bin_ptr = idx_bin_ptr + pid_b * stride_ib + pid_h * stride_ih
    bin_start = tl.load(idx_bin_ptr + pid_q * stride_it)
    bin_end = tl.load(idx_bin_ptr + (pid_q + 1) * stride_it)
    num_active_block = bin_end - bin_start

    # block_idx_ptr 指向实际的激活块列表
    block_idx_ptr = (
        block_idx_ptr + pid_b * stride_bb + pid_h * stride_bh + bin_start * stride_bt
    )

    # 构造 Q, K, V 的 block pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )

    # 加载 Q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    # 初始化注意力统计数据
    off_m = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q - grid_offset
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)

    # 循环处理激活的块
    for i in tl.range(0, num_active_block):
        # 计算当前块的实际列偏移 c
        c = tl.load(block_idx_ptr).to(tl.int32) % NUM_BLOCK * BLOCK_SIZE_K - grid_offset
        block_idx_ptr = block_idx_ptr + stride_bt

        # 读 K
        k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1,), padding_option="zero")

        # 计算 q * k^T
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        # 确保只在合法的序列范围进行注意力计算
        qk += tl.where((c + off_n)[None, :] >= 0, 0, float("-inf"))
        qk += tl.where(off_m[:, None] >= (c + off_n)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * softmax_scale

        # 更新归一化过程
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]

        # 读 V 并更新输出
        v = tl.load(tl.advance(v_ptrs, (c, 0)), boundary_check=(0,), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # 更新统计
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)

    # 计算最终输出
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]

    # 存储结果
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0,))


def triton_block_wise_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: Union[torch.Tensor, List[List[torch.Tensor]]],
    block_size: int,
    grid_offset: int = 0,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """
    预填充(Prefill)模式的块稀疏注意力计算。

    参数：
        q, k, v: [batch_size, seq_len, num_heads, head_dim]
        block_idx: [batch_size, num_heads, activated_block_num]，表示扁平网格中激活的块索引
        block_size: 块大小
        grid_offset: 用于因果遮罩时的偏移
        softmax_scale: softmax 缩放系数
        gqa_interleave: GQA 交叉模式
    
    返回：
        [batch_size, seq_len, num_heads, head_dim] 的注意力输出
    """
    batch_size, q_len, num_q_heads, head_dim = q.shape
    batch_size, k_len, num_kv_heads, head_dim = k.shape
    assert q.dtype == torch.bfloat16
    assert q_len == k_len
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    assert block_size in {
        32,
        64,
        128,
    }, "only support block size in {16, 32, 64, 128}"
    total_q_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
        q_len - grid_offset, block_size
    )
    total_k_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
        k_len - grid_offset, block_size
    )

    # 如果 block_idx 是 Python 列表，需要先转换为统一的 Torch 张量
    if not isinstance(block_idx, torch.Tensor):
        assert (
            isinstance(block_idx, list)
            and isinstance(block_idx[0], list)
            and isinstance(block_idx[0][0], torch.Tensor)
        )
        assert len(block_idx) == batch_size and len(block_idx[0]) == num_q_heads
        # 先拼成 (batch_size * num_q_heads) x (若干行)
        block_idx = [item.view(-1, 1) for sublist in block_idx for item in sublist]
        block_idx = torch.nn.utils.rnn.pad_sequence(
            block_idx,
            batch_first=True,
            padding_value=total_k_blocks * (total_k_blocks + 1),
        )
        block_idx = block_idx.view(batch_size, num_q_heads, -1)

    batch_size, num_q_heads, num_block = block_idx.shape
    assert q_len == k_len
    assert num_block <= total_q_blocks * (total_q_blocks + 1) // 2
    # GQA 分组
    assert num_q_heads % num_kv_heads == 0
    gqa_groups = num_q_heads // num_kv_heads

    # softmax_scale
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)

    # 对 block_idx 做排序
    block_idx = block_idx.sort(-1).values
    # 对激活块做分 bin 统计
    idx_bins = triton_column_count_cumsum(block_idx, total_k_blocks)

    # 准备输出张量
    o = torch.empty_like(q)

    # 根据 block_size 设置阶段数
    num_warps = 8
    num_stages = 3 if block_size >= 128 else 5

    # 启动内核 kernel
    block_wise_prefill_attention_kernel[
        (batch_size, num_q_heads, total_q_blocks)
    ](
        q,
        k,
        v,
        o,
        block_idx,
        idx_bins,
        batch_size,
        num_q_heads,
        num_kv_heads,
        gqa_groups,
        q_len,
        k_len,
        head_dim,
        total_q_blocks,
        grid_offset,
        softmax_scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        idx_bins.stride(0),
        idx_bins.stride(1),
        idx_bins.stride(2),
        BLOCK_SIZE_Q=block_size,
        BLOCK_SIZE_K=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def triton_block_wise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: torch.Tensor,
    block_size: int,
    grid_offset: int = 0,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """
    对外暴露的统一接口：如果 q 的长度 > 1，就进入 prefill 的块状注意力逻辑；
    否则进入 decode 的块状注意力逻辑。
    """
    if q.shape[1] > 1:
        return triton_block_wise_prefill_attention(
            q,
            k,
            v,
            block_idx,
            block_size,
            grid_offset,
            softmax_scale,
            gqa_interleave,
        )
    else:
        return triton_block_wise_decode_attention(
            q, k, v, block_idx, block_size, softmax_scale, gqa_interleave
        )


@triton.jit
def bnhd_pool_kernel(
    x_ptr,
    y_ptr,
    # pool type: avg=0, max=1, min=2, max abs=3, sum=4
    pool_type: tl.constexpr,
    # shape
    batch_size,
    seq_len,
    num_heads,
    head_dim: tl.constexpr,
    # stride
    stride_xb,
    stride_xn,
    stride_xh,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yh,
    stride_yd,
    # META
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    按照 (batch, n, h, d) 的排列进行分块池化运算。
    因为 Triton 对高维数据的支持需要自行计算 index，所以这里需要手动处理 offsets、mask 等逻辑。
    """
    pid_b = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # seq
    pid_h = tl.program_id(2)  # head

    # 计算 x_ptr 的偏移
    x_ptr = (
        x_ptr
        + pid_b * stride_xb
        + pid_n * BLOCK_SIZE_N * stride_xn
        + pid_h * BLOCK_SIZE_H * stride_xh
    )

    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    cur_block_size_n = min(seq_len - pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)

    # 构造 mask，保证读取到的是有效数据
    x_mask = (
        (off_n < seq_len - pid_n * BLOCK_SIZE_N)[:, None, None]
        & (off_h < num_heads - pid_h * BLOCK_SIZE_H)[None, :, None]
        & (off_d < head_dim)[None, None, :]
    )
    # 加载 x
    x = tl.load(
        x_ptr
        + off_n[:, None, None] * stride_xn
        + off_h[None, :, None] * stride_xh
        + off_d[None, None, :] * stride_xd,
        mask=x_mask,
        other=0,
    )

    # 根据 pool_type 执行不同的池化操作
    if pool_type == 0:
        # avg
        y = tl.sum(x, axis=0) / cur_block_size_n
    elif pool_type == 1:
        # max
        y = tl.max(x, axis=0)
    elif pool_type == 2:
        # min
        y = tl.min(x, axis=0)
    elif pool_type == 3:
        # maxabs
        y = tl.max(tl.abs(x), axis=0)
    elif pool_type == 4:
        # sum
        y = tl.sum(x, axis=0)
    else:
        # 默认 avg
        y = tl.sum(x, axis=0) / cur_block_size_n

    # 把结果写回 y
    y_ptr = (
        y_ptr + pid_b * stride_yb + pid_n * stride_yn + pid_h * BLOCK_SIZE_H * stride_yh
    )
    y_mask = (off_h < num_heads - pid_h * BLOCK_SIZE_H)[:, None] & (off_d < head_dim)[
        None, :
    ]
    tl.store(
        y_ptr + off_h[:, None] * stride_yh + off_d[None, :] * stride_yd, y, mask=y_mask
    )


def triton_bnhd_pool(x: torch.Tensor, kernel_size: int, pool_type: str = "avg"):
    """
    针对 x（形状: [batch_size, seq_len, num_heads, head_dim]）做分块池化，
    每个 block 大小为 kernel_size，只在 seq_len 维度上分块。
    pool_type 可以为 avg, max, min, maxabs, sum 等。
    """
    x = x.to("cuda")
    b, n, h, d = x.shape
    assert d in {16, 32, 64, 128}
    assert kernel_size in {16, 32, 64, 128, 256, 512}
    m = triton.cdiv(n, kernel_size)
    y = torch.zeros(b, m, h, d, device=x.device, dtype=x.dtype)

    # 如果是“last”策略，单独处理一下
    if pool_type == "last":
        if n % kernel_size == 0:
            return x[:, kernel_size - 1 :: kernel_size, ...]
        else:
            return torch.cat(
                (x[:, kernel_size - 1 :: kernel_size, ...], x[:, -1:, ...]), dim=1
            )

    # 设置合适的 BLOCK_SIZE_H，以保证单次块大小不过大
    block_size_h = triton.next_power_of_2(h)
    while kernel_size * block_size_h * d > 128 * 128 * 128:
        block_size_h = block_size_h // 2

    # 设置合适的 BLOCK_SIZE_D
    block_size_d = triton.next_power_of_2(d)
    pool_str_to_type = {"avg": 0, "max": 1, "min": 2, "maxabs": 3, "sum": 4}
    pool_type_value = pool_str_to_type[pool_type]

    # grid 定义
    grid = lambda META: (
        b,
        triton.cdiv(n, META["BLOCK_SIZE_N"]),
        triton.cdiv(h, META["BLOCK_SIZE_H"]),
    )

    # 启动 kernel
    bnhd_pool_kernel[grid](
        x,
        y,
        pool_type_value,
        b,
        n,
        h,
        d,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        BLOCK_SIZE_N=kernel_size,
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_D=block_size_d,
    )
    return y


@triton.jit
def bhn_sumpool_kernel(
    x_ptr,
    y_ptr,
    # shape
    batch_size,
    num_heads,
    seq_len,
    # stride
    stride_xb,
    stride_xh,
    stride_xn,
    stride_yb,
    stride_yh,
    stride_yn,
    # META parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    针对 [batch_size, head, seq_len] 形状的数据，在 seq_len 维度进行 sum pooling。
    每个块大小为 BLOCK_SIZE_N。
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    # 计算 x_ptr 偏移
    x_ptr = (
        x_ptr
        + pid_b * stride_xb
        + pid_h * BLOCK_SIZE_H * stride_xh
        + pid_n * BLOCK_SIZE_N * stride_xn
    )
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    # 构造 mask
    x_mask = (off_n < seq_len - pid_n * BLOCK_SIZE_N)[None, :] & (
        off_h < num_heads - pid_h * BLOCK_SIZE_H
    )[:, None]

    # 读取并累加
    x = tl.load(
        x_ptr + off_h[:, None] * stride_xh + off_n[None, :] * stride_xn,
        mask=x_mask,
        other=0,
    )
    y = tl.sum(x, axis=1)

    # 写回 y
    y_ptr = (
        y_ptr + pid_b * stride_yb + pid_h * BLOCK_SIZE_H * stride_yh + pid_n * stride_yn
    )
    y_mask = off_h < num_heads - pid_h * BLOCK_SIZE_H
    tl.store(y_ptr + off_h * stride_yh, y, mask=y_mask)


def triton_bhn_sumpool(x: torch.Tensor, kernel_size: int):
    """
    在 [batch, head, seq_len] 维度上分块求和(sum pool)。
    """
    b, h, n = x.shape
    assert kernel_size in {16, 32, 64, 128, 256, 512}
    m = triton.cdiv(n, kernel_size)
    y = torch.empty(b, h, m, device=x.device, dtype=x.dtype)

    block_size_h = triton.next_power_of_2(h)
    grid = lambda META: (
        b,
        triton.cdiv(h, META["BLOCK_SIZE_H"]),
        triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )

    # 启动内核
    bhn_sumpool_kernel[grid](
        x,
        y,
        b,
        h,
        n,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        BLOCK_SIZE_N=kernel_size,
        BLOCK_SIZE_H=block_size_h,
    )
    return y


def torch_bhn_sumpool(x: torch.Tensor, kernel_size: int):
    """
    使用 PyTorch 对 [batch, head, seq_len] 做 sum pool，属于对照或参考实现。
    """
    b, h, n = x.shape
    x = torch.nn.functional.pad(
        x,
        (
            0,
            math.ceil(n / kernel_size) * kernel_size - n,
        ),
        value=0,
    )
    x = x.view(b, h, -1, kernel_size).sum(-1)
    return x


def score_cover_topk(x: torch.Tensor, score: float):
    """
    给定注意力分布 x(已排序)，求使得累加和达到 score 所需的 topk。
    返回每个 batch/head 的 topk 块数。
    """
    cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1)
    topk = torch.sum(cumsum_x <= score, dim=-1) + 1
    return topk


def score_cover_idx(x: torch.Tensor, score: float, padding_value=0):
    """
    类似于 score_cover_topk，但这里返回对应的索引，把超过 score 累加的部分索引都置为 padding_value。
    """
    x, idx = torch.sort(x, dim=-1, descending=True)
    cumsum_x = torch.cumsum(x, dim=-1)
    idx[cumsum_x > score] = padding_value
    return idx


def sum_all_diagonal_matrix(mat: torch.tensor):
    """
    将 4 维张量 mat [b, h, n, m] 的每个 [n,m] 矩阵沿对角线进行累加，
    返回形状 [b, h, m] 的累加和。
    """
    b, h, n, m = mat.shape
    # 先对 mat 做 padding，再通过 as_strided "错位"来得到对角线的求和
    mat_padded = torch.nn.functional.pad(mat, (n - 1, 0), value=0)
    mat_strided = mat_padded.as_strided(
        (b, h, m, n), (h * n * (n + m - 1), n * (n + m - 1), 1, n + m)
    )
    sum_diags = torch.sum(mat_strided, -1)
    return sum_diags


def transform_veritcal_slash_idx(v_idx, s_idx, num_blocks):
    """
    将垂直索引 v_idx 和斜线索引 s_idx 合并成一个集合，然后去重。
    对应在 2D 网格 (num_blocks x num_blocks) 上选取。
    """
    batch_size, num_heads, _ = v_idx.shape
    range_blocks = torch.arange(num_blocks, device=s_idx.device)[None, None, :, None]

    # 垂直 block 计算： v_idx[b,h] * num_blocks + vertical
    v_idx = (
        torch.arange(0, num_blocks, device=v_idx.device)[None, None, :, None]
        * num_blocks
        + v_idx[:, :, None, :]
    ).view(batch_size, num_heads, -1)
    v_idx[v_idx // num_blocks < v_idx % num_blocks] = 0

    # 斜线 block 计算
    s_idx = (
        range_blocks * num_blocks + range_blocks + s_idx[:, :, None, :] * num_blocks
    ).view(batch_size, num_heads, -1)
    s_idx[s_idx >= num_blocks * num_blocks] = 0

    # 合并
    vs_idx = torch.cat((s_idx, v_idx), dim=-1)
    block_idx = [
        [torch.unique(vs_idx[b, h]) for h in range(num_heads)]
        for b in range(batch_size)
    ]
    return block_idx


causal_mask = None


def get_block_vertical_slash_from_qk(
    qk: torch.Tensor,
    block_size: int,
):
    """
    从 qk 分布中，分别计算竖直方向(Vertical)和对角线方向(Slash)的注意力聚合结果。
    用于找出在哪些 block 上有较多注意力。
    """
    batch_size, num_heads, last_q_len, seq_len = qk.shape
    # slash：对 qk 沿着对角线方向求和
    slash = sum_all_diagonal_matrix(qk)
    # 再对 slash 在 seq_len 上按 block_size 做 sum pool
    slash = torch_bhn_sumpool(slash, block_size)
    slash = slash / last_q_len

    # vertical：把 qk 沿 j 方向 (最后一维) 求和
    vertical = qk.sum(-2)
    vertical = torch_bhn_sumpool(vertical, block_size)
    vertical = vertical / last_q_len
    return vertical, slash


def square_root_js_divergence(p: torch.Tensor, q: torch.Tensor):
    """
    计算 p, q 之间的 JS 散度，然后取平方根。
    JS 散度可以度量两个分布的相似性。
    """
    m = (p + q) / 2
    return torch.sqrt(
        0.5 * (p * torch.log(p / m)).sum(-1) + 0.5 * (q * torch.log(q / m)).sum(-1)
    )


def get_active_blocks(
    q,
    k,
    v,
    block_size,
    gamma,
    min_budget,
    max_budget,
    tau=0,
    gqa_interleave=False,
):
    """
    根据最后一个 block 的注意力分布，动态决定哪些 block 应该被激活(在预填充时)。
    大致思路：
    1. 取最后 block 的 q 与全量 k 做 qk。
    2. 沿着斜线(slash)和垂直(vertical)方向聚合。
    3. 根据 gamma 设置的阈值，看哪些块在注意力上贡献大于阈值。
    4. 如果发现某些块分布异常(通过 js 散度判断)，就保留最小的活跃块数量。
    5. 最终返回一个 block_idx 列表，用于后续的稀疏注意力。
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    gqa_groups = num_heads // k.shape[2]
    num_blocks = math.ceil(seq_len / block_size)
    max_budget = min(max_budget, num_blocks)
    # 取最后 block 的 Q
    last_q = q[:, -block_size:, :, :] / math.sqrt(head_dim)

    # 根据是否 gqa_interleave 来计算 qk
    if not gqa_interleave:
        qk = torch.einsum(
            "bihgd, bjhgd -> bhgij",
            last_q.view(last_q.shape[0], last_q.shape[1], -1, gqa_groups, head_dim),
            k.view(k.shape[0], k.shape[1], -1, 1, head_dim),
        )
    else:
        qk = torch.einsum(
            "bihgd, bjhgd -> bhgij",
            last_q.view(last_q.shape[0], last_q.shape[1], gqa_groups, -1, head_dim),
            k.view(k.shape[0], k.shape[1], 1, -1, head_dim),
        )

    global causal_mask
    # 构造因果遮罩，只对最后一个 block 内部做下三角
    if causal_mask is None:
        causal_mask = torch.arange(0, block_size, device=last_q.device)
        causal_mask = causal_mask[:, None] >= causal_mask[None, :]
        causal_mask = causal_mask[None, None, None, ...]

    # 对最后 block 部分做因果遮罩
    qk[..., -block_size:].masked_fill_(
        ~causal_mask[..., :block_size, :block_size].to(qk.device), float("-inf")
    )
    # 做 softmax
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    # 把 (b, h, g, i, j) reshape 成 (b, h*g, i, j)
    qk = rearrange(qk, "b h g i j -> b (h g) i j")

    # 计算 slash & vertical
    slash = sum_all_diagonal_matrix(qk) / qk.shape[-2]
    vertical = qk.mean(-2)

    # 根据 gamma 找出需要覆盖的块数
    num_vertical_blocks = score_cover_topk(vertical, gamma) // 128 + 1
    num_slash_blocks = score_cover_topk(slash, gamma) // 128 + 1

    # 对应最小和最大限制
    num_vertical_blocks[num_vertical_blocks < min_budget] = min_budget
    num_vertical_blocks[num_vertical_blocks > max_budget] = max_budget
    num_slash_blocks[num_slash_blocks < min_budget] = min_budget
    num_slash_blocks[num_slash_blocks > max_budget] = max_budget

    # 做 block_size 的池化
    vertical = torch_bhn_sumpool(vertical, block_size)
    slash = torch_bhn_sumpool(slash, block_size)

    # 如果使用 gqa_interleave，就对 K 做分块池化后再重复
    if not gqa_interleave:
        avg_k = triton_bnhd_pool(k, block_size).repeat_interleave(gqa_groups, 2)
    else:
        avg_k = triton_bnhd_pool(k, block_size).repeat(1, 1, gqa_groups, 1)

    # avg_qk 用于判断块稀疏分布情况
    avg_qk = torch.einsum(
        "bihd, bjhd -> bhij", last_q.mean(1, keepdim=True), avg_k
    ).squeeze(2)
    avg_qk = torch.softmax(avg_qk, dim=-1, dtype=torch.float32)
    kl_div = square_root_js_divergence(avg_qk, vertical)
    block_sparse_mask = kl_div < tau

    # 如果分布过于 sparse，就把块预算设成 min_budget
    num_vertical_blocks[block_sparse_mask] = min_budget
    num_slash_blocks[block_sparse_mask] = min_budget

    # 保留最左的块
    vertical[..., :1] = torch.inf
    # 保留最右的块
    slash[..., -1:] = torch.inf

    # 分别获取 topk
    num_slash_blocks = num_slash_blocks.view(batch_size * num_heads)
    slash = slash.view(batch_size * num_heads, -1)
    slash_topk = (num_blocks - 1) - slash.topk(
        min(num_slash_blocks.max().item(), num_blocks), -1
    ).indices
    slash_topk[
        torch.arange(slash_topk.shape[-1], device=num_slash_blocks.device)[None, :]
        >= num_slash_blocks[:, None]
    ] = 0
    slash_topk = slash_topk.view(batch_size, num_heads, -1)

    num_vertical_blocks = num_vertical_blocks.view(batch_size * num_heads)
    vertical = vertical.view(batch_size * num_heads, -1)
    vertical_topk = vertical.topk(
        min(num_vertical_blocks.max().item(), num_blocks), -1
    ).indices
    vertical_topk[
        torch.arange(vertical_topk.shape[-1], device=num_vertical_blocks.device)[None, :]
        >= num_vertical_blocks[:, None]
    ] = 0
    vertical_topk = vertical_topk.view(batch_size, num_heads, -1)

    # 最后组合
    block_idx = transform_veritcal_slash_idx(vertical_topk, slash_topk, num_blocks)

    # 对剩下非常 sparse 的情况，再做一次小优化
    block_causal_mask = None
    for b, h in block_sparse_mask.nonzero():
        # 建立因果遮罩
        if block_causal_mask is None:
            block_causal_mask = torch.tril(
                torch.ones(num_blocks, num_blocks, device=q.device, dtype=torch.bool)
            )
        # 做 k 的池化
        pad_q = math.ceil(seq_len / block_size) * block_size - seq_len
        avg_q = (
            torch.nn.functional.pad(q[b, :, h, :], (0, 0, 0, pad_q), value=0)
            .view(num_blocks, block_size, head_dim)
            .mean(1)
        )
        # 最后一块需要处理 padding
        avg_q[-1, :] = avg_q[-1, :] * block_size / (block_size - pad_q)

        # 因果注意力
        attn = torch.einsum(
            "id, jd -> ij", avg_q / math.sqrt(head_dim), avg_k[b, :, h, :]
        ).masked_fill_(~block_causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).view(-1)

        block_topk = score_cover_idx(attn, gamma * num_blocks)
        block_idx[b][h] = torch.unique(torch.cat((block_idx[b][h], block_topk), dim=-1))

    return block_idx


def flex_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: float,
    tau: float = 0,
    min_budget: int = None,
    max_budget: int = None,
    gqa_interleave: bool = False,
    softmax_scale: Optional[float] = None,
    block_size: int = 128,
):
    """
    动态(柔性)预填充块注意力(FlexPrefill)的核心函数：
    1. 当序列不长时，直接用 FlashAttention。
    2. 当序列较长时，先通过 get_active_blocks 找出最重要的块，然后只计算那些块的注意力。
    """
    import time
    import math

    batch_size, seq_len, num_heads, head_dim = q.shape
    assert q.shape[1] == k.shape[1]
    assert head_dim in {16, 32, 64, 128}
    assert block_size in {16, 32, 64, 128}
    num_blocks = math.ceil(seq_len / block_size)

    min_budget = 1 if min_budget is None else min_budget
    max_budget = 2147483647 if max_budget is None else max_budget

    # 如果 seq_len 不大，直接用 flash_attn
    # if seq_len <= max(2 * block_size, math.ceil(min_budget / block_size) * block_size):
    #     return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=True)

    # 查找块模式计时
    # decide_pattern_times = []
    # for _ in range(10):
    #     start = time.time()
    block_idx = get_active_blocks(
        q,
        k,
        v,
        block_size,
        gamma,
        math.ceil(min_budget / block_size),
        math.ceil(max_budget / block_size),
        tau,
        gqa_interleave,
    )
        # torch.cuda.synchronize()
        # end = time.time()
        # decide_pattern_times.append(end - start)
    
    # print(f"[decide_pattern_times] {decide_pattern_times}")
    # print(f"block_idx: {block_idx}")

    # 稀疏注意力计时
    # block_sparse_times = []
    # for _ in range(10):
    #     start = time.time()
    attn_out = triton_block_wise_attention(
        q,
        k,
        v,
        block_idx,
        block_size,
        softmax_scale=softmax_scale,
        gqa_interleave=gqa_interleave,
    )
        # torch.cuda.synchronize()
        # end = time.time()
        # block_sparse_times.append(end - start)
    
    # print(f"[block_sparse_times] {block_sparse_times}")

    # 全量KV注意力计时
    # full_kv_times = []
    # for _ in range(10):
    #     start = time.time()
    #     flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=True)
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     full_kv_times.append(end - start)

    # print(f"[full_kv_times] {full_kv_times}")

    return attn_out


def flexprefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config,
):
    """
    一个示例性的前向函数，展示如何调用 flex_prefill_attention。
    根据 config 中的 attn_forward_config 设定来执行动态预填充注意力。
    """
    gamma = config["attn_forward_config"].get("gamma", 0.9)
    tau = config["attn_forward_config"].get("tau", 0.1)
    min_budget = config["attn_forward_config"].get("min_budget", None)
    max_budget = config["attn_forward_config"].get("max_budget", None)
    block_size = config["attn_forward_config"].get("block_size", 128)

    # 先做一次 transpose，把输入从 [B, H, N, D] 变成 [B, N, H, D]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    # 调用主函数
    out = flex_prefill_attention(
        q, k, v, gamma, tau, min_budget, max_budget, block_size
    )

    # 输出前再变回来
    return out.transpose(1, 2)


if __name__ == "__main__":
    # 测试入口，示例性演示如何调用
    torch.manual_seed(0)
    B, N, H, D = 1, 32000, 32, 128  # batch, seq_len, heads, head_dim
    gamma = 0.9
    tau = 0.1

    # 构造随机的 q, k, v
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

    # 调用柔性预填充注意力
    flex_prefill_output = flex_prefill_attention(q, k, v, gamma, tau)
    # 这里不做任何额外输出，仅作为功能性测试
