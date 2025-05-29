import torch

def get_mean_q(q, k):
    attn_weigth = torch.matmul(q, k)
    return attn_weigth

def get_compress_attn_weight(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    # Need a faster kernel implementation
    q_length, q_dim = q.shape
    k_length, k_dim = k.shape
    assert k_dim == q_dim
    compress_q = torch.mean(q.contiguous().view(q_length // block_size, block_size, q_dim), dim=-2)
    compress_k = torch.mean(k.contiguous().view(k_length // block_size, block_size, k_dim), dim=-2)

    # q_length/block, k_length
    compress_q_attn = torch.matmul(compress_q, k.transpose(-1, -2))

    # q_length, k_length/block
    compress_k_attn = torch.matmul(q, compress_k.transpose(-1, -2))

    # q_length/block, k_length/block
    merge_k_mean_attn, _ = torch.max(
        compress_k_attn.view(q_length // block_size, block_size, k_length // block_size), dim=-2
    )

    # q_length/block, k_length
    merge_k_mean_attn = (
        merge_k_mean_attn.unsqueeze(-1)
        .expand(q_length // block_size, k_length // block_size, block_size)
        .contiguous()
    )
    merge_k_mean_attn = merge_k_mean_attn.view(q_length // block_size, k_length)

    # Take the maximum value
    attn = torch.max(compress_q_attn, merge_k_mean_attn)
    return attn

# Without horizontal optimization
def get_compress_attn_weight_with_query(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    # Need a faster kernel implementation
    q_length, q_dim = q.shape
    k_length, k_dim = k.shape
    assert k_dim == q_dim
    compress_q = torch.mean(q.contiguous().view(q_length // block_size, block_size, q_dim), dim=-2)
    compress_q_attn = torch.matmul(compress_q, k.transpose(-1, -2))
    return compress_q_attn