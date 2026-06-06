import torch
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import math
import os

def llama_flash_attn2_get_qkv_tensor(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    torch.cuda.empty_cache()
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if self.layer_idx <= 31:
        dir_path = f'tensor/{self.layer_idx}/{query_states.size(1)}'
        os.makedirs(dir_path, exist_ok=True)
        if self.config.dataset:
            torch.save(query_states, f'{dir_path}/query_states_{self.config.dataset}.pt')
            torch.save(key_states, f'{dir_path}/key_states_{self.config.dataset}.pt')
            torch.save(value_states, f'{dir_path}/value_states_{self.config.dataset}.pt')
        else:
            torch.save(query_states, f'{dir_path}/query_states.pt')
            torch.save(key_states, f'{dir_path}/key_states.pt')
            torch.save(value_states, f'{dir_path}/value_states.pt')
    else:
        assert False, f"All tensors have been saved to tensor/{self.layer_idx}"

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None
