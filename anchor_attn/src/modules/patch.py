import copy
import json
import os
import types
import warnings
import functools

import torch
from transformers import PreTrainedModel


from my_utils import Logger

ATTENTION_CONFIG_EXAMPLE = {
    "default": {},
    "flash": {},
    "streaming_llm": {
        "global_window": 1024,
        "local_window": 1024 * 8,
    },
    "minfer": {
        "model_type": "llama3.1",
    },
    "vertical_slash": {
        "block_size": 128,
        "vertical_size": 1024,
        "slash_size": 1024 * 8,
    },
    "flex_prefill": {
        "block_size": 128,
        "flex_prefill_gamma": 0.95,
        "flex_prefill_tau": 0.1,
        "flex_prefill_min_budget": 1024,
        "flex_prefill_max_budget": None,
    },
    "anchor_attn": {
        "block_size_M": 128,
        "theta": 12,
        "step": 16,
    },
    "anchor_attn_lite": {
        "block_size_M": 128,
        "theta": 12,
        "step": 16,
    },
    "get_qkv_tensor": {
        "dataset": None,
    },
    "max_kv_distribute": {
        "dataset": None,
    }
}


def get_config_example(pattern: str = None):
    if pattern is None:
        return copy.deepcopy(ATTENTION_CONFIG_EXAMPLE)
    else:
        return copy.deepcopy(ATTENTION_CONFIG_EXAMPLE[pattern])


def patch_model_config(model: PreTrainedModel, pattern: str, cfg: dict):
    if cfg is None:
        cfg = get_config_example(pattern)
    else:
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)  # Parse JSON string into dict
            except json.JSONDecodeError as e:
                raise ValueError(f"Configuration string is not valid JSON: {e}")
        t_cfg = get_config_example(pattern)
        for k, v in t_cfg.items():
            if k not in cfg:
                cfg[k] = v

    if pattern == "minfer":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mt = cfg["model_type"].lower()
        if mt == "llama3.1":
            cfg_path = os.path.join(
                current_dir,
                "../ops/minfer/config/Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json",
            )
        elif mt == "glm4":
            cfg_path = os.path.join(
                current_dir,
                "../ops/minfer/config/GLM_4_9B_1M_instruct_kv_out_v32_fit_o_best_pattern.json",
            )
        elif mt == "qwen2":
            cfg_path = os.path.join(
                current_dir,
                "../ops/minfer/config/Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json",
            )
        elif mt == "yi":
            cfg_path = os.path.join(
                current_dir,
                "../ops/minfer/config/Yi_9B_200k_kv_out_v32_fit_o_best_pattern.json",
            )
        else:
            raise ValueError(f"Unknown minfer model type {cfg['model_type']}")
        print(f"cfg_path:{cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        setattr(model.config, "minfer_config", cfg)

    elif pattern in ("anchor_attn", "anchor_attn_lite"):
        for k, v in cfg.items():
            setattr(model.config, k, v)
        Logger.log(f"model.config:{model.config}")

    else:
        for k, v in cfg.items():
            setattr(model.config, k, v)


def patch_llama_attention(model: PreTrainedModel, pattern: str):
    if pattern == "default":
        return

    from transformers.models.llama.modeling_llama import (
        LlamaFlashAttention2,
        LlamaForCausalLM,
        LlamaMLP,
    )
    assert isinstance(model, LlamaForCausalLM)

    if pattern == "flash":
        from anchor_attn.src.modules.llama.flash_attention import llama_flash_attention_forward
        new_forward = llama_flash_attention_forward
    elif pattern == "minfer":
        from anchor_attn.src.modules.llama.minfer_attention import llama_minfer_attention_forward
        new_forward = llama_minfer_attention_forward
    elif pattern == "vertical_slash":
        from anchor_attn.src.modules.llama.vertical_slash_attention import llama_vertical_slash_attention_forward
        new_forward = llama_vertical_slash_attention_forward
    elif pattern == "streaming_llm":
        from anchor_attn.src.modules.llama.streaming_llm_attention import llama_streaming_llm_attention_forward
        new_forward = llama_streaming_llm_attention_forward
    elif pattern == "flex_prefill":
        from anchor_attn.src.modules.llama.flex_prefill_attention import llama_flex_prefill_attention_forward
        new_forward = llama_flex_prefill_attention_forward
    elif pattern in ("anchor_attn", "anchor_attn_lite"):
        from anchor_attn.src.modules.llama.anchor_attention import llama_flash_attn2_forward_anchorattn
        new_forward = llama_flash_attn2_forward_anchorattn
    else:
        raise ValueError(f"Unknown attention pattern {pattern}")

    print(f"use {pattern} pattern")
    Logger.log(f"use {pattern} pattern")

    for _, m in model.named_modules():
        if isinstance(m, LlamaFlashAttention2):
            m.forward = types.MethodType(new_forward, m)

    from anchor_attn.src.modules.llama.causal_model_forward import llama_causal_model_forward
    if isinstance(model.forward, functools.partial):
        model.forward.__wrapped__ = types.MethodType(llama_causal_model_forward, model)
    else:
        model.forward = types.MethodType(llama_causal_model_forward, model)

    from anchor_attn.src.modules.llama.llama_mlp_forward import llama_mlp_forward
    for _, m in model.named_modules():
        if isinstance(m, LlamaMLP):
            m.forward = types.MethodType(llama_mlp_forward, m)


def patch_glm_attention(model: PreTrainedModel, pattern: str):
    if pattern == "default":
        return

    assert type(model).__name__ == "ChatGLMForConditionalGeneration"

    if pattern == "flash":
        from anchor_attn.src.modules.glm.flash_attention import glm_flash_attention_forward
        new_forward = glm_flash_attention_forward
    elif pattern == "streaming_llm":
        from anchor_attn.src.modules.glm.streaming_llm_attention import glm_streaming_llm_attention_forward
        new_forward = glm_streaming_llm_attention_forward
    elif pattern == "minfer":
        from anchor_attn.src.modules.glm.minfer_attention import glm_minfer_attention_forward
        new_forward = glm_minfer_attention_forward
    elif pattern == "vertical_slash":
        from anchor_attn.src.modules.glm.vertical_slash_attention import glm_vertical_slash_attention_forward
        new_forward = glm_vertical_slash_attention_forward
    elif pattern == "flex_prefill":
        from anchor_attn.src.modules.glm.flex_prefill_attention import glm_flex_prefill_attention_forward
        new_forward = glm_flex_prefill_attention_forward
    elif pattern in ("anchor_attn", "anchor_attn_lite"):
        from anchor_attn.src.modules.glm.anchor_attention import glm_anchor_attention_forward
        new_forward = glm_anchor_attention_forward
    # elif pattern == "max_kv_distribute":
    #     from anchor_attn.src.modules.glm.max_position import glm_max_attention_forward
    #     new_forward = glm_max_attention_forward
    else:
        raise ValueError(f"Unknown attention pattern {pattern}")

    from anchor_attn.src.modules.glm.glm_mlp_forward import glm_mlp_forward
    from anchor_attn.src.modules.glm.glm_self_attention_foward import glm_self_attention_forward

    for _, m in model.named_modules():
        if type(m).__name__ == "FlashAttention2":
            m.forward = types.MethodType(new_forward, m)
        if type(m).__name__ == "SelfAttention":
            m.forward = types.MethodType(glm_self_attention_forward, m)
        if type(m).__name__ == "MLP":
            m.forward = types.MethodType(glm_mlp_forward, m)


def patch_mistral_attention(model: PreTrainedModel, pattern: str):
    if pattern == "default":
        return

    from transformers.models.mistral.modeling_mistral import MistralFlashAttention2

    if pattern == "flash":
        from anchor_attn.src.modules.mistral.flash_attention import mistral_flash_attention_forward
        new_forward = mistral_flash_attention_forward
    elif pattern in ("anchor_attn", "anchor_attn_lite"):
        from anchor_attn.src.modules.mistral.anchor_attention import mistral_anchor_attention_forward
        new_forward = mistral_anchor_attention_forward
    # elif pattern == "max_kv_distribute":
    #     from anchor_attn.src.modules.mistral.max_position import mistral_max_attention_forward
    #     new_forward = mistral_max_attention_forward
    else:
        raise ValueError(f"Unknown attention pattern {pattern}")

    for _, m in model.named_modules():
        if isinstance(m, MistralFlashAttention2):
            m.forward = types.MethodType(new_forward, m)


def patch_qwen2_attention(model: PreTrainedModel, pattern: str):
    if pattern == "default":
        return

    from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2, Qwen2ForCausalLM, Qwen2MLP
    assert isinstance(model, Qwen2ForCausalLM)

    if pattern == "flash":
        from anchor_attn.src.modules.qwen2.flash_attention import qwen2_flash_attention_forward
        new_forward = qwen2_flash_attention_forward
    elif pattern == "streaming_llm":
        from anchor_attn.src.modules.qwen2.streaming_llm_attention import qwen2_streaming_llm_attention_forward
        new_forward = qwen2_streaming_llm_attention_forward
    elif pattern == "minfer":
        from anchor_attn.src.modules.qwen2.minfer_attention import qwen2_minfer_attention_forward
        new_forward = qwen2_minfer_attention_forward
    elif pattern == "vertical_slash":
        from anchor_attn.src.modules.qwen2.vertical_slash_attention import qwen2_vertical_slash_attention_forward
        new_forward = qwen2_vertical_slash_attention_forward
    elif pattern == "flex_prefill":
        from anchor_attn.src.modules.qwen2.flex_prefill_attention import qwen2_flex_prefill_attention_forward
        new_forward = qwen2_flex_prefill_attention_forward
    elif pattern in ("anchor_attn", "anchor_attn_lite"):
        from anchor_attn.src.modules.qwen2.anchor_attention import qwen2_anchor_attention_forward
        new_forward = qwen2_anchor_attention_forward
    else:
        raise ValueError(f"Unknown attention pattern {pattern}")

    from anchor_attn.src.modules.qwen2.qwen_mlp_forward import qwen2_mlp_forward
    for _, m in model.named_modules():
        if isinstance(m, Qwen2FlashAttention2):
            m.forward = types.MethodType(new_forward, m)
        if isinstance(m, Qwen2MLP):
            m.forward = types.MethodType(qwen2_mlp_forward, m)

    from anchor_attn.src.modules.qwen2.causal_model_forward import qwen2_causal_model_forward
    if isinstance(model.forward, functools.partial):
        model.forward.__wrapped__ = types.MethodType(qwen2_causal_model_forward, model)
    else:
        model.forward = types.MethodType(qwen2_causal_model_forward, model)


def disable_hf_flash_attention_check():
    from transformers import PreTrainedModel

    def _enable_flash_attn_2(
        config,
        torch_dtype=None,
        device_map=None,
        check_device_map=True,
        hard_check_only=False,
    ):
        config._attn_implementation = "flash_attention_2"
        return config

    PreTrainedModel._check_and_enable_flash_attn_2 = _enable_flash_attn_2


def patch_hf_model(model: PreTrainedModel, pattern: str, cfg: dict):
    model_name = type(model).__name__
    Logger.log(f"model_name:{model_name}")
    lname = model_name.lower()
    if "llama" in lname:
        patch_llama_attention(model, pattern)
    elif "glm" in lname:
        patch_glm_attention(model, pattern)
    # elif "qwen2" in lname:
    #     patch_qwen2_attention(model, pattern)
    # elif "mis" in lname:
    #     patch_mistral_attention(model, pattern)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    patch_model_config(model, pattern, cfg)
    message = f"# replace flash attention with {pattern} attention #"
    print("#" * len(message))
    print(message)
    print("#" * len(message))


def patch_model(model, pattern: str, config: dict = None):
    """
    Patch the attention mechanism of a Transformers model to use a specified pattern.

    Args:
        model: The model to be patched. Must be a Hugging Face PreTrainedModel.
        pattern: The attention pattern to apply. Supported:
            - 'default', 'flash', 'streaming_llm', 'minfer', 'flex_prefill',
              'vertical_slash', 'anchor_attn', 'anchor_attn_lite', 'get_qkv_tensor',
        config: Configuration for the pattern. See ATTENTION_CONFIG_EXAMPLE.
    """
    if isinstance(model, PreTrainedModel):
        patch_hf_model(model, pattern, config)
    else:
        raise ValueError("Only Hugging Face transformers models are supported")
    return model