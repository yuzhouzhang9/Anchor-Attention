import copy
import json
import types
import functools

from transformers import PreTrainedModel



SUPPORTED_METHODS = {
    "baseline",
    "baseline_flash",
    "baseline_streaming_llm",
    "baseline_minference",
    "baseline_vertical_slash",
    "baseline_flex_prefill",
    "anchorattn",
}


METHOD_ALIASES = {
    "anchorattention": "anchorattn",
}


ATTENTION_CONFIG_EXAMPLE = {
    "baseline": {},
    "baseline_flash": {},
    "baseline_streaming_llm": {
        "global_window": 1024,
        "local_window": 1024 * 8,
    },
    "baseline_minference": {
        "model_type": "llama3.1",
    },
    "baseline_vertical_slash": {
        "block_size": 128,
        "vertical_size": 1024,
        "slash_size": 1024 * 8,
    },
    "baseline_flex_prefill": {
        "block_size": 128,
        "flex_prefill_gamma": 0.95,
        "flex_prefill_tau": 0.1,
        "flex_prefill_min_budget": 1024,
        "flex_prefill_max_budget": None,
    },
    "anchorattn": {
        "block_size_M": 128,
        "theta": 12,
        "step": 16,
    },
}

MINFERENCE_MODEL_ALIASES = {
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "yi": "01-ai/Yi-9B-200K",
}


def normalize_method(method: str) -> str:
    method = METHOD_ALIASES.get(method, method)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown attention method: {method}")
    return method


def get_config_example(method: str = None):
    if method is None:
        return copy.deepcopy(ATTENTION_CONFIG_EXAMPLE)
    return copy.deepcopy(ATTENTION_CONFIG_EXAMPLE[normalize_method(method)])


def _load_config(method: str, cfg: dict | str | None) -> dict:
    if cfg is None:
        return get_config_example(method)
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Configuration string is not valid JSON: {exc}") from exc

    merged = get_config_example(method)
    merged.update(cfg)
    return merged


def patch_model_config(model: PreTrainedModel, method: str, cfg: dict | str | None):
    cfg = _load_config(method, cfg)

    if method == "baseline_minference":
        from baseline.minference.configs.model2path import MODEL2PATH

        model_type = cfg["model_type"]
        model_key = MINFERENCE_MODEL_ALIASES.get(model_type.lower(), model_type)
        try:
            cfg_path = MODEL2PATH[model_key]
        except KeyError as exc:
            raise ValueError(f"Unknown minference model type: {model_type}") from exc
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        setattr(model.config, "minfer_config", cfg)
        return method

    for key, value in cfg.items():
        setattr(model.config, key, value)
    return method


def _llama_forward_for(method: str):
    if method == "baseline_flash":
        from baseline.modules.llama.flash import llama_flash_attention_forward
        return llama_flash_attention_forward
    if method == "baseline_minference":
        from baseline.modules.llama.minference import llama_minfer_attention_forward
        return llama_minfer_attention_forward
    if method == "baseline_vertical_slash":
        from baseline.modules.llama.vertical import llama_vertical_slash_attention_forward
        return llama_vertical_slash_attention_forward
    if method == "baseline_streaming_llm":
        from baseline.modules.llama.streaming import llama_streaming_llm_attention_forward
        return llama_streaming_llm_attention_forward
    if method == "baseline_flex_prefill":
        from baseline.modules.llama.flex import llama_flex_prefill_attention_forward
        return llama_flex_prefill_attention_forward
    if method == "anchorattn":
        from anchorattention.modules.llama.attention import llama_flash_attn2_forward_anchorattn
        return llama_flash_attn2_forward_anchorattn
    raise ValueError(f"Method {method} does not patch Llama attention")


def _qwen2_forward_for(method: str):
    if method == "baseline_flash":
        from baseline.modules.qwen2.flash import qwen2_flash_attention_forward
        return qwen2_flash_attention_forward
    if method == "baseline_minference":
        from baseline.modules.qwen2.minference import qwen2_minfer_attention_forward
        return qwen2_minfer_attention_forward
    if method == "baseline_vertical_slash":
        from baseline.modules.qwen2.vertical import qwen2_vertical_slash_attention_forward
        return qwen2_vertical_slash_attention_forward
    if method == "baseline_streaming_llm":
        from baseline.modules.qwen2.streaming import qwen2_streaming_llm_attention_forward
        return qwen2_streaming_llm_attention_forward
    if method == "baseline_flex_prefill":
        from baseline.modules.qwen2.flex import qwen2_flex_prefill_attention_forward
        return qwen2_flex_prefill_attention_forward
    if method == "anchorattn":
        from anchorattention.modules.qwen2.attention import qwen2_anchorattn_forward
        return qwen2_anchorattn_forward
    raise ValueError(f"Method {method} does not patch Qwen2 attention")


def patch_llama_attention(model: PreTrainedModel, method: str):
    if method == "baseline":
        return

    from transformers.models.llama.modeling_llama import LlamaFlashAttention2, LlamaForCausalLM, LlamaMLP

    assert isinstance(model, LlamaForCausalLM)
    new_forward = _llama_forward_for(method)

    for _, module in model.named_modules():
        if isinstance(module, LlamaFlashAttention2):
            module.forward = types.MethodType(new_forward, module)

    from modules.llama.causal import llama_causal_model_forward
    if isinstance(model.forward, functools.partial):
        model.forward.__wrapped__ = types.MethodType(llama_causal_model_forward, model)
    else:
        model.forward = types.MethodType(llama_causal_model_forward, model)

    from modules.llama.mlp import llama_mlp_forward
    for _, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            module.forward = types.MethodType(llama_mlp_forward, module)


def patch_qwen2_attention(model: PreTrainedModel, method: str):
    if method == "baseline":
        return

    from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2, Qwen2ForCausalLM, Qwen2MLP

    assert isinstance(model, Qwen2ForCausalLM)
    new_forward = _qwen2_forward_for(method)

    for _, module in model.named_modules():
        if isinstance(module, Qwen2FlashAttention2):
            module.forward = types.MethodType(new_forward, module)

    from modules.qwen2.causal import qwen2_causal_model_forward
    if isinstance(model.forward, functools.partial):
        model.forward.__wrapped__ = types.MethodType(qwen2_causal_model_forward, model)
    else:
        model.forward = types.MethodType(qwen2_causal_model_forward, model)

    from modules.qwen2.mlp import qwen2_mlp_forward
    for _, module in model.named_modules():
        if isinstance(module, Qwen2MLP):
            module.forward = types.MethodType(qwen2_mlp_forward, module)


def patch_hf_model(model: PreTrainedModel, method: str, cfg: dict | str | None):
    method = patch_model_config(model, method, cfg)

    model_name = type(model).__name__
    lname = model_name.lower()
    if "llama" in lname:
        patch_llama_attention(model, method)
    elif "qwen2" in lname:
        patch_qwen2_attention(model, method)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    print(f"use {method} attention")


def patch_model(model, pattern: str = "anchorattn", config: dict | str | None = None):
    """Patch a Transformers model with a baseline method or AnchorAttn.

    Supported methods:
        baseline, baseline_flash, baseline_streaming_llm, baseline_minference,
        baseline_vertical_slash, baseline_flex_prefill, anchorattn.
    """
    if not isinstance(model, PreTrainedModel):
        raise ValueError("Only Hugging Face transformers models are supported")

    method = normalize_method(pattern)
    patch_hf_model(model, method, config)
    return model
