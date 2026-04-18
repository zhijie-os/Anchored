import math
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.cache_utils import DynamicCache

from transformers.models.mistral.modeling_mistral import MistralAttention

def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    total_k = min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1))
    last_page = chunk_attn_weights.shape[-1] - 1

    # 2. Prevent Page 0 and Last Page from being selected by topk
    chunk_attn_weights[..., 0] = float('-inf')
    chunk_attn_weights[..., last_page] = float('-inf')

    # 3. Calculate the remaining dynamic budget (k - 5)
    # Using max(0, ...) ensures it never requests a negative budget
    dynamic_k = max(0, total_k - 5)

    # 4. Search for the best remaining pages
    if dynamic_k > 0:
        _, topk = chunk_attn_weights.topk(k=dynamic_k, dim=-1)
    else:
        # Create an empty tensor if all budget was diverted to anchors
        topk = torch.empty(
            (chunk_attn_weights.shape[0], chunk_attn_weights.shape[1], chunk_attn_weights.shape[2], 0), 
            dtype=torch.long, device=chunk_attn_weights.device
        )
    # =========================================================================

    # Initialize the binary mask to entirely False
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

    # Only apply the dynamic indices if we actually found any
    if dynamic_k > 0:
        # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
        topk = topk.unsqueeze(-1).repeat(
            1, 1, 1, 1, chunk_size
        ) * chunk_size + torch.arange(chunk_size, device=topk.device)
        topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
        
        # Turn the selected switches to True
        mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    
    # New cache format
    if isinstance(past_key_value, DynamicCache):
        kv_seq_len = past_key_value.get_seq_length()
    # Legacy cache format
    else:
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            assert isinstance(past_key_value, tuple)
            kv_seq_len += past_key_value[0].shape[-2]
    
    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    # New cache format
    if isinstance(past_key_value, DynamicCache):
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
    # Legacy cache format
    else:
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    postive_query = query_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        quantized_weight = quantized_weight + attention_mask
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )

    token_budget = min(kv_seq_len, self.token_budget)

    attn_weights_for_selection = quantized_weight

    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, self.chunk_size
        )  # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

    # keep the first page and last page data alive
    mask_bottom[..., :self.chunk_size] = True
    mask_bottom[..., -self.chunk_size:] = True

    mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    #
    k = self.token_budget // self.chunk_size
    if k >= 3:
        # 2. Extract the raw tensor data for the 16 tokens of Page 0 and Last Page
        page_0_attn = attn_weights[..., :self.chunk_size]
        last_page_attn = attn_weights[..., -self.chunk_size:]
        
        # Note: value_states has the sequence length on dim=-2
        page_0_v = value_states[..., :self.chunk_size, :]
        last_page_v = value_states[..., -self.chunk_size:, :]

        extra_attn = []
        extra_v = []

        # 3. Append physical clones based on your exact math
        # (The base sequence already contains 1 copy of Front and 1 copy of Back)
        if k == 3:
            # Goal: 1 Front, 2 Back -> Add 1 copy of Back
            extra_attn.extend([last_page_attn])
            extra_v.extend([last_page_v])
        elif k == 4:
            # Goal: 2 Front, 2 Back -> Add 1 copy of Front, 1 copy of Back
            extra_attn.extend([page_0_attn, last_page_attn])
            extra_v.extend([page_0_v, last_page_v])
        elif k >= 5:
            # Goal: 2 Front, 3 Back -> Add 1 copy of Front, 2 copies of Back
            extra_attn.extend([page_0_attn, last_page_attn, last_page_attn])
            extra_v.extend([page_0_v, last_page_v, last_page_v])

        # 4. Concatenate them to the sequence dimensions
        if extra_attn:
            orig_len = attn_weights.shape[-1] # Capture the original length
            # attn_weights sequence is on dim=-1
            attn_weights = torch.cat([attn_weights] + extra_attn, dim=-1)
            # value_states sequence is on dim=-2
            value_states = torch.cat([value_states] + extra_v, dim=-2)
            if getattr(self, "layer_id", -1) == 0: # turn this into 2 for printout
                print(f"\n[DEBUG] Budget K={k} | Added {len(extra_attn)} extra pages. KV Length Grew: {orig_len} -> {attn_weights.shape[-1]}", flush=True)
    # =========================================================================
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


global layer_id
layer_id = 32


def enable_quest_attention_eval(model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_eval(
                module,
                args,
            )

        global layer_id
        if isinstance(module, (LlamaAttention, MistralAttention)):
            # For longchat model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                forward, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
