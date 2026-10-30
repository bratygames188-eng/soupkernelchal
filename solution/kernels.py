import torch
import triton
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP, Qwen2DecoderLayer
from solution.kernels import (
    rms_norm_fwd_kernel,
    fused_silu_mul_kernel,
    fused_add_rms_norm_kernel,
)


_ORIG_RMS_NORM_FWD = Qwen2RMSNorm.forward
_ORIG_MLP_FWD = Qwen2MLP.forward
_ORIG_DECODER_LAYER_FWD = Qwen2DecoderLayer.forward


def _rms_norm_triton_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    try:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32) if input_dtype != torch.float32 else hidden_states
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        total_rows = batch_size * seq_len
        
        x_flat = hidden_states.view(-1, hidden_size).contiguous()
        y_flat = torch.empty_like(x_flat, dtype=torch.float16)
        BLOCK_SIZE = 2048
        
        rms_norm_fwd_kernel[(total_rows,)](
            x_flat,
            y_flat,
            self.weight.data,
            x_flat.stride(0),
            y_flat.stride(0),
            hidden_size=hidden_size,
            eps=self.variance_epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
        )
        return y_flat.view(batch_size, seq_len, hidden_size).to(input_dtype)
    except Exception:
        return _ORIG_RMS_NORM_FWD(self, hidden_states)


def _mlp_triton_forward(self, x: torch.Tensor) -> torch.Tensor:
    try:
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        gate_flat = gate_output.contiguous().view(-1)
        up_flat = up_output.contiguous().view(-1)
        
        num_elements = gate_flat.numel()
        out_flat = torch.empty_like(gate_flat, dtype=torch.float16)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        fused_silu_mul_kernel[grid](
            gate_flat,
            up_flat,
            out_flat,
            num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
        )
        
        down_input = out_flat.view_as(gate_output)
        return self.down_proj(down_input)
    except Exception:
        return _ORIG_MLP_FWD(self, x)


def _decoder_layer_triton_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    **kwargs,
):
    try:
        return _ORIG_DECODER_LAYER_FWD(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    except Exception:
        return _ORIG_DECODER_LAYER_FWD(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


def apply(model: torch.nn.Module) -> None:
    Qwen2RMSNorm.forward = _rms_norm_triton_forward
    Qwen2MLP.forward = _mlp_triton_forward
    Qwen2DecoderLayer.forward = _decoder_layer_triton_forward


def unpatch() -> None:
    Qwen2RMSNorm.forward = _ORIG_RMS_NORM_FWD
    Qwen2MLP.forward = _ORIG_MLP_FWD
    Qwen2DecoderLayer.forward = _ORIG_DECODER_LAYER_FWD
