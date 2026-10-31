import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr, up_ptr, out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # SwiGLU logic: silu(gate) * up = gate * sigmoid(gate) * up
        sig_gate = tl.sigmoid(gate)
        out = gate * sig_gate * up

        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        grad_out_ptr, gate_ptr, up_ptr,
        dgate_ptr, dup_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig_gate = tl.sigmoid(gate)
        silu_gate = gate * sig_gate

        dsilu = sig_gate * (1.0 + gate * (1.0 - sig_gate))
        
        dgate = grad_out * up * dsilu
        dup = grad_out * silu_gate

        tl.store(dgate_ptr + offsets, dgate.to(dgate_ptr.dtype.element_ty), mask=mask)
        tl.store(dup_ptr + offsets, dup.to(dup_ptr.dtype.element_ty), mask=mask)


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        
        if HAS_TRITON and gate.is_cuda and up.is_cuda:
            out = torch.empty_like(gate)
            n_elements = gate.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            _swiglu_fwd_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=1024)
            return out
        else:
            # CPU Fallback
            return torch.nn.functional.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        
        if HAS_TRITON and grad_output.is_cuda:
            dgate = torch.empty_like(gate)
            dup = torch.empty_like(up)
            n_elements = gate.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            _swiglu_bwd_kernel[grid](
                grad_output, gate, up, dgate, dup, n_elements, BLOCK_SIZE=1024
            )
            return dgate, dup
        else:
            # PyTorch Autograd Fallback
            with torch.enable_grad():
                gate_c = gate.detach().requires_grad_(True)
                up_c = up.detach().requires_grad_(True)
                out = torch.nn.functional.silu(gate_c) * up_c
                out.backward(grad_output)
                return gate_c.grad, up_c.grad


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Публичная функция для вызова SwiGLU оптимизации."""
    return SwiGLUFunction.apply(gate, up)
