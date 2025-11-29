import torch
import triton
import triton.language as tl
import numpy as np
import pdb
from jaxtyping import Float
from torch import Tensor
from einops import einsum

DEVICE = torch.device(f"cuda:0")

@triton.autotune(
    configs=[
        triton.Config(
            {"Q_TILE_SIZE": 16, "K_TILE_SIZE": 16},
            num_stages=3, num_warps=4
        ),        
        triton.Config(
            {"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64},
            num_warps=4, num_stages=3
        ),
        triton.Config(
            {"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64},
            num_stages=4, num_warps=8
        ),
        triton.Config(
            {"Q_TILE_SIZE": 16, "K_TILE_SIZE": 32},
            num_stages=4, num_warps=8
        ),
    ],
    key=["N_QUERIES", "N_KEYS"],
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    o_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) - float("inf")
    l_acc = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    i = 0
    q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    max_valid_keys = (query_tile_index + 1)* Q_TILE_SIZE if is_causal else N_KEYS
    for _ in range(0, max_valid_keys, K_TILE_SIZE):
        k = tl.trans(tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero"))
        v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        
        s = tl.dot(q, k) / scale
        
        if is_causal:
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_indices[:, None] >= k_indices[None, :]
            s = tl.where(mask == 1, s, float('-inf'))
        
        row_max = tl.max(s, axis=-1)
        m_ij = tl.maximum(row_max, m)

        p = tl.exp(s - m_ij[:, None])
        p = p.to(v.dtype)
        f = tl.exp(m - m_ij)

        l_acc = f * l_acc + tl.sum(p, axis=-1)
        
        o_acc = f[:, None] * o_acc
        o_acc = tl.dot(p, v, o_acc)
        
        m = m_ij
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        i += 1

    o_acc = o_acc / l_acc[:, None]
    log_sum_i = m + tl.log(l_acc)

    tl.store(pointer=O_block_ptr, value=o_acc, boundary_check=(0,1))
    tl.store(pointer=L_block_ptr, value=log_sum_i, boundary_check=(0,))



class MyFlashAttnTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: Float[Tensor, "... q d"],
                    k: Float[Tensor, "... k d"],
                    v: Float[Tensor, "... v d"],
                    is_causal=False):
        assert len(q.shape) == 3
        assert len(k.shape) == 3


        batch_size = q.shape[0]
        d = q.shape[-1]
        n_q = q.shape[1]
        n_k = k.shape[1]
        o = torch.zeros((batch_size, n_q, d), dtype=torch.float32, device=DEVICE)
        l = torch.zeros((batch_size, n_q), dtype=torch.float32, device=DEVICE)

        stride_qb, stride_qq, stride_qd = q.stride()
        stride_kb, stride_kk, stride_kd = k.stride()
        stride_vb, stride_vv, stride_vd = v.stride()
        stride_ob, stride_oq, stride_od = o.stride()
        stride_lb, stride_lq = l.stride()
        def grid(meta):
            return (
                triton.cdiv(n_q, meta["Q_TILE_SIZE"]),
                batch_size,
            )
        flash_fwd_kernel[grid](
            q, k, v,
            o, l,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vv, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            n_q, n_k,
            np.sqrt(d),
            d,
            is_causal=is_causal
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, l)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, log_sumexp = ctx.saved_tensors
        d = q.shape[-1]
        n_q = q.shape[-2]
        n_k = k.shape[-2]
        s = einsum(q, k, "... q d, ... k d -> ... q k") / np.sqrt(d)
        if ctx.is_causal:
            mask = torch.arange(n_q)[:, None] >= torch.arange(n_k)[None, :]
            mask = mask.to(s.device)
            s = torch.where(mask == 1, s, float("-inf"))
        

        p = torch.exp(s - log_sumexp[..., None])

        dv = einsum(p, grad_output, "... q k, ... q d -> ... k d")
        dp = einsum(grad_output, v, "... q d, ... k d -> ... q k")

        D = torch.sum(p * dp, dim=-1)
        ds = p * (dp - D[..., None])
        dq = einsum(ds, k, "... q k, ... k d -> ... q d") / np.sqrt(d)
        dk = einsum(ds, q, "... q k, ... q d -> ... k d") / np.sqrt(d)
        return dq, dk, dv, None
