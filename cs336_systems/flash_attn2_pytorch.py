import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from einops import einsum

class MyFlashAttn(torch.autograd.Function):
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
        query_block_size = 4
        key_block_size = 2
        

        q_tiles = q.view(batch_size, n_q//query_block_size, query_block_size, -1)
        k_tiles = k.view(batch_size, n_k//key_block_size, key_block_size, -1)
        v_tiles = v.view(batch_size, n_k//key_block_size, key_block_size, -1)
        n_q_tiles = n_q//query_block_size
        n_k_tiles = n_k//key_block_size
        
        result = torch.empty(batch_size, n_q, d)
        log_sumexp = torch.empty(batch_size, n_q)
        
        for i in range(n_q_tiles):
            m = torch.ones(batch_size, query_block_size) * float("-inf")
            l = torch.zeros(batch_size, query_block_size)
            o = torch.zeros(batch_size, query_block_size, d)
            Q = q_tiles[:, i].squeeze()
            q_inds = torch.arange(i*query_block_size, (i+1) * query_block_size)
            for j in range(n_k_tiles):
                k_inds = torch.arange(j*key_block_size, (j+1)*key_block_size)
                valid = q_inds[:, None] < k_inds[None, :] # Q * K

                valid = valid.unsqueeze(0) # Make it broadcastable

                K = k_tiles[:, j].squeeze()
                V = v_tiles[:, j].squeeze()
                s = einsum(Q, K, "... q d, ... k d -> ... q k") / np.sqrt(d)
                if is_causal:
                    s = s.masked_fill(valid, float("-inf"))
                row_max, _ = torch.max(s, dim=-1)
                m_ij = torch.maximum(row_max, m)
                p = torch.exp(s - m_ij.unsqueeze(-1))
                l = torch.exp(m - m_ij) * l + torch.sum(p, dim=-1)
                o = torch.exp(m - m_ij).unsqueeze(-1) * o + einsum(p, V, "... q k, ... k d -> ... q d")
                m = m_ij
            o = o / l.unsqueeze(-1)
            log_sum_i = m + torch.log(l)
            log_sumexp[:, i * query_block_size: (i+1) * query_block_size] = log_sum_i
            result[:, i * query_block_size: (i+1) * query_block_size, :] = o

        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, log_sumexp)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, log_sumexp = ctx.saved_tensors
        d = q.shape[-1]
        s = einsum(q, k, "... q d, ... k d -> ... q k") / np.sqrt(d)

        p = torch.exp(s - log_sumexp[..., None])

        dv = einsum(p, grad_output, "... q k, ... q d -> ... k d")
        dp = einsum(grad_output, v, "... q d, ... k d -> ... q k")

        D = torch.sum(p * dp, dim=-1)
        ds = p * (dp - D[..., None])
        dq = einsum(ds, k, "... q k, ... k d -> ... q d") / np.sqrt(d)
        dk = einsum(ds, q, "... q k, ... q d -> ... k d") / np.sqrt(d)
        return dq, dk, dv, None
