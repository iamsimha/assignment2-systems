# benchmark_flash_attn2.py
import math
import torch
import torch.nn.functional as F
import triton.testing as testing
import numpy as np
from torch import Tensor
from einops import einsum
from jaxtyping import Float
from cs336_basics.nn_ops import softmax

from flash_attn2_triton import MyFlashAttnTriton


def scaled_dot_product_attn(Q: Float[Tensor, "... q d"],
                            K: Float[Tensor, "... k d"],
                            V: Float[Tensor, "... k d"],
                            mask):
    d = Q.shape[-1]
    dot_prod = einsum(Q, K, "... q d, ... k d -> ... q k")
    dot_prod.masked_fill(~mask, float("-inf"))
    attn = softmax(dot_prod/np.sqrt(d), dim=-1)
    return einsum(attn, V, "... q k, ... k d -> ... q d")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_inputs(seq_len: int, d: int, dtype, device="cuda"):
    """
    Create Q, K, V and a causal mask.

    Q, K, V: (B=1, L, D)
    mask: (1, L, L), used ONLY for the naive baseline.
    """
    B = 1
    q = torch.randn(B, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, seq_len, d, device=device, dtype=dtype, requires_grad=True)

    causal = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    ).unsqueeze(0)  # (1, L, L)

    return q, k, v, causal


# ---------------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------------

def bench_case(seq_len: int, d: int, dtype, device="cuda"):
    """
    Benchmark one (seq_len, d, dtype) triple for:

      - Triton FA2 forward/backward
      - Naive PyTorch (einsum+softmax) forward/backward

    Returns a dict:
      {
        'triton': {...},
        'torch_naive': {...},
      }
    """
    torch.cuda.synchronize()
    q, k, v, mask = make_inputs(seq_len, d, dtype, device=device)

    results = {}

    # ---------- Triton implementation ---------- #

    def triton_fwd():
        with torch.no_grad():
            MyFlashAttnTriton.apply(q, k, v, True)

    def triton_fwd_bwd():
        # Avoid gradient accumulation between benchmark iterations
        q.grad = k.grad = v.grad = None
        out = MyFlashAttnTriton.apply(q, k, v, True)
        loss = out.sum()
        loss.backward()

    triton_fwd_ms = testing.do_bench(triton_fwd)
    triton_fwd_bwd_ms = testing.do_bench(triton_fwd_bwd)
    triton_bwd_ms = triton_fwd_bwd_ms - triton_fwd_ms

    results["triton"] = {
        "fwd_ms": triton_fwd_ms,
        "bwd_ms": triton_bwd_ms,
        "fwd_bwd_ms": triton_fwd_bwd_ms,
    }

    torch.cuda.synchronize()

    # ---------- Naive PyTorch (einsum + softmax) ---------- #

    def torch_naive_fwd():
        with torch.no_grad():
            scaled_dot_product_attn(q, k, v, mask)

    def torch_naive_fwd_bwd():
        q.grad = k.grad = v.grad = None
        out = scaled_dot_product_attn(q, k, v, mask)
        loss = out.sum()
        loss.backward()

    torch_naive_fwd_ms = testing.do_bench(torch_naive_fwd)
    torch_naive_fwd_bwd_ms = testing.do_bench(torch_naive_fwd_bwd)
    torch_naive_bwd_ms = torch_naive_fwd_bwd_ms - torch_naive_fwd_ms

    results["torch_naive"] = {
        "fwd_ms": torch_naive_fwd_ms,
        "bwd_ms": torch_naive_bwd_ms,
        "fwd_bwd_ms": torch_naive_fwd_bwd_ms,
    }

    torch.cuda.synchronize()

    return results


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    device = "cuda"

    # seq_len: powers of 2 from 128 up to 65536
    seq_lens = [2 ** i for i in range(7, 17)]   # 128 .. 65536
    d_models = [128]
    # precisions
    dtypes = [torch.float32]

    header = (
        f"{'seq':>7} {'d':>5} {'dtype':>9}  "
        f"{'impl':>12}  {'fwd_ms':>10} {'bwd_ms':>10} {'fwd+bwd_ms':>12}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in seq_lens:
        for d in d_models:
            for dtype in dtypes:
                dtype_name = "bf16" if dtype is torch.bfloat16 else "fp32"

                try:
                    res = bench_case(seq_len, d, dtype, device=device)
                except RuntimeError as e:
                    msg = str(e)
                    if "out of memory" in msg.lower():
                        print(
                            f"{seq_len:7d} {d:5d} {dtype_name:>9}  "
                            f"{'OOM':>12}  {'-':>10} {'-':>10} {'-':>12}"
                        )
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

                for impl_key, impl_label in [
                    ("triton", "triton"),
                    ("torch_naive", "torch_naive"),
                ]:
                    impl_res = res.get(impl_key, None)
                    if impl_res is None:
                        # e.g. FLASH_ATTENTION backend not available
                        print(
                            f"{seq_len:7d} {d:5d} {dtype_name:>9}  "
                            f"{impl_label:>12}  "
                            f"{'-':>10} {'-':>10} {'-':>12}"
                        )
                        continue

                    fwd_ms = impl_res["fwd_ms"]
                    bwd_ms = impl_res["bwd_ms"]
                    fwd_bwd_ms = impl_res["fwd_bwd_ms"]

                    print(
                        f"{seq_len:7d} {d:5d} {dtype_name:>9}  "
                        f"{impl_label:>12}  "
                        f"{fwd_ms:10.3f} {bwd_ms:10.3f} {fwd_bwd_ms:12.3f}"
                    )


if __name__ == "__main__":
    main()
