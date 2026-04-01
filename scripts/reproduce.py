#!/usr/bin/env python3
"""
One-click reproducibility for every data point in the operator benchmark dashboard.

Usage:
    # Reproduce a specific cell
    python3 reproduce.py --op gemm --model DeepSeek-R1 --M 256 --N 7168 --K 18432 --precision fp8 --gpu amd
    python3 reproduce.py --op gemm --model DeepSeek-R1 --M 256 --N 7168 --K 18432 --precision fp8 --gpu nvidia

    # Generate all commands for a category
    python3 reproduce.py --op gemm --list-all

    # Generate shell script for full reproducibility
    python3 reproduce.py --generate-all --output repro_all.sh

    # Run a specific cell directly
    python3 reproduce.py --op gemm --model DeepSeek-R1 --M 256 --N 7168 --K 18432 --precision bf16 --run
"""

import argparse
import json
import os
import sys
from pathlib import Path


def get_repro_command(op, params, gpu_side):
    """Generate the exact command to reproduce a benchmark data point."""

    if gpu_side == "amd":
        python = "python3"
        preamble = "# Run inside MI355X container (mi355-gpu-15):\n# docker exec -it pensun-jit-test bash\n"
    else:
        python = "~/bench-venv/bin/python3"
        preamble = "# Run on B300 (smcb300-ccs-aus-j13-21.cs-aus.dcgpu):\n"

    M = params.get("M", params.get("batch", 1))

    if op == "gemm":
        precision = params.get("op", "gemm_bf16")
        N = params["N"]
        K = params["K"]
        tp = params.get("tp", 1)

        script = f"""{preamble}{python} -c "
import torch
M, N, K = {M}, {N}, {K}
"""
        if "fp8" in precision and "blockscale" not in precision:
            script += f"""A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
sa = torch.tensor(1.0, device='cuda', dtype=torch.float32)
sb = torch.tensor(1.0, device='cuda', dtype=torch.float32)
fn = lambda: torch._scaled_mm(A, B.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
"""
        elif "blockscale" in precision:
            script += f"""A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
sa = torch.ones(M, (K+127)//128, device='cuda', dtype=torch.float32)
sb = torch.ones(N, (K+127)//128, device='cuda', dtype=torch.float32)
try:
    fn = lambda: torch._scaled_mm(A, B.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    fn()
except:
    sa = torch.tensor(1.0, device='cuda', dtype=torch.float32)
    sb = torch.tensor(1.0, device='cuda', dtype=torch.float32)
    fn = lambda: torch._scaled_mm(A, B.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
"""
        else:  # bf16
            script += f"""A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
fn = lambda: torch.matmul(A, B)
"""
        script += f"""# Benchmark
for _ in range(15): fn()
torch.cuda.synchronize()
import time; t0 = time.time()
for _ in range(80): fn()
torch.cuda.synchronize()
ms = (time.time()-t0)/80*1000
tflops = 2.0*M*N*K / ms * 1e-9
print(f'GEMM {precision} M={{M}} N={{N}} K={{K}}: {{tflops:.2f}} TFLOPS ({{ms:.3f}} ms)')
"
"""
        return script

    elif op == "attention":
        mode = params.get("mode", params.get("op", "prefill"))
        batch = params.get("batch", 1)
        hq = params.get("hq", 128)
        hk = params.get("hk", 8)
        seq_q = params.get("seq_q", 4096)
        seq_k = params.get("seq_k", 4096)
        head_dim = params.get("head_dim", 128)
        is_decode = "decode" in str(mode)

        script = f"""{preamble}{python} -c "
import torch, torch.nn.functional as F
batch, hq, hk, head_dim = {batch}, {hq}, {hk}, {head_dim}
seq_q, seq_k = {seq_q}, {seq_k}
dtype = torch.bfloat16
q = torch.randn(batch, hq, seq_q, head_dim, dtype=dtype, device='cuda')
k = torch.randn(batch, hk, seq_k, head_dim, dtype=dtype, device='cuda')
v = torch.randn(batch, hk, seq_k, head_dim, dtype=dtype, device='cuda')
if hk != hq:
    k = k.repeat_interleave(hq//hk, dim=1)
    v = v.repeat_interleave(hq//hk, dim=1)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    ctx = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    fn = lambda: sdpa_kernel(ctx).__enter__() or None; fn = lambda: None
    def fn():
        with sdpa_kernel(ctx):
            return F.scaled_dot_product_attention(q, k, v, is_causal={'True' if not is_decode else 'False'})
except:
    fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal={'True' if not is_decode else 'False'})
for _ in range(5): fn()
torch.cuda.synchronize()
import time; t0 = time.time()
for _ in range(30): fn()
torch.cuda.synchronize()
ms = (time.time()-t0)/30*1000
flops = 2.0*batch*hq*seq_q*seq_k*(2*head_dim)
mode_str = 'decode' if {is_decode} else 'prefill'
print(f'Attention {{mode_str}} B={{batch}} HQ={{hq}} HK={{hk}} S={{seq_q}}/{{seq_k}}: {{flops/ms*1e-9:.2f}} TFLOPS ({{ms:.3f}} ms)')
"
"""
        return script

    elif op == "moe_fused":
        N = params["N"]
        K = params["K"]
        E = params["E"]
        top_k = params["top_k"]

        if gpu_side == "amd":
            script = f"""{preamble}# Requires AITER installed
cd /opt/aiter/src
PYTHONPATH=/opt/aiter/src {python} -c "
import torch, triton
from op_tests.triton_tests.moe.test_moe import input_helper
from aiter.ops.triton.moe.moe_op_silu_fused import fused_moe_silu
from aiter.ops.triton.utils.types import torch_to_triton_dtype
M, N, K, E, top_k = {M}, {N}, {K}, {E}, {top_k}
dtype = torch.float16
(a,b,out,out_silu,b_zp,a_s,b_s,tw,ti,si,ei,np,cfg) = input_helper(M,N,K,top_k,E,routed_weight=False,dtype=dtype,fp8_w8a8=False,int8_w8a16=False)
fn = lambda: fused_moe_silu(a,b,out_silu,a_s,b_s,b_zp,tw,ti,si,ei,np,False,top_k,torch_to_triton_dtype[dtype],False,False,use_int4_w4a16=False,config=cfg)
ms = triton.testing.do_bench(fn, warmup=10, rep=50)
tflops = 2.0*M*top_k*K*N / ms * 1e-9
print(f'AITER fused_moe_silu M={{M}} N={{N}} K={{K}} E={{E}} top_k={{top_k}}: {{tflops:.2f}} TFLOPS ({{ms:.3f}} ms)')
"
"""
        else:
            script = f"""{preamble}# Requires vLLM installed
{python} -c "
import torch, time
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
M, N, K, E, top_k = {M}, {N}, {K}, {E}, {top_k}
dtype = torch.bfloat16
w1 = torch.randn(E, 2*N, K, dtype=dtype, device='cuda')
w2 = torch.randn(E, K, N, dtype=dtype, device='cuda')
x = torch.randn(M, K, dtype=dtype, device='cuda')
router = torch.randn(M, E, dtype=dtype, device='cuda')
probs = router.float().softmax(dim=-1)
tw, ti = torch.topk(probs, top_k, dim=-1)
tw = (tw / tw.sum(dim=-1, keepdim=True)).to(dtype)
for _ in range(5): fused_experts(x, w1, w2, tw, ti)
torch.cuda.synchronize()
t0 = time.time()
reps = max(5, min(30, 3000//max(1,M)))
for _ in range(reps): fused_experts(x, w1, w2, tw, ti)
torch.cuda.synchronize()
ms = (time.time()-t0)/reps*1000
tflops = 2.0*M*top_k*K*N / ms * 1e-9
print(f'vLLM fused_experts M={{M}} N={{N}} K={{K}} E={{E}} top_k={{top_k}}: {{tflops:.2f}} TFLOPS ({{ms:.3f}} ms)')
"
"""
        return script

    elif op in ("rmsnorm", "rope", "softmax", "quant"):
        N = params.get("N", params.get("K", 7168))

        if op == "rmsnorm":
            script = f"""{preamble}{python} -c "
import torch
M, N = {M}, {N}
x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
w = torch.ones(N, dtype=torch.bfloat16, device='cuda')
eps = 1e-6
def fn():
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w
for _ in range(15): fn()
torch.cuda.synchronize()
import time; t0 = time.time()
for _ in range(80): fn()
torch.cuda.synchronize()
ms = (time.time()-t0)/80*1000
bw = (M*N*2 + N*2 + M*N*2) / (ms*1e-3) * 1e-9
print(f'RMSNorm M={{M}} N={{N}}: {{bw:.1f}} GB/s ({{ms:.4f}} ms)')
"
"""
        elif op == "rope":
            num_heads = params.get("num_heads", 128)
            head_dim = params.get("head_dim", 128)
            seq_len = params.get("seq_len", 4096)
            batch = params.get("batch", 1)
            script = f"""{preamble}{python} -c "
import torch
batch, seq_len, num_heads, head_dim = {batch}, {seq_len}, {num_heads}, {head_dim}
x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
freqs = torch.randn(seq_len, head_dim//2, dtype=torch.bfloat16, device='cuda')
cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)
sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)
def fn():
    x1 = x[..., :head_dim//2]; x2 = x[..., head_dim//2:]
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)
for _ in range(15): fn()
torch.cuda.synchronize()
import time; t0 = time.time()
for _ in range(80): fn()
torch.cuda.synchronize()
ms = (time.time()-t0)/80*1000
bw = batch*seq_len*num_heads*head_dim*2*2 / (ms*1e-3) * 1e-9
print(f'RoPE B={{batch}} S={{seq_len}} H={{num_heads}} D={{head_dim}}: {{bw:.1f}} GB/s ({{ms:.4f}} ms)')
"
"""
        elif op == "softmax":
            script = f"""{preamble}{python} -c "
import torch, torch.nn.functional as F
M, N = {M}, {N}
x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
fn = lambda: F.softmax(x, dim=-1)
for _ in range(15): fn()
torch.cuda.synchronize()
import time; t0 = time.time()
for _ in range(80): fn()
torch.cuda.synchronize()
ms = (time.time()-t0)/80*1000
bw = M*N*2*2 / (ms*1e-3) * 1e-9
print(f'Softmax M={{M}} N={{N}}: {{bw:.1f}} GB/s ({{ms:.4f}} ms)')
"
"""
        elif op == "quant":
            script = f"""{preamble}{python} -c "
import torch
M, K = {M}, {N}
x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
def fn():
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / 448.0
    return (x / scale).to(torch.float8_e4m3fn), scale
for _ in range(15): fn()
torch.cuda.synchronize()
import time; t0 = time.time()
for _ in range(80): fn()
torch.cuda.synchronize()
ms = (time.time()-t0)/80*1000
bw = (M*K*2 + M*K*1 + M*4) / (ms*1e-3) * 1e-9
print(f'FP8 Quant M={{M}} K={{K}}: {{bw:.1f}} GB/s ({{ms:.4f}} ms)')
"
"""
        return script

    return f"# Unsupported op: {op}\n"


def generate_all_commands(data, output_path):
    """Generate a shell script with all reproducibility commands."""
    with open(output_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Operator Benchmark Reproducibility Script\n")
        f.write("# Generated from op-perf.json\n")
        f.write("# Each section can be run independently\n\n")

        for cat in data["categories"]:
            op = cat["id"]
            f.write(f"\n{'#'*60}\n# {cat['name']} ({len(cat['results'])} configs)\n{'#'*60}\n\n")

            for i, r in enumerate(cat["results"]):
                for gpu in ["amd", "nvidia"]:
                    cmd = get_repro_command(op, r, gpu)
                    f.write(f"# --- {cat['name']} #{i+1} ({gpu.upper()}) ---\n")
                    f.write(f"# Model: {r.get('model','')} | ")
                    if op == "gemm":
                        f.write(f"M={r.get('M','')} N={r.get('N','')} K={r.get('K','')} {r.get('op','')}\n")
                    elif op == "attention":
                        f.write(f"B={r.get('batch','')} HQ={r.get('hq','')} sq={r.get('seq_q','')} sk={r.get('seq_k','')}\n")
                    elif op == "moe_fused":
                        f.write(f"M={r.get('M','')} N={r.get('N','')} E={r.get('E','')} top_k={r.get('top_k','')}\n")
                    else:
                        f.write(f"M={r.get('M','')} N={r.get('N','')}\n")
                    f.write(cmd)
                    f.write("\n")

    os.chmod(output_path, 0o755)
    print(f"Generated {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce any benchmark data point")
    parser.add_argument("--op", choices=["gemm", "attention", "moe_fused", "rmsnorm", "rope", "softmax", "quant"])
    parser.add_argument("--model", type=str)
    parser.add_argument("--M", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--E", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--gpu", choices=["amd", "nvidia", "both"], default="both")
    parser.add_argument("--run", action="store_true", help="Execute the command")
    parser.add_argument("--generate-all", action="store_true", help="Generate full repro script")
    parser.add_argument("--output", type=str, default="repro_all.sh")
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    data_path = args.data or str(Path(__file__).parent.parent / "project-dashboard-pr" / "docs" / "_data" / "op-perf.json")
    if not os.path.exists(data_path):
        data_path = str(Path(__file__).parent.parent / "docs" / "_data" / "op-perf.json")

    if args.generate_all:
        with open(data_path) as f:
            data = json.load(f)
        generate_all_commands(data, args.output)
        return

    if not args.op:
        parser.print_help()
        return

    params = {}
    if args.M: params["M"] = args.M
    if args.N: params["N"] = args.N
    if args.K: params["K"] = args.K
    if args.E: params["E"] = args.E
    if args.top_k: params["top_k"] = args.top_k
    if args.precision: params["op"] = f"gemm_{args.precision}"
    if args.model: params["model"] = args.model

    sides = ["amd", "nvidia"] if args.gpu == "both" else [args.gpu]
    for side in sides:
        print(f"\n{'='*50}")
        print(f"{side.upper()} command:")
        print(f"{'='*50}")
        cmd = get_repro_command(args.op, params, side)
        print(cmd)

        if args.run:
            print("Running...")
            os.system(cmd)


if __name__ == "__main__":
    main()
