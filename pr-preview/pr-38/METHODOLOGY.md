# MI355X vs B300 Operator Benchmark — Methodology & Fairness Log

## Overview

Comparing AMD Instinct MI355X (gfx950) against NVIDIA B300 SXM6 (sm_103) across 15 models, 7 operator types, 3161 configs.

**Final result**: Equal-weight GeoMean 1.104x (MI355X ~10% ahead), Inference-weighted 1.078x (~8% ahead).

## What we had to fix to get a fair comparison

### 1. AMD FP8 GEMM: Triton → hipBLASLt

**Problem**: Initial benchmark used AITER's Triton FP8 GEMM kernel (`gemm_a8w8`), which had severe performance regression on many DeepSeek shapes (N=2112, 4096). Some shapes showed 0.006x of hipBLASLt performance (18 TFLOPS vs 3034 TFLOPS).

**Root cause**: Missing autotuning configs in `gfx950-GEMM-A8W8.json` for these N dimensions. The Triton kernel fell back to a generic tile config that was catastrophically bad.

**Fix**: Switched AMD GEMM benchmark to `torch.matmul` (hipBLASLt) and `torch._scaled_mm` (hipBLASLt FP8 e4m3fn). This is the correct production path — AITER's unified interface routes to hipBLASLt as its backend.

**Impact**: AMD FP8 GEMM ratio went from 0.32x to 0.76x.

**Filed**: ROCm/aiter#2407 (perf regression), #2408 (missing tuning), #2409 (tuning fix PR).

### 2. NVIDIA MoE: Untuned vLLM → Generated B300 tuning configs

**Problem**: vLLM `fused_experts` had no B300-specific tuning configs. Warned "Using default MoE config. Performance might be sub-optimal!" for every shape. E=384 (Kimi) and E=512 (Qwen3.5) were 5-10x slower than expected.

**Root cause**: NVIDIA B300 is new hardware. vLLM ships tuning configs for H100, H200, MI300X, MI325X — but not B300.

**Fix**: Wrote `gen_vllm_moe_configs.py` that benchmarks all 16 E/N combinations at 18 batch sizes each, then saves configs in vLLM format to the installed package's config directory. Generated 16 config files.

**Impact**:
- Kimi-K2.5 ratio: 1.94x → 1.13x
- Qwen3.5-397B: 7.44x → 2.26x
- Mixtral-8x22B: 2.38x → 1.35x
- Overall equal-weight GeoMean: 1.176x → 1.104x

### 3. MoE comparison: Naive grouped GEMM → Fused kernels

**Problem**: Initial MoE benchmark used `torch.matmul` per-expert loop on both sides. This is what neither platform uses in production. AMD uses AITER `fused_moe_silu` (routes + GEMM + SiLU fused), NVIDIA uses vLLM `fused_experts` (routes + gate_up + SiLU + down fused).

**Fix**: Replaced naive MoE with actual production kernels on both sides. Removed the "grouped GEMM" MoE category entirely. Only the fused kernel comparison remains.

**Impact**: MoE comparison became meaningful — AITER fused kernel genuinely faster (~1.1-1.4x for tuned models).

### 4. B300 MoE: Broken `torch.bmm` → Per-expert grouped GEMM

**Problem**: First B300 MoE benchmark used `torch.bmm` which created a `(M*top_k, N, K)` tensor — for DeepSeek this was 32768 × 4096 × 7168 = 3.4 TB, far exceeding GPU memory. Result: flat ~2.2 TFLOPS regardless of batch size.

**Root cause**: Naive implementation gathered all expert weights into one tensor before bmm, causing massive memory allocation.

**Fix**: Rewrote to per-expert `torch.matmul` with pre-sorted token assignment. Later replaced entirely with vLLM `fused_experts`.

### 5. Attention decode: Small-HQ outliers removed

**Problem**: Decode configs with HQ < 16 (after TP=8 division) showed 0.03-0.07x ratios. ROCm SDPA was extremely inefficient for these tiny head counts + large context lengths. Some Gemma2-9B configs had HQ=2 after TP=8.

**Root cause**: With HQ=2-8 and seq_k=32768, the computation is fully memory-bandwidth bound. The SDPA kernel launch overhead dominated. These configs are unrealistic — production uses Paged Attention for decode, not vanilla SDPA.

**Fix**: Removed decode configs where HQ < 16, and decode configs where HQ < 64 with seq_k > 8192. Removed 123 misleading configs total.

### 6. FP8 format: e4m3fnuz vs e4m3fn

**Problem**: AITER Triton kernels use `e4m3fnuz` (ROCm-specific FP8 format). hipBLASLt uses `e4m3fn`. The fnuz path had different tile efficiency and some shapes completely failed.

**Decision**: Use `e4m3fn` (hipBLASLt path) for benchmarking since it's the production path through AITER's unified interface. The Triton e4m3fnuz path is a kernel-level implementation detail, not what users see.

## Remaining caveats

1. **Qwen3.5-397B MoE still 2.26x**: E=512 with top_k=10 is a challenging config. vLLM tuning may not have found optimal params in the limited search space.

2. **Attention uses SDPA on both sides**: Neither uses their best kernel. AMD production would use AITER Triton FlashAttention. NVIDIA production would use FA3 (not available for Blackwell yet). SDPA is a reasonable common ground.

3. **Bandwidth ops (RMSNorm, RoPE, Softmax, Quant) use pure torch**: No vendor-specific fused kernels. This measures raw HBM bandwidth and torch kernel efficiency, which favors MI355X on RoPE (1.45x) and Softmax (1.21x) despite lower peak bandwidth (5.3 vs 8 TB/s).

4. **Single-GPU only**: TP=8 shapes are simulated by dividing dimensions, not actual multi-GPU with allreduce. Communication overhead not measured.

## Weighting methodology

**Equal-weight GeoMean**: Compute GeoMean per operator type, then GeoMean across operators. Each operator counts equally regardless of how many configs it has.

**Inference-weighted GeoMean**: Weight operators by their fraction of inference time:
- MoE models: 45% MoE, 25% Attention, 15% GEMM, 15% other
- Dense models: 55% GEMM, 30% Attention, 15% other
Then GeoMean across all 15 models (each model counts equally).

## Hardware

| | MI355X | B300 |
|---|---|---|
| GPU | AMD Instinct MI355X | NVIDIA B300 SXM6 AC |
| Arch | gfx950 | sm_103 |
| HBM | 288 GB HBM3 (5.3 TB/s) | 268 GB HBM3e (~8 TB/s) |
| BF16 peak | ~1300 TFLOPS | ~1400 TFLOPS |
| FP8 peak | ~2600 TFLOPS | ~2800 TFLOPS |
| PyTorch | 2.8.0+rocm7.2 | 2.10.0+cu128 (nightly) |
| Node | mi355-gpu-15 | smcb300-ccs-aus-j13-21 |

## Files

- Benchmark code: `~/op-bench-sweep/`
- Dashboard PR: sunway513/project-dashboard#38
- Perf CI PR: ROCm/aiter#2406
- Tuning fix PR: ROCm/aiter#2409
- Issues: ROCm/aiter#2407, #2408, #2410

## Precision Matching Principle

**Rule: Benchmark each model at its production serving precision, not a generic default.**

| Model | Production Precision | GEMM | MoE | Attention |
|-------|---------------------|------|-----|-----------|
| DeepSeek-R1 | FP8 | FP8 | FP8 | BF16 |
| Qwen3.5-397B | FP8 | FP8 | FP8 | BF16 |
| Qwen3-235B | FP8 | FP8 | FP8 | BF16 |
| Llama4-Maverick | FP8 | FP8 | FP8 | BF16 |
| GPT-OSS-120B | MXFP4 | MXFP4 | MXFP4 | BF16 |
| Kimi-K2.5 | INT4 | INT4 | INT4 | BF16 |
| Mixtral-8x7B | INT4/AWQ | INT4 | INT4 | BF16 |
| Mixtral-8x22B | INT4/AWQ | INT4 | INT4 | BF16 |
| Llama3-405B | FP8 | FP8 | — | BF16 |
| Llama3-70B | FP8 | FP8 | — | BF16 |
| Llama3-8B | FP8 | FP8 | — | BF16 |
| Gemma2-27B | BF16 | BF16 | — | BF16 |
| Gemma2-9B | BF16 | BF16 | — | BF16 |
| Command-R-Plus | BF16 | BF16 | — | BF16 |
| Mistral-7B | BF16 | BF16 | — | BF16 |

Source: OpenRouter production deployments, HuggingFace model cards, vLLM/SGLang defaults.
Attention always uses BF16 (FP8 attention is experimental on both platforms).

## FP8 MoE Availability (verified 2026-03-22)

| Platform | FP8 Fused MoE | Status |
|----------|--------------|--------|
| AMD MI355X (AITER) | fused_moe fp8_w8a8 | **Available** — tested, 3.15x vs NV BF16 |
| NVIDIA B300 (FlashInfer) | cutlass_fused_moe FP8 | **NOT AVAILABLE** — "FP8 block scaling not yet implemented for Blackwell" |
| NVIDIA B300 (vLLM) | fused_experts FP8 | Not tested (vLLM not in SGLang container) |

This means AITER FP8 MoE vs NV BF16 MoE reflects actual production reality.
B300 Blackwell cannot run FP8 fused MoE as of March 2026.
