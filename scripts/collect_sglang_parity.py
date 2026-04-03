#!/usr/bin/env python3
"""Collect sglang CI test parity data (CUDA vs AMD) from registered test files.

Parses sglang's test/registered/ directory using ci_register.py to determine
which tests are registered for CUDA, AMD, or both.  Applies known
platform-incompatible patterns to identify exclusive tests, then computes
a parity percentage.

Usage:
    python scripts/collect_sglang_parity.py --sglang-dir /path/to/sglang
    python scripts/collect_sglang_parity.py --clone
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sglang"

# ---------------------------------------------------------------------------
# AMD-incompatible test patterns (NVIDIA-only features)
# Matched against the file **basename** via substring.
# ---------------------------------------------------------------------------
AMD_INCOMPATIBLE_PATTERNS = [
    # FlashInfer - NVIDIA-only attention backend
    "flashinfer",
    "test_mla_flashinfer",
    "test_flashmla",
    "test_moe_wna16",
    "test_cutedsl_moe",
    # FlashAttention 3/4 - NVIDIA H100+/B200+ only
    "test_fa3",
    "flash_attention_4",
    "test_hybrid_attn_backend",
    # CUDA Graph - weak_ref_tensor not implemented for ROCm
    "test_piecewise_cuda_graph",
    # EAGLE Speculative Decoding - requires FA3/FlashInfer
    "test_eagle_infer",
    "test_eagle_dp_attention",
    "test_ngram_speculative_decoding",
    "test_standalone_speculative_decoding",
    # MLA - private models or NVIDIA-specific
    "test_mla_fp8",
    "test_mla_int8_deepseek_v3",
    # ModelOpt - NVIDIA quantization toolkit
    "modelopt",
    # Quantization methods not supported on ROCm
    "test_bnb",
    "test_gguf",
    "test_gptqmodel_dynamic",
    "test_quantization.py",
    "test_awq.py",
    "test_marlin_moe",
    "test_autoround",
    "test_fp8_utils",
    "test_w4a8_deepseek_v3",
    # VLM issues on ROCm
    "test_vision_openai_server_a",
    "test_vlm_input_format",
    "test_encoder_embedding_models",
    # Disaggregation tests - various ROCm issues
    "test_disaggregation_different_tp",
    "test_disaggregation_pp",
    "test_disaggregation_dp_attention",
    "test_epd_disaggregation",
    # Hardware-specific tests (B200/H100/GB200 only)
    "4-gpu-b200",
    "4-gpu-h100",
    "4-gpu-gb200",
    "test_deepseek_v3_fp4_4gpu",
    "test_gpt_oss_4gpu",
    "test_llama31_fp4",
    "test_fp8_blockwise_gemm",
    "test_deepseek_v3_cutedsl",
    "test_pp_single_node",
    "test_multi_instance_release_memory",
    "test_qwen3_next_models",
    "test_deepseek_v3_fp4_mtp_small",
    "ep/test_deepep",
    # Other NVIDIA-specific features
    "trtllm",
    "cutlass",
    "mamba",
    "nvidia_nemotron",
    "torchao",
    "test_deepseek_v32",
    "test_mimo_models",
]

# Detailed exclusion reasons for dashboard display (keyed by basename)
AMD_EXCLUSION_REASONS = {
    "test_piecewise_cuda_graph_small_1_gpu.py": "NotImplementedError: weak_ref_tensor is implemented only for CUDA and NPU",
    "test_piecewise_cuda_graph_large_1_gpu.py": "NotImplementedError: weak_ref_tensor is implemented only for CUDA and NPU",
    "test_flashmla.py": "Private model (lmsys/sglang-ci-dsv3-test)",
    "test_mla_flashinfer.py": "Private model (lmsys/sglang-ci-dsv3-test)",
    "test_mla_int8_deepseek_v3.py": "Private model (sgl-project/sglang-ci-dsv3-block-int8-test)",
    "test_encoder_embedding_models.py": "Requires FlashInfer",
    "test_fp8_utils.py": "Requires deepGEMM",
    "test_ngram_speculative_decoding.py": "Requires FA3, FlashInfer",
    "test_standalone_speculative_decoding.py": "Requires FA3, FlashInfer",
    "test_modelopt_loader.py": "modelopt_fp8 quantization not supported on ROCm",
    "test_fa3.py": "Requires FA3, FlashInfer",
    "test_flash_attention_4.py": "Requires FA3, FlashInfer",
    "test_hybrid_attn_backend.py": "Requires FA3, FlashInfer",
    "test_autoround.py": "gptq_marlin_repack not defined",
    "test_eagle_infer_a.py": "EAGLE not supported",
    "test_eagle_infer_b.py": "EAGLE not supported",
    "test_eagle_infer_beta.py": "EAGLE not supported",
    "test_eagle_dp_attention.py": "FA3 import error",
    "test_vision_openai_server_a.py": "ROCm compiler error (uint32_t)",
    "test_vlm_input_format.py": "Assertion error on ROCm",
    "test_cutedsl_moe.py": "FlashInfer not available",
    "test_awq.py": "awq_marlin quantization mismatch",
    "test_marlin_moe.py": "sgl_kernel.moe_wna16_marlin_gemm not found",
    "test_bnb.py": "bitsandbytes not supported on ROCm",
    "test_gptqmodel_dynamic.py": "torch.bfloat16 not supported for gptq",
    "test_quantization.py": "gptq_shuffle not defined",
    "test_gguf.py": "gguf quantization not supported on ROCm",
    "test_mimo_models.py": "Test failed on ROCm",
    "test_w4a8_deepseek_v3.py": "w4afp8 quantization not supported on ROCm",
    "test_disaggregation_different_tp.py": "Test failed on ROCm",
    "test_disaggregation_pp.py": "Timeout on ROCm",
    "test_disaggregation_dp_attention.py": "Private model not accessible",
    "test_deepseek_v3_fp4_mtp_small.py": "Requires 4-GPU B200",
    "test_qwen3_next_models.py": "Requires 4-GPU H100",
    "test_gpt_oss_4gpu.py": "Requires 4-GPU H100/B200",
    "test_multi_instance_release_memory_occupation.py": "Requires 4-GPU H100",
    "test_pp_single_node.py": "Requires 4-GPU H100",
    "test_epd_disaggregation.py": "Requires 4-GPU H100",
    "test_deepep_small.py": "Requires 4-GPU H100",
    "test_deepseek_v3_fp4_4gpu.py": "Requires 4-GPU B200",
    "test_fp8_blockwise_gemm.py": "Requires 4-GPU B200",
    "test_llama31_fp4.py": "Requires 4-GPU B200",
    "test_deepseek_v3_cutedsl_4gpu.py": "Requires 4-GPU GB200",
}

# ---------------------------------------------------------------------------
# CUDA-incompatible test patterns (AMD/ROCm-only features)
# Matched against the file **basename** via substring.
# ---------------------------------------------------------------------------
CUDA_INCOMPATIBLE_PATTERNS = [
    "aiter",
]

CUDA_EXCLUSION_REASONS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def is_amd_incompatible(filename: str) -> tuple[bool, str]:
    """Check if a test file is incompatible with AMD (NVIDIA-exclusive).

    Matches against the file basename only.
    Returns (is_incompatible, reason).
    """
    basename = Path(filename).name

    if basename in AMD_EXCLUSION_REASONS:
        return True, AMD_EXCLUSION_REASONS[basename]

    for pattern in AMD_INCOMPATIBLE_PATTERNS:
        if pattern in basename:
            return True, f"Matched pattern: {pattern}"

    return False, ""


def is_cuda_incompatible(filename: str) -> tuple[bool, str]:
    """Check if a test file is incompatible with CUDA (AMD/ROCm-exclusive).

    Matches against the file basename only.
    Returns (is_incompatible, reason).
    """
    basename = Path(filename).name

    if basename in CUDA_EXCLUSION_REASONS:
        return True, CUDA_EXCLUSION_REASONS[basename]

    for pattern in CUDA_INCOMPATIBLE_PATTERNS:
        if pattern in basename:
            return True, f"Matched pattern: {pattern}"

    return False, ""


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def setup_sglang_imports(sglang_dir: str):
    """Add sglang's ci_register module to sys.path."""
    ci_path = Path(sglang_dir) / "python" / "sglang" / "test" / "ci"
    if not ci_path.exists():
        print(f"Error: ci_register.py not found at {ci_path}", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, str(ci_path))


def collect_all_tests(registered_dir: str):
    """Collect all CI registrations from the registered directory."""
    from ci_register import ut_parse_one_file

    files = glob.glob(f"{registered_dir}/**/*.py", recursive=True)
    all_tests = []

    for file in sorted(files):
        try:
            registries = ut_parse_one_file(file)
            all_tests.extend(registries)
        except Exception as e:
            print(f"Warning: Failed to parse {file}: {e}", file=sys.stderr)

    return all_tests


def get_folder_name(filename: str) -> str:
    """Extract subfolder name under test/registered/."""
    parts = Path(filename).parts
    if "registered" in parts:
        idx = parts.index("registered")
        if idx + 1 < len(parts) - 1:
            return parts[idx + 1]
    return "root"


def get_commit_sha(sglang_dir: str) -> str:
    """Get the current HEAD commit SHA of the sglang repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=sglang_dir,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def clone_sglang(dest: str) -> str:
    """Shallow-clone sglang into dest and return the path."""
    repo_url = "https://github.com/sgl-project/sglang.git"
    clone_dir = os.path.join(dest, "sglang")
    print(f"Cloning sglang (shallow) into {clone_dir}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, clone_dir],
        check=True,
    )
    return clone_dir


# ---------------------------------------------------------------------------
# Parity calculation
# ---------------------------------------------------------------------------

def compute_parity(tests, sglang_dir: str) -> dict:
    """Compute CUDA vs AMD parity from CIRegistry list."""
    from ci_register import HWBackend

    # --- Backend summary ---
    by_backend: dict[str, list] = defaultdict(list)
    for t in tests:
        by_backend[t.backend.name].append(t)

    backend_summary = {}
    for backend in ["CUDA", "AMD"]:
        bt = by_backend.get(backend, [])
        if not bt:
            continue
        b_total = len(bt)
        b_disabled = sum(1 for t in bt if t.disabled)
        b_enabled = b_total - b_disabled
        b_per_commit = sum(1 for t in bt if not t.nightly and not t.disabled)
        b_nightly = sum(1 for t in bt if t.nightly and not t.disabled)
        backend_summary[backend] = {
            "total": b_total,
            "enabled": b_enabled,
            "disabled": b_disabled,
            "per_commit": b_per_commit,
            "nightly": b_nightly,
        }

    # --- Group registrations by unique file ---
    file_backends: dict[str, dict] = defaultdict(lambda: {
        "backends": set(),
        "cuda_info": None,
        "amd_info": None,
    })

    for t in tests:
        rel_path = str(Path(t.filename).relative_to(sglang_dir))
        entry = file_backends[rel_path]
        entry["backends"].add(t.backend)
        if t.backend == HWBackend.CUDA and entry["cuda_info"] is None:
            entry["cuda_info"] = t
        if t.backend == HWBackend.AMD and entry["amd_info"] is None:
            entry["amd_info"] = t

    # --- Classify each file ---
    cuda_exclusive = []
    amd_exclusive = []

    total_cuda = 0
    total_amd = 0

    for filepath, info in sorted(file_backends.items()):
        backends = info["backends"]
        has_cuda = HWBackend.CUDA in backends
        has_amd = HWBackend.AMD in backends

        if has_cuda:
            total_cuda += 1
        if has_amd:
            total_amd += 1

        basename = Path(filepath).name
        folder = get_folder_name(filepath)

        if has_cuda and not has_amd:
            incomp, reason = is_amd_incompatible(filepath)
            if incomp:
                t = info["cuda_info"]
                cuda_exclusive.append({
                    "file": basename,
                    "path": filepath,
                    "folder": folder,
                    "suite": t.suite if t else "",
                    "est_time": t.est_time if t else 0,
                    "reason": reason,
                })
        elif has_amd and not has_cuda:
            incomp, reason = is_cuda_incompatible(filepath)
            if incomp:
                t = info["amd_info"]
                amd_exclusive.append({
                    "file": basename,
                    "path": filepath,
                    "folder": folder,
                    "suite": t.suite if t else "",
                    "est_time": t.est_time if t else 0,
                    "reason": reason,
                })

    denom = total_cuda - len(cuda_exclusive)
    numer = total_amd - len(amd_exclusive)
    parity_pct = round(numer / denom * 100, 1) if denom > 0 else 0

    commit_sha = get_commit_sha(sglang_dir)

    return {
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit_sha": commit_sha,
        "backend_summary": backend_summary,
        "summary": {
            "cuda": total_cuda,
            "amd": total_amd,
            "cuda_exclusive": len(cuda_exclusive),
            "amd_exclusive": len(amd_exclusive),
            "parity_pct": parity_pct,
        },
        "cuda_exclusive": cuda_exclusive,
        "amd_exclusive": amd_exclusive,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect sglang CI test parity (CUDA vs AMD)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sglang-dir",
        help="Path to local sglang checkout",
    )
    group.add_argument(
        "--clone",
        action="store_true",
        help="Shallow-clone sglang into a temp directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/sglang/ci_parity.json)",
    )
    args = parser.parse_args()

    if args.clone:
        tmpdir = tempfile.mkdtemp()
        sglang_dir = clone_sglang(tmpdir)
    else:
        sglang_dir = os.path.abspath(args.sglang_dir)

    if not os.path.isdir(sglang_dir):
        print(f"Error: {sglang_dir} is not a directory", file=sys.stderr)
        return 1

    registered_dir = os.path.join(sglang_dir, "test", "registered")
    if not os.path.isdir(registered_dir):
        print(f"Error: {registered_dir} not found", file=sys.stderr)
        return 1

    setup_sglang_imports(sglang_dir)

    print(f"Parsing test registrations from {registered_dir}...")
    tests = collect_all_tests(registered_dir)
    print(f"  Found {len(tests)} registrations")

    report = compute_parity(tests, sglang_dir)

    # Print backend summary
    print(f"\n{'=' * 70}")
    print(f"SGLang CI Parity Report (commit: {report['commit_sha']})")
    print(f"{'=' * 70}")

    bs = report["backend_summary"]
    print(f"\n{'Backend':<10} {'Total':>6} {'Enabled':>8} {'Disabled':>9} {'Per-Commit':>11} {'Nightly':>8}")
    print("-" * 56)
    for backend in ["CUDA", "AMD"]:
        if backend not in bs:
            continue
        b = bs[backend]
        print(f"{backend:<10} {b['total']:>6} {b['enabled']:>8} {b['disabled']:>9} {b['per_commit']:>11} {b['nightly']:>8}")

    # Print parity summary
    s = report["summary"]
    print(f"\n{'=' * 70}")
    print(f"CUDA vs AMD Parity")
    print(f"{'=' * 70}")
    print(f"  CUDA:                      {s['cuda']}")
    print(f"  AMD:                       {s['amd']}")
    print(f"  CUDA exclusive:            {s['cuda_exclusive']}")
    print(f"  AMD exclusive:             {s['amd_exclusive']}")
    print(f"  Parity: (AMD - AMD_excl) / (CUDA - CUDA_excl)")
    print(f"          ({s['amd']} - {s['amd_exclusive']}) / ({s['cuda']} - {s['cuda_exclusive']}) = {s['parity_pct']}%")

    # Write JSON
    output_path = args.output or str(DATA / "ci_parity.json")
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
