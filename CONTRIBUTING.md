# Contributing to Project Dashboard

This repo is the **source of truth** for AMD GPU ecosystem unit test and parity data. The dashboard tracks 13 projects across upstream and internal repos. Your contributions ensure the data is accurate and complete for every project.

## Projects We Track

### Upstream Projects — Automated CI Collection

These 6 projects have automated test result collection via `scripts/collect_tests.py`, pulling ROCm vs CUDA pass rates from GitHub Actions daily at 8am UTC:

| Project | ROCm Workflow | CUDA Workflow | Data Level |
|---------|--------------|---------------|------------|
| **pytorch** | `rocm-mi300` | `trunk` | Test-level (JUnit XML) |
| **sglang** | `Nightly Test (AMD)` | `PR Test` | Job-level |
| **triton** | `Integration Tests` (AMD filter) | `Integration Tests` (NVIDIA filter) | Job-level |
| **jax** | `CI - Bazel ROCm tests` | `CI - Bazel H100 and B200 CUDA tests` | Job-level |
| **xla** | `CI ROCm` | `CI` (GPU L4 filter) | Job-level |
| **transformer_engine** | `TransformerEngine CI` | — (ROCm only) | Job-level |

**What the team contributes for these projects:**

1. **Validate automated data** — review `data/<project>/test_results.json` for accuracy. If CI workflows change (renamed, new IDs, different job filters), update the `WORKFLOWS` dict in `scripts/collect_tests.py`.

2. **Submit parity data** — automated collection captures pass/fail rates, but **parity analysis** (ROCm vs CUDA test-by-test comparison) requires manual CSV submission. Currently only PyTorch has a parity parser (`scripts/collect_parity.py`). For other projects, submit parity results as described below.

3. **Fix manual overrides** — when automated collection fails or is inaccurate (e.g., JAX CUDA workflows that gate on `changed_files` and show all-skipped), submit corrected data via `config/test_results_manual.yaml`.

### Upstream Projects — Manual Data Needed

| Project | Status | What's Needed |
|---------|--------|---------------|
| **vllm** | No `test_results.json` | CI is on Buildkite (not GitHub Actions). Submit results via `test_results_manual.yaml` |
| **llvm** | No test data | Identify relevant AMDGPU test workflows and add to `collect_tests.py` or submit manually |

### Internal Projects — No Test Data Yet

These active-development projects have PR/issue/release tracking but **no test results or parity data**:

| Project | Repo | What's Needed |
|---------|------|---------------|
| **aiter** | ROCm/aiter | Test results + parity data |
| **atom** | ROCm/ATOM | Test results + parity data |
| **mori** | ROCm/mori | Test results + parity data |
| **flydsl** | ROCm/FlyDSL | Test results + parity data |
| **migraphx** | ROCm/AMDMIGraphX | Test results + parity data |

## How to Contribute Data

### 1. PyTorch Unit Test Parity (existing parser)

Run the `frameworks-internal` parity suite and submit the parsed results:

```bash
# 1. Obtain CSV from parity.sh
#    (produces all_tests_status.csv)

# 2. Parse into dashboard JSON
python3 scripts/collect_parity.py \
    --csv /path/to/all_tests_status.csv \
    --sha <pytorch_commit_sha> \
    --arch mi300 \
    --date 2026-03-24

# 3. Commit both generated files
git add data/pytorch/parity_report.json data/pytorch/parity_history.json
```

**Input CSV columns** (from `parity.sh`):

| Column | Description |
|--------|-------------|
| `test_file`, `test_class`, `test_name` | Test identifier |
| `work_flow_name` | CI workflow that ran the test |
| `skip_reason` | Why skipped (if applicable) |
| `assignee` | Person investigating |
| `status_set1` | **ROCm** result: `PASSED` / `SKIPPED` / `MISSED` / `FAILED` |
| `status_set2` | **CUDA** result: `PASSED` / `SKIPPED` / `MISSED` / `FAILED` |

**Output metrics** (`parity_report.json`):
- Per-workflow ROCm/CUDA test counts, skipped/missed/rocmonly breakdowns
- `parity_pct` = `(1 - gap / total_cuda) × 100`
- Top skip reasons, running time comparison

### 2. Test Results for Any Project (manual YAML)

For projects without automated collection, or to override bad automated data, edit `config/test_results_manual.yaml`:

```yaml
vllm:
  rocm:
    workflow_name: "ROCm Buildkite"
    run_url: "https://buildkite.com/vllm/..."
    run_date: "2026-03-24T12:00:00Z"
    conclusion: "success"
    summary:
      total_jobs: 10
      passed: 9
      failed: 1
      skipped: 0
      pass_rate: 90.0
  cuda:
    workflow_name: "CUDA Buildkite"
    run_url: "https://buildkite.com/vllm/..."
    run_date: "2026-03-24T12:00:00Z"
    conclusion: "success"
    summary:
      total_jobs: 12
      passed: 12
      failed: 0
      skipped: 0
      pass_rate: 100.0
```

This works for **any** project — vllm, aiter, atom, mori, flydsl, migraphx, llvm, or overrides for the automated projects.

### 3. Add/Fix Automated Collection for a Project

To add a new project to automated CI collection or fix an existing one:

**a.** Add workflow config in `scripts/collect_tests.py`:

```python
WORKFLOWS = {
    ...
    "aiter": {
        "rocm": {
            "workflow_id": <GitHub Actions workflow ID>,
            "name": "ROCm CI",
        },
        "cuda": {
            "workflow_id": <workflow_id>,
            "name": "CUDA CI",
        },
    },
}
```

**b.** Register the project in `AUTOMATED_PROJECTS` (same file):

```python
AUTOMATED_PROJECTS = {
    ...
    "aiter": lambda cfg: collect_job_level("aiter", cfg),
}
```

**c.** Ensure the project exists in `config/projects.yaml` with its repo path.

To find a workflow ID: go to the repo's Actions tab → click the workflow → the ID is in the URL (`/workflows/<id>`).

### 4. Add a New Project to the Dashboard

Edit `config/projects.yaml`:

```yaml
  newproject:
    repo: org/repo-name
    role: upstream_watch    # or active_dev
    track_authors: []
    track_labels: ["relevant-label"]
    track_keywords: [ROCm, AMD]
    keyword_scope: title
    depends_on: []
    build_workflows: []
```

Then add test collection via path 2 (manual YAML) or path 3 (automated).

## PR Workflow

1. **Branch** from `main`:
   ```bash
   git checkout -b data/<your-name>/<description>
   ```

2. **Make changes** — data files, config, or scripts

3. **Commit** with a clear prefix:
   ```bash
   git commit -m "data: update pytorch parity for mi300 (2026-03-24)"
   ```

4. **Push and open a PR**:
   ```bash
   git push origin data/<your-name>/<description>
   ```

5. On merge → dashboard auto-deploys via GitHub Pages

### Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Data submission | `data/<name>/<desc>` | `data/pensun/mi300-parity-w12` |
| Script changes | `scripts/<name>/<desc>` | `scripts/ljin1/add-aiter-collection` |
| Config changes | `config/<name>/<desc>` | `config/wenchen2/add-new-project` |
| Dashboard/docs | `docs/<name>/<desc>` | `docs/jnair/fix-trend-chart` |

### Commit Prefixes

- `data:` — raw data updates or parity submissions
- `scripts:` — collection/parsing script changes
- `config:` — project configuration or manual overrides
- `docs:` — dashboard UI or documentation
- `ci:` — GitHub Actions workflow changes

## Repository Structure

```
├── config/
│   ├── projects.yaml              # 13 tracked projects
│   └── test_results_manual.yaml   # Manual test result overrides
├── data/
│   └── <project>/
│       ├── prs.json               # (automated) Pull requests
│       ├── issues.json            # (automated) Issues
│       ├── releases.json          # (automated) Releases
│       ├── activity.json          # (automated) PR velocity, contributors
│       ├── build_times.json       # (automated) CI build times
│       ├── test_results.json      # CI test pass rates (automated or manual)
│       ├── parity_report.json     # UT parity snapshot (team-submitted)
│       └── parity_history.json    # UT parity trend (team-submitted)
├── scripts/
│   ├── collect.py                 # PRs, issues, releases (automated)
│   ├── collect_tests.py           # CI test results (automated, 6 projects)
│   ├── collect_activity.py        # Activity metrics (automated)
│   ├── collect_build_times.py     # Build time stats (automated)
│   ├── collect_parity.py          # Parity CSV → JSON (PyTorch, manual input)
│   ├── snapshot.py                # Weekly trend snapshots
│   └── render.py                  # Generate dashboards + site data
├── docs/                          # GitHub Pages interactive dashboard
└── dashboards/                    # Markdown dashboard views
```

## Data Pipeline

```
Daily (8am UTC, automated):
  GitHub API → collect.py → prs/issues/releases
  GitHub Actions → collect_tests.py → test_results (6 upstream projects)
  Metrics → collect_activity.py → PR velocity, CI health
  Aggregate → snapshot.py → weekly trends
  Render → render.py → dashboards + site data → GitHub Pages

Team PRs (manual):
  parity.sh CSV → collect_parity.py → parity_report + parity_history
  Manual YAML → test_results_manual.yaml → merged into test_results
  New workflows → collect_tests.py WORKFLOWS dict → automated going forward
```

## Questions?

Open an issue or reach out to the team.
