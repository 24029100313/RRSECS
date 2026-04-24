# AGENTS.md

> This file is the immutable contract between the human researcher and Codex.
> Mutable state lives in `PROJECT_PROGRESS.md`. Do not put TBD fields here.

---

## 0. Project identity

This repository is used for RRSECS research: **Referring Remote Sensing Expression Comprehension and Segmentation**.

Task definition:

```text
remote sensing image + referring expression -> bounding box + pixel-level mask
```

The long-term research direction is **not** CCFormer++.
The target method is:

```text
Shared canonical semantic-geometric representation
+ task-specific optimization routing
+ canonical Fourier-contour / residual contour representation
```

The intended final model derives:

```text
box  = Rect(C_img)
mask = Rasterize/Fill(C_img)
```

from the same transformed contour `C_img`, rather than predicting `bbox` and `mask` through unrelated task heads.

Working name for the method: **CanFoR-RRSECS**.

---

## 1. Research objective

We start from the CCFormer codebase because it provides the RRSECS dataset interface, training/eval framework, Swin/BERT backbone, and current strong baseline.

The final paper must gradually replace CCFormer's core innovation points:

| CCFormer component              | Final replacement direction                               | Stage |
| ------------------------------- | --------------------------------------------------------- | ----- |
| Mask2Former-style mask head     | canonical Fourier-contour / residual contour decoder      | V1    |
| MSDeformAttnTransformerVG       | transform / contour-to-box localization router            | V1    |
| LGDM / LAGD / gate_decoupler    | semantic-geometric task-specific optimization routing     | V2    |
| MCFM / CCFormer_fusion          | lightweight language-conditioned scale fusion (LCSF)      | V3    |
| cross-task collaborative / CCLoss | boundary-reconstruction-consistency losses (BRC)        | V3    |

**Do not** present the future method as "CCFormer + one extra module".
The final model must have its own representation, routing, loss, and ablation story.

---

## 2. Session bootstrap protocol (required)

**Every Codex session must begin with the following sequence. No experiment logged without this bootstrap is valid.**

1. `view AGENTS.md` — read this file in full.
2. `view PROJECT_PROGRESS.md` — read current state.
3. `git rev-parse HEAD` and `git status --short` — record current commit and dirty state.
4. **Compute `git hash-object AGENTS.md` and `git hash-object PROJECT_PROGRESS.md`.**
5. **Append a new row to the `## Session log` table in `PROJECT_PROGRESS.md`** with:
   - timestamp (UTC);
   - AGENTS.md hash;
   - PROJECT_PROGRESS.md hash;
   - git commit hash;
   - planned experiment IDs for this session;
   - one-sentence plan.
6. **Cite at least one specific line-number from AGENTS.md in the first response**, to prove the file was actually read (e.g. "per AGENTS.md §5.3, I will not modify baseline loss").

Sessions without this bootstrap are treated as `failed / invalid` and their outputs must not be used.

---

## 3. Hardware constraints

Available hardware:

```text
8 x NVIDIA V100, 16 GB per GPU
```

Rules:

- Start all smoke tests on **single GPU**.
- Start with **batch size 1** for forward/backward smoke tests.
- Prefer **AMP fp16** for training.
- Do **not** assume A100 / H100 / 4090 memory.
- Report **peak memory** for every train/profile experiment.
- For DDP, verify single-GPU correctness first, then 2-GPU, then 4-GPU, then 8-GPU. **Never jump straight to 8-GPU.**
- If the node is shared, run `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv` before any training command and log the free memory; if another user occupies > 2 GB on a target GPU, skip that GPU.

### 3.1 Known V100-16GB risk for CCFormer baseline

CCFormer uses Mask2Former-style pixel decoder + Swin-Tiny + 512×512 input. Training with batch size 1 may OOM on V100-16GB.

Escalation ladder if OOM occurs at S1-E06 (backward smoke test):

1. Enable AMP fp16.
2. Enable gradient checkpointing on Swin stages.
3. Reduce image size to 448 (**log as deviation, not as benchmark run**).
4. Enable gradient accumulation (accum_steps = 2, effective batch unchanged).
5. **Do not change architecture.**
6. **Do not silently reduce `num_queries` or similar design parameters.**

Every deviation must be appended to the `## Deviations` table in `PROJECT_PROGRESS.md`.

---

## 4. Working rules for Codex

Every session, Codex must:

1. Execute the session bootstrap protocol (§2).
2. State a short plan before editing or running long commands.
3. Inspect code before editing. Prefer reversible changes.
4. Keep baseline code and new-method code separate (§6).
5. Update `PROJECT_PROGRESS.md` after every code audit, profile, smoke test, train run, eval run, or failed attempt.
6. When reporting, always include:
   - changed files;
   - commands run;
   - key logs (not full logs; extract failures and final metrics);
   - metrics;
   - peak memory;
   - wall-clock time;
   - **confidence label (§8)**;
   - reproducibility notes;
   - next action.
7. **Cite file paths and line numbers** when reporting any code fact.
8. **Never fabricate metrics.** When unsure, use confidence label `to verify` or `failed / invalid`.
9. **Never report `estimated` numbers as `measured`.**
10. **Never treat ablation split as full benchmark.**
11. Ask before deleting files, renaming large directories, or broad refactors.

---

## 5. Hard stops (Codex must NOT do these without asking)

These are absolute red lines. Violating any of them invalidates the session.

1. Do not run any command that downloads > 100 MB without asking.
2. Do not run any training command with `--epochs > 5` during Stage 1. Stage 1 is profile, not training.
3. Do not `pip install` packages without recording in the `## Installed packages` table of `PROJECT_PROGRESS.md`.
4. Do not write to any path outside `{repo}`, `/tmp`, `outputs/`, `experiments/`, or explicitly allowed directories.
5. Do not modify `configs/recs.yaml`. Create `configs/recs_baseline_*.yaml` instead.
6. Do not `git commit` or `git push`. Leave all git write operations to the user.
7. Do not run `eval_recs` without a real checkpoint. Do not fabricate checkpoint paths.
8. If a command has been running > 30 minutes without output, kill it and report. Do not wait silently.
9. If `PROJECT_PROGRESS.md` shows an existing blocker for the current stage, do not proceed past that stage until the blocker is resolved.
10. **If 3 consecutive experiments in the same stage fail, STOP and report before attempting a 4th.**
11. Do not delete or rewrite rows in `PROJECT_PROGRESS.md` that have confidence `failed / invalid` (§9).
12. Do not patch CCLoss into the baseline without recording it as a separate `fair-baseline-with-ccloss` variant (§7.3).
13. Do not merge new-method configs into baseline configs.

---

## 6. File modification policy

### 6.1 Allowed in Stage 0–1 without asking

- Documentation: `AGENTS.md`, `PROJECT_PROGRESS.md`, `outputs/stage1_exit_report.md`.
- Non-invasive audit/profile scripts under:
  - `tools/stage1_*.py`
  - `tools/profile_*.py`
  - `tools/trace_*.py`
- Debug/baseline configs under:
  - `configs/stage1_*.yaml`
  - `configs/recs_baseline_*.yaml`
- Output folders under:
  - `outputs/stage1_*/`
  - `experiments/stage1_*/`

### 6.2 Not allowed in Stage 1 unless explicitly requested

- Modifying baseline model internals.
- Deleting original CCFormer files.
- Renaming core directories.
- Merging new-method configs into baseline configs.
- Silently changing dataset paths.
- Changing loss implementation "to improve results" before the active-loss audit is logged.
- Patching CCLoss into the baseline without creating a separate variant config.

### 6.3 Allowed after Stage 2+ with a clear plan

- New method modules under a clear namespace:
  - `models/canfor/`
  - `models/heads/canonical_fourier_head.py`
  - `losses/canfor_losses.py`
- New method configs:
  - `configs/canfor_v1_*.yaml`
  - `configs/canfor_v2_*.yaml`
  - `configs/canfor_v3_*.yaml`
- Stage-specific tools:
  - `tools/mask_to_polygon.py`
  - `tools/efd_fit.py`
  - `tools/eval_representation.py`

### 6.4 Tool script contract

Every `tools/stage1_*.py` and `tools/profile_*.py` script must satisfy all of the following:

1. **Read-only by default.** Must not write to `configs/`, must not modify model files, must not start training.
2. **Starts with** `import torch; torch.set_grad_enabled(False)` unless explicitly profiling backward.
3. **Accepts `--dry-run` flag.** In dry-run mode, the script prints every path it would read and every module it would instantiate, but does not actually read/instantiate. The first use of any new script must be with `--dry-run`.
4. **Structured output.** Prints a Markdown table or JSON to stdout and **also** saves to `outputs/stage1_<experiment_id>/audit.json` (and/or `audit.md`).
5. **Self-identifying header.** First lines of stdout must be:
   ```text
   # script: tools/stage1_xxx.py
   # git commit: <hash>
   # started: <UTC ISO8601>
   # dry-run: true/false
   ```
6. **No `os.system`, no `subprocess` on arbitrary strings.** If a subprocess is needed, use `subprocess.run` with a list of arguments.
7. **No silent exception swallowing.** Any `except` must log the exception with traceback to the output file.

### 6.5 PROJECT_PROGRESS.md update discipline

`PROJECT_PROGRESS.md` sections are classified as:

- **append-only** (new rows/entries only, never rewrite or delete existing rows):
  - Session log
  - Experiment overview table
  - Detailed experiment records
  - Decisions made
  - Deviations
  - Installed packages
- **updatable** (may be overwritten):
  - Current one-line status
  - Project snapshot
  - Current best known facts
  - Current blockers
  - Next immediate action
  - Stage checklist (checkbox toggles only; do not delete checkboxes)
- **never-delete** (preserved forever for post-mortem):
  - Any row whose Confidence is `failed / invalid`.
  - Any row in the Deviations table.

When updating `PROJECT_PROGRESS.md`, Codex must use `str_replace` on specific sections, never regenerate the whole file.

---

## 7. Split / loss / protocol discipline

### 7.1 Split classification (strict, mechanical)

Codex must classify the active split into exactly one of:

- **`ablation-subset`** if any of:
  - path contains any of `{abla_splits, ablation, subset, /10/, /50/, debug, tiny}`; OR
  - train sample count < 5000.
- **`full-train`** if:
  - path does not match any ablation pattern above; AND
  - train sample count matches the documented RefDIOR full-train size (record expected value as `to verify` until confirmed against the paper).
- **`unknown`** otherwise.

If `unknown`, Codex **must not** proceed to S1-E12 (baseline reproduction) and must raise this as a blocker in `PROJECT_PROGRESS.md`.

Ablation split results **must never** be reported in the `# Baseline reproduction results` table as full-benchmark numbers. Create a separate `## Ablation-subset sanity runs` table for them.

### 7.2 Loss effectiveness (three-column rule)

Grep alone is not enough. For each candidate loss term, Codex must record three independent pieces of evidence:

| Loss term | Defined in class? | Called in forward? | Weight > 0 in active config? | Effective? |
| --------- | ----------------- | ------------------ | ---------------------------- | ---------- |
| `loss_ris` | `losses/recs_loss.py:42` | `losses/recs_loss.py:118` | `configs/recs.yaml:L_ris=1.0` | **yes** |

"Effective" is `yes` only if all three columns are `yes`. A loss that is defined but wrapped in `if self.use_cc_loss` where `use_cc_loss=False` is **not effective** and must be recorded as such.

### 7.3 Fair baseline variant

If the active `RECSLoss` turns out to exclude `CCLoss` but the paper reports results with `CCLoss`:

- The as-found active baseline is labeled `baseline-as-found`.
- A variant with `CCLoss` patched in (using the paper's specification) is labeled `baseline-fair-with-ccloss`.
- Both must be trained and reported separately. Neither may be silently conflated with the paper's reported numbers.
- The `CCLoss` patch is **not** a contribution of this work.

---

## 8. Confidence labels

Use exactly these labels. No others.

```text
code-measured
paper-reported
estimated
sanity-only
failed / invalid
to verify
```

Rules:

- **`code-measured`**: produced by code in this repo and logged with command + config + commit hash. Full protocol, not subset.
- **`paper-reported`**: copied from a paper; cite source (arxiv ID, page, table) in notes.
- **`estimated`**: formula or rough calculation. Must include the formula. Never report as measured.
- **`sanity-only`**: smoke test or tiny subset (< 5000 train samples, or < 5 epochs). Not benchmark.
- **`failed / invalid`**: command failed, protocol invalid, or bootstrap missing.
- **`to verify`**: suspected but not confirmed by evidence.

### 8.1 FLOPs reconciliation rule

Report FLOPs from **at least 2 tools** (choose 2 from: torchinfo, fvcore, thop, ptflops, torch.profiler).

- If the 2 tools agree within 15%: report the range and label `code-measured`.
- If they disagree by > 15%: cross-check against a manual estimate for Swin-Tiny (~4.5 G at 224×224; scale quadratically to 512×512 ≈ 23.5 G) and BERT-base (~22 G per forward at L=21). If still no reconciliation, label `to verify` and report all three numbers.
- **Never report a single number.** Always report range and tool list.
- For any module that fvcore lists as "unsupported ops", explicitly note it and fall back to torch.profiler.

---

## 9. Stage workflow overview

### Stage 0 — Documentation and repo safety setup

Exit criteria: `AGENTS.md` and `PROJECT_PROGRESS.md` exist; session bootstrap works; `## Session log` has at least one valid entry.

### Stage 1 — CCFormer baseline reproduction, code audit, profiling

Stage 1 does **not** pursue SOTA. It establishes a trustworthy baseline and a replacement map. Detailed task groups in §10.

### Stage 2 — Representation-only feasibility study

No model training. Compare representations on GT masks offline. Numerical go/kill thresholds in §11.

### Stage 3 — V1 minimal head replacement

Keep Swin / BERT / CCFormer_fusion / LAGD. Replace mask + box heads with canonical Fourier-contour head. Derive box/mask from contour. 5% then 10% subset.

### Stage 4 — V2 task-specific routing replacement

Delete/bypass LAGD. Add shared canonical representation + localization router + segmentation router. Prove routing improves over unified no-routing head.

### Stage 5 — V3 full method (paper version)

Replace MCFM / CCFormer_fusion → LCSF. Replace LGDM / LAGD → SGOR. Replace CCLoss → BRC. Keep dataset + training + eval framework. Produce a method that does not depend on CCFormer's core innovations.

---

## 10. Stage 1 detailed protocol

### 10.1 Task groups

| ID | Name | Invasive? | Can run before previous? |
| --- | --- | --- | --- |
| S1-E00 | Environment audit | read-only | first |
| S1-E01 | Repo entrypoint audit | read-only | after E00 |
| S1-E02 | Split audit | read-only | after E01 |
| S1-E03 | Active loss audit (three-column) | read-only | after E01 |
| S1-E04 | Single-batch dataloader test | read-only | after E02 |
| S1-E04.5 | Checkpoint loading sanity | read-only | after E01 |
| S1-E05 | Single-GPU forward smoke | non-invasive | after E04, E04.5 |
| S1-E06 | Single-GPU backward smoke | non-invasive | after E05 |
| S1-E07 | Tiny-subset overfit | non-invasive | after E06 |
| S1-E08 | Eval pipeline smoke | non-invasive | after E07 |
| S1-E09 | Tensor shape tracing | non-invasive | after E05 |
| S1-E10 | Params / FLOPs / memory profile | non-invasive | after E05 |
| S1-E11a | 2-GPU DDP sanity | non-invasive | after E07 |
| S1-E11b | 4-GPU DDP sanity | non-invasive | after E11a |
| S1-E11c | 8-GPU DDP sanity | non-invasive | after E11b |
| S1-E12 | Baseline reproduction short run | non-invasive | after E11c |
| S1-E13 | Replacement readiness audit | read-only | after E09, E10 |

### 10.2 Checkpoint loading sanity (S1-E04.5)

Before any forward smoke test, verify:

- Swin pretrained weights: file exists, size matches a documented checksum if available, loads without `missing_keys` covering core stages.
- BERT pretrained weights: same.
- RECS-specific pretrained checkpoint (if any): count `missing_keys` and `unexpected_keys`.
- **If `missing_keys` covers attention, fusion, or head modules → BLOCKER.** Do not proceed to forward smoke.

### 10.3 DDP ladder (S1-E11)

Run with `NCCL_DEBUG=INFO` for every DDP smoke test in Stage 1.

```bash
# 2-GPU first
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py --config configs/stage1_smoke_ddp.yaml
# 4-GPU only after 2-GPU passes
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO torchrun --nproc_per_node=4 train.py --config configs/stage1_smoke_ddp.yaml
# 8-GPU only after 4-GPU passes
NCCL_DEBUG=INFO torchrun --nproc_per_node=8 train.py --config configs/stage1_smoke_ddp.yaml
```

### 10.4 Stage 1 exit criteria

Stage 1 is complete only when all of the following are true:

- [ ] Repo inventory complete (all entrypoints resolved to file:line).
- [ ] Environment audit complete (Python, PyTorch, CUDA, GPUs recorded).
- [ ] Split classified as `ablation-subset`, `full-train`, or `unknown` (§7.1).
- [ ] Active loss audit complete with three-column evidence (§7.2).
- [ ] Dataloader smoke passes.
- [ ] Checkpoint loading sanity passes.
- [ ] Forward smoke passes on single GPU.
- [ ] Backward smoke passes on single GPU (or deviation logged).
- [ ] Tiny-subset overfit passes.
- [ ] Eval smoke passes (or failure logged with confidence `failed / invalid`).
- [ ] Tensor shape table complete (§ shape trace in PROJECT_PROGRESS).
- [ ] Params/FLOPs/memory profile has at least two tools agreeing within 15%, or reconciliation documented.
- [ ] 2/4/8-GPU DDP ladder complete.
- [ ] Baseline reproduction short run complete on active split.
- [ ] Replacement map complete.
- [ ] **`outputs/stage1_exit_report.md` produced** (§10.5).

### 10.5 Stage 1 exit report (required)

Before marking Stage 1 complete, produce `outputs/stage1_exit_report.md` containing:

1. **Active chain confirmed**: file:line for each of `{config, train entry, dataset builder, model builder, visual encoder, text encoder, fusion, LAGD/decoupler, mask head, box head, loss, eval}`.
2. **Active split**: one of `{ablation-subset, full-train, unknown}` with sample counts and path.
3. **Active loss**: three-column effectiveness table for every candidate loss term. Flag whether `CCLoss` is effective.
4. **Baseline metrics**: mIoU, oIoU, P@0.5 on the active split, with confidence label. Use `sanity-only` if subset.
5. **Profile summary**: params, FLOPs (range across ≥2 tools), peak memory (training), peak memory (inference), latency per forward.
6. **Top-3 replaceable modules by FLOPs**: name, file:line, FLOPs, % of total.
7. **Blockers remaining**: list or "none".
8. **Recommended Stage 2 first action**: one paragraph.

Stage 2 must not start without this report.

---

## 11. Stage 2 numerical go/kill thresholds

Stage 2 compares representations on GT masks. No model training.

### 11.1 Representations to evaluate

- dense mask oracle (GT)
- axis-aligned box fill (HBB)
- rotated box fill (OBB)
- raw polygon, N ∈ {20, 40, 80, 120}
- CBCS polygon, N ∈ {20, 40, 80, 120}
- EFD, K ∈ {4, 8, 16, 32, 64}
- canonical contour: axis-aligned bbox / rotated bbox / PCA / learned orientation
- canonical EFD, K ∈ {4, 8, 16, 32, 64}
- EFD + residual contour, K ∈ {16, 32}, M ∈ {16, 32, 64}
- low-res SDF / occupancy 32² and 64²

### 11.2 Metrics

reconstruction mIoU, oIoU, Boundary IoU, Boundary F-score, Chamfer distance, 95% Hausdorff, scalar count, conversion failure rate, disconnected-component failure rate, orientation ambiguity rate; category-wise and scale-wise breakdown.

### 11.3 Go / kill table

| Condition (on RefDIOR, on GT masks) | Decision |
| --- | --- |
| EFD K=32 mIoU vs CBCS N=80 mIoU gap ≤ 3 pts **AND** Boundary F gap ≤ 5 pts | **GO**: EFD as main coarse representation |
| EFD K=32 gap 3–7 pts, but EFD K=32 + residual M=32 closes gap to ≤ 3 pts vs CBCS N=80 | **GO**: EFD + residual as main route |
| EFD K=64 + residual M=64 still > 5 pts behind CBCS N=80 | **KILL pure EFD**; pivot to canonical contour + residual polygon |
| airplane / bridge / harbor / windmill / slender classes: EFD fails in > 20% of samples | **KILL pure EFD**; keep residual contour mandatory |
| canonical EFD improves mIoU by ≥ 2 pts **or** reduces Chamfer by ≥ 10% vs image-space EFD | **GO canonical** |
| Canonicalization orientation ambiguity rate > 10% | **DROP PCA/rotated canonical**; use axis-aligned or learned orientation only |
| Low-res SDF 64² clearly outperforms EFD+residual | Pivot to SDF auxiliary, but do not make it the main dense route |
| Box fill / rotated box match contour representations closely | Shapes too regular; paper story shifts to efficiency/consistency, not contour fidelity |

---

## 12. Code navigation checklist

Codex should inspect these first and fill exact `file:line` locations in `PROJECT_PROGRESS.md`. Listed path patterns are hypotheses to verify, not ground truth.

```text
configs/recs.yaml                            # to verify
train.py                                     # to verify
<dataset builder>                            # create_dataset function
<model builder>                              # recs_model_builder
SwinTransformer                              # visual encoder
CCFormer_fusion                              # fusion module
MCM                                          # fusion sub-module
FusionLayer                                  # fusion sub-module
residual gate                                # fusion sub-module
CCFormerHead                                 # task head
Mask2Former-style predictor                  # mask branch
MSDeformAttnTransformerVG                    # box branch
LAGD / gate_decoupler                        # decoupling module
RECSLoss                                     # loss class
loss_ris                                     # RIS loss term
loss_vg                                      # VG loss term
loss_ris2vg / RIS2VGLoss / CCLoss            # cross-task loss (critical to verify)
eval_recs                                    # eval entrypoint
```

---

## 13. Naming conventions

### 13.1 Tool scripts

```text
tools/stage1_env_audit.py
tools/stage1_split_audit.py
tools/stage1_loss_audit.py
tools/stage1_dataloader_smoke.py
tools/stage1_ckpt_sanity.py
tools/stage1_model_audit.py
tools/trace_shapes.py
tools/profile_model.py

tools/mask_to_polygon.py       # Stage 2+
tools/cbcs_sampler.py           # Stage 2+
tools/efd_fit.py                # Stage 2+
tools/reconstruct_mask.py       # Stage 2+
tools/eval_representation.py    # Stage 2+
tools/analyze_shape_buckets.py  # Stage 2+
```

### 13.2 Experiment IDs

```text
S1-E00 .. S1-E13       # Stage 1
S2-E00 ..              # Stage 2
S3-E00 ..              # Stage 3 (V1)
S4-E00 ..              # Stage 4 (V2)
S5-E00 ..              # Stage 5 (V3)
```

### 13.3 Output structure

```text
outputs/
  stage1_env/
  stage1_split/
  stage1_loss/
  stage1_smoke/
  stage1_profile/
  stage1_exit_report.md
  stage2_representation/
  stage3_v1_head/
  stage4_v2_routing/
  stage5_v3_full/
```

### 13.4 Configs

```text
configs/recs.yaml                              # original, do not modify
configs/recs_baseline_ablation.yaml            # baseline on ablation split
configs/recs_baseline_full.yaml                # baseline on full split
configs/recs_baseline_fair_ccloss.yaml         # baseline with patched CCLoss (if needed)
configs/stage1_smoke.yaml                      # single-GPU smoke
configs/stage1_smoke_ddp.yaml                  # DDP smoke
configs/canfor_v1_head_replacement.yaml        # Stage 3
configs/canfor_v2_routing.yaml                 # Stage 4
configs/canfor_v3_full.yaml                    # Stage 5
```

Never mix baseline and new-method results in the same output folder.

---

## 14. Command templates

Adjust flags only after inspecting the repo. All paths and tool names below are templates; resolve them in Stage 1.

```bash
# 0. Session bootstrap
pwd
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git hash-object AGENTS.md
git hash-object PROJECT_PROGRESS.md

# 1. Environment audit (S1-E00)
python tools/stage1_env_audit.py --out outputs/stage1_env/

# 2. Split audit (S1-E02)
python tools/stage1_split_audit.py --config configs/recs.yaml --dry-run
python tools/stage1_split_audit.py --config configs/recs.yaml

# 3. Loss audit (S1-E03)
grep -RnE "class RECSLoss|loss_ris|loss_vg|loss_ris2vg|RIS2VGLoss|CCLoss|collaborative" .
python tools/stage1_loss_audit.py --config configs/recs.yaml --dry-run
python tools/stage1_loss_audit.py --config configs/recs.yaml

# 4. Checkpoint sanity (S1-E04.5)
python tools/stage1_ckpt_sanity.py --config configs/recs.yaml

# 5. Dataloader smoke (S1-E04)
python tools/stage1_dataloader_smoke.py --config configs/recs.yaml --batch-size 1 --num-workers 0

# 6. Forward smoke (S1-E05)
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/stage1_smoke.yaml --dry-run-forward

# 7. Shape trace (S1-E09)
CUDA_VISIBLE_DEVICES=0 python tools/trace_shapes.py --config configs/recs.yaml --batch-size 1 --image-size 512 --text-len 21

# 8. Profile (S1-E10) - run with at least two tools
CUDA_VISIBLE_DEVICES=0 python tools/profile_model.py --config configs/recs.yaml --tool torchinfo    --batch-size 1 --image-size 512 --text-len 21
CUDA_VISIBLE_DEVICES=0 python tools/profile_model.py --config configs/recs.yaml --tool fvcore       --batch-size 1 --image-size 512 --text-len 21
CUDA_VISIBLE_DEVICES=0 python tools/profile_model.py --config configs/recs.yaml --tool torch-profiler --batch-size 1 --image-size 512 --text-len 21

# 9. DDP ladder (S1-E11a/b/c)
CUDA_VISIBLE_DEVICES=0,1       NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py --config configs/stage1_smoke_ddp.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3   NCCL_DEBUG=INFO torchrun --nproc_per_node=4 train.py --config configs/stage1_smoke_ddp.yaml
                               NCCL_DEBUG=INFO torchrun --nproc_per_node=8 train.py --config configs/stage1_smoke_ddp.yaml
```

---

## 15. Stage 1 experiment questions (summary)

| ID       | Question                                                            |
| -------- | ------------------------------------------------------------------- |
| S1-E00   | Is the environment able to run CCFormer?                            |
| S1-E01   | Where are train/eval/model/loss/dataset entrypoints (file:line)?    |
| S1-E02   | Is current config `ablation-subset`, `full-train`, or `unknown`?    |
| S1-E03   | Does active `RECSLoss` really enable `CCLoss` (three-column test)?  |
| S1-E04   | Does dataloader load image/text/box/mask correctly?                 |
| S1-E04.5 | Do pretrained checkpoints load without missing core-module keys?    |
| S1-E05   | Can batch size 1 forward pass run on one V100-16GB?                 |
| S1-E06   | Can loss/backward run on one V100-16GB?                             |
| S1-E07   | Can model overfit one tiny subset for one epoch?                    |
| S1-E08   | Can `eval_recs` run end-to-end?                                     |
| S1-E09   | What are key tensor shapes (full table)?                            |
| S1-E10   | Where do params/FLOPs/memory go (module-level, ≥2 tools)?           |
| S1-E11a  | Does 2-GPU DDP work?                                                |
| S1-E11b  | Does 4-GPU DDP work?                                                |
| S1-E11c  | Does 8-GPU DDP work?                                                |
| S1-E12   | What is the short-run baseline on the active split?                 |
| S1-E13   | Which modules are ready for replacement (V1/V2/V3 map)?             |

Detailed records live in `PROJECT_PROGRESS.md`.

---

## 16. Result logging rules

After every experiment, update `PROJECT_PROGRESS.md` with:

- experiment ID;
- status (planned / running / completed / failed / invalid);
- purpose;
- hypothesis;
- command (exact);
- config path;
- seed;
- batch size;
- GPUs (count and IDs);
- precision (fp32 / amp);
- dataset split (with classification per §7.1);
- changed files;
- diff summary;
- checkpoint/log paths;
- metrics with confidence labels;
- peak memory per GPU;
- wall-clock time;
- params/FLOPs if relevant;
- confidence label (§8);
- failure/caveats;
- decision (continue / repeat / fix / kill / pivot);
- next action.

**No backfill means the experiment is invalid.** Backfill must happen before starting the next experiment.

---

## 17. Kill / pivot reference

See also §11 for Stage 2 numerical thresholds.

| Criterion                                                                          | Kill / Pivot action                                                                 |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| EFD K=32 reconstruction mIoU much worse than CBCS N=80                             | Kill pure EFD; pivot to EFD + residual or canonical polygon                         |
| EFD K=64 still fails on slender / multi-part objects                               | Do not use EFD as sole representation                                               |
| Canonicalization causes orientation ambiguity > 10%                                | Drop PCA/rotated canonical; use axis-aligned or learned orientation                 |
| Box from contour underperforms explicit box branch by > 3 pts P@0.5                | Strengthen transform loss or add temporary auxiliary box supervision                |
| V1 does not reduce decoder FLOPs                                                   | Verify Mask2Former and MSDeformAttn box decoder were actually removed from graph    |
| V2 routing does not improve over unified no-routing head                           | Simplify routing; do not make it the main contribution                              |
| Full split contradicts ablation split                                              | Report full split; use ablation only for debugging                                  |
| Boundary F-score collapses on bridge / harbor / windmill                           | Pivot to canonical contour + residual polygon                                       |
| V3 fusion replacement causes > 5 pts drop that cannot be recovered by simple scale fusion | Keep simple fusion; reframe paper story around representation + routing             |

---

## 18. Open questions (to fill in Stage 1)

These must be resolved in `PROJECT_PROGRESS.md` during Stage 1. They stay `to verify` until concrete evidence is recorded.

- exact dataset root
- exact active config
- exact active train/val/test split files and sample counts
- active model class file:line
- active loss class file:line
- whether `CCLoss` is effective (three-column test)
- exact output tensor shapes through the full pipeline
- exact eval metric names and their implementation file:line
- exact eval script invocation
- exact checkpoint format and loading behavior
- exact module-level params/FLOPs/peak-memory (reconciled ≥2 tools)
- exact full-benchmark protocol (from paper, to cite)
- replacement-ready insertion points for V1/V2/V3
