# PROJECT_PROGRESS.md

> Mutable state for the RRSECS / CanFoR-RRSECS project.
> Immutable contract lives in `AGENTS.md`.
> Update discipline: see `AGENTS.md §6.5`.
> Session bootstrap required before any experiment: see `AGENTS.md §2`.

---

## Current one-line status

```text
Stage 0: complete (AGENTS.md and PROJECT_PROGRESS.md established).
Stage 1: not started.
Current focus: establish trustworthy CCFormer baseline; code audit; split audit; loss audit; tensor shapes; profiling.
```

---

## Session log (append-only)

Every Codex session must append a row here **before** any other action.
Never delete or rewrite rows in this table.

| Timestamp (UTC) | AGENTS.md hash | PROJECT_PROGRESS.md hash | Git commit | Planned experiments | One-sentence plan |
| --------------- | -------------- | ------------------------ | ---------- | ------------------- | ----------------- |
| _none yet_      | _fill in_      | _fill in_                | _fill in_  | _fill in_           | _fill in_         |

---

## Project snapshot (updatable)

| Item                      | Value                           |
| ------------------------- | ------------------------------- |
| Date                      | TBD                             |
| Repo root                 | TBD                             |
| Branch                    | TBD                             |
| Commit hash               | TBD                             |
| Dataset path              | TBD                             |
| Current stage             | Stage 0 complete, Stage 1 pending |
| Current objective         | Stage 1 code audit + baseline profile |
| Hardware                  | 8 × NVIDIA V100, 16 GB each     |
| Node shared or exclusive? | TBD                             |
| Active config             | TBD                             |
| Active split              | TBD (to classify per AGENTS.md §7.1) |
| Active loss               | TBD (to verify per AGENTS.md §7.2)   |
| Current best result       | none                            |
| Confidence of best result | N/A                             |
| Last updated by           | initial commit                  |

---

## Current best known facts (updatable)

All claims below are user-provided context. They must be verified in Stage 1 against code.

| Topic                | Fact                                                                                                          | Confidence | Evidence                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------- |
| Training chain       | configs/recs.yaml → train.py → create_dataset → recs_model_builder → Swin → CCFormer_fusion → CCFormerHead → RECSLoss → eval_recs | to verify  | user-provided; verify in S1-E01   |
| Visual encoder       | Swin-Tiny                                                                                                     | to verify  | verify in S1-E01                  |
| Text encoder         | BERT-base-uncased                                                                                             | to verify  | verify in S1-E01                  |
| Fusion               | CCFormer_fusion = MCM + FusionLayer + residual gate                                                           | to verify  | verify in S1-E01                  |
| Segmentation branch  | Mask2Former-style predictor                                                                                   | to verify  | verify in S1-E01                  |
| Grounding branch     | MSDeformAttnTransformerVG                                                                                     | to verify  | verify in S1-E01                  |
| Output               | `{'mask': pred_masks, 'bbox': pred_bboxs}`                                                                    | to verify  | verify in S1-E01                  |
| Active loss risk     | `RECSLoss` may only include `loss_ris + loss_vg`                                                              | to verify  | verify in S1-E03 (three-column)   |
| Split risk           | May use ablation split (`abla_splits/10/labeled.json`, `val_ablation.json`) instead of full benchmark          | to verify  | verify in S1-E02                  |
| Paper-reported FLOPs | CCFormer ≈ 119 G; SeeFormer ≈ 68 G; RECS4R ≈ 45 G                                                             | paper-reported | cite source when quoted          |

---

## Current blockers (updatable)

| Blocker                             | Severity | Owner       | Next step                                                                    |
| ----------------------------------- | -------- | ----------- | ---------------------------------------------------------------------------- |
| Dataset root path unknown           | high     | user        | Provide absolute path; fill in Project snapshot                              |
| Node shared/exclusive status        | medium   | user        | Confirm; if shared, add `nvidia-smi` check to every training command         |
| Active split unknown                | high     | Codex       | Run S1-E02 and classify per AGENTS.md §7.1                                   |
| Active loss effectiveness unknown   | high     | Codex       | Run S1-E03 with three-column evidence                                        |
| Checkpoint availability unknown     | high     | user/Codex  | User to confirm Swin/BERT/RECS pretrained paths; Codex to run S1-E04.5       |
| Profile not measured                | high     | Codex       | Run S1-E10 with ≥2 tools                                                     |
| Baseline not reproduced             | high     | Codex       | Complete S1-E00 through S1-E12                                               |

---

## Next immediate action

```text
1. Read AGENTS.md in full.
2. Execute session bootstrap protocol (AGENTS.md §2):
   - record hashes, commit, plan into Session log below.
3. Run Stage 1.1 to Stage 1.4 only:
   - S1-E00 Environment audit
   - S1-E01 Repo entrypoint audit
   - S1-E02 Split audit
   - S1-E03 Active loss audit (three-column)
4. Do NOT modify baseline model internals.
5. Backfill all four experiments into the Detailed experiment records below
   before proceeding to S1-E04.
```

---

## Stage checklist (updatable; checkbox toggles only, do not delete items)

### Stage 0 — Documentation and repo safety setup

- [x] AGENTS.md created
- [x] PROJECT_PROGRESS.md created
- [x] Experiment folder convention established (AGENTS.md §13.3)
- [x] Naming convention established (AGENTS.md §13)
- [x] Baseline / new-method separation policy established (AGENTS.md §6)
- [x] Logging template established (AGENTS.md §16)
- [x] Tool script contract established (AGENTS.md §6.4)
- [x] Update discipline established (AGENTS.md §6.5)

### Stage 1 — CCFormer baseline reproduction, code audit, profiling

- [ ] S1-E00 Environment audit
- [ ] S1-E01 Repo entrypoint audit
- [ ] S1-E02 Split audit
- [ ] S1-E03 Active loss audit (three-column)
- [ ] S1-E04 Single-batch dataloader test
- [ ] S1-E04.5 Checkpoint loading sanity
- [ ] S1-E05 Single-GPU forward smoke
- [ ] S1-E06 Single-GPU backward smoke
- [ ] S1-E07 Tiny-subset overfit
- [ ] S1-E08 Eval pipeline smoke
- [ ] S1-E09 Tensor shape tracing
- [ ] S1-E10 Module-level profiling (≥2 tools, FLOPs reconciliation)
- [ ] S1-E11a 2-GPU DDP sanity
- [ ] S1-E11b 4-GPU DDP sanity
- [ ] S1-E11c 8-GPU DDP sanity
- [ ] S1-E12 Baseline reproduction short run
- [ ] S1-E13 Replacement readiness audit
- [ ] outputs/stage1_exit_report.md produced
- [ ] Stage 1 exit criteria satisfied (AGENTS.md §10.4)

### Stage 2 — Representation-only feasibility study

- [ ] GT mask / box loader prepared
- [ ] Polygon conversion implemented (`tools/mask_to_polygon.py`)
- [ ] CBCS sampler implemented (`tools/cbcs_sampler.py`)
- [ ] EFD fitting implemented (`tools/efd_fit.py`)
- [ ] Canonicalization strategies implemented (axis-aligned / rotated / PCA / learned)
- [ ] Reconstruction metrics implemented (`tools/eval_representation.py`)
- [ ] Bucket analysis complete (`tools/analyze_shape_buckets.py`)
- [ ] Go / kill decision made per AGENTS.md §11.3

### Stage 3 — V1 minimal head replacement

- [ ] V1 canonical Fourier-contour head implemented
- [ ] Mask2Former branch bypassed or replaced
- [ ] Box decoder bypassed or replaced
- [ ] Box / mask derived from contour (`Rect(C_img)` and `Rasterize/Fill(C_img)`)
- [ ] 5% subset training complete
- [ ] 10% subset training complete
- [ ] V1 profile complete (params / FLOPs / memory / latency)

### Stage 4 — V2 task-specific routing replacement

- [ ] LAGD / gate_decoupler bypassed
- [ ] Shared canonical representation implemented
- [ ] Localization router implemented
- [ ] Segmentation router implemented
- [ ] Unified vs routing ablation complete

### Stage 5 — V3 full method (paper version)

- [ ] CCFormer_fusion replaced (LCSF)
- [ ] LGDM / LAGD replaced (SGOR)
- [ ] CCLoss replaced (BRC)
- [ ] Full-split training complete
- [ ] Main ablation matrix complete
- [ ] Failure analysis complete
- [ ] Paper-ready result table complete

---

## Experiment overview table (append-only)

Never rewrite existing rows. Add a new row for every attempt, including failures.

| ID       | Name                                | Status  | Split | Key result | Confidence | Last updated |
| -------- | ----------------------------------- | ------- | ----- | ---------- | ---------- | ------------ |
| S1-E00   | Environment audit                   | planned | N/A   | TBD        | to verify  | TBD          |
| S1-E01   | Repo entrypoint audit               | planned | N/A   | TBD        | to verify  | TBD          |
| S1-E02   | Split audit                         | planned | TBD   | TBD        | to verify  | TBD          |
| S1-E03   | Active loss audit (three-column)    | planned | N/A   | TBD        | to verify  | TBD          |
| S1-E04   | Single-batch dataloader test        | planned | TBD   | TBD        | to verify  | TBD          |
| S1-E04.5 | Checkpoint loading sanity           | planned | N/A   | TBD        | to verify  | TBD          |
| S1-E05   | Single-GPU forward smoke            | planned | TBD   | TBD        | to verify  | TBD          |
| S1-E06   | Single-GPU backward smoke           | planned | TBD   | TBD        | to verify  | TBD          |
| S1-E07   | Tiny-subset overfit                 | planned | tiny  | TBD        | to verify  | TBD          |
| S1-E08   | Eval pipeline smoke                 | planned | TBD   | TBD        | to verify  | TBD          |
| S1-E09   | Tensor shape tracing                | planned | N/A   | TBD        | to verify  | TBD          |
| S1-E10   | Module-level profiling              | planned | N/A   | TBD        | to verify  | TBD          |
| S1-E11a  | 2-GPU DDP sanity                    | planned | tiny  | TBD        | to verify  | TBD          |
| S1-E11b  | 4-GPU DDP sanity                    | planned | tiny  | TBD        | to verify  | TBD          |
| S1-E11c  | 8-GPU DDP sanity                    | planned | tiny  | TBD        | to verify  | TBD          |
| S1-E12   | Baseline reproduction short run     | planned | TBD   | TBD        | to verify  | TBD          |
| S1-E13   | Replacement readiness audit         | planned | N/A   | TBD        | to verify  | TBD          |

---

## Detailed experiment records (append-only)

Use the template below for every experiment. Append, do not rewrite.
Any row with confidence `failed / invalid` is **never-delete** (AGENTS.md §6.5).

### Template

```markdown
## [S?-E??] <name>

**Status:** planned / running / completed / failed / invalid
**Purpose:** <one sentence>
**Hypothesis:** <expected outcome>

**Command:**
```bash
<exact command>
```

**Config:**
| Item | Value |
|---|---|
| Config file | TBD |
| Key overrides | TBD |
| Seed | TBD |
| Batch size | TBD |
| GPUs (count and IDs) | TBD |
| Precision | fp32 / amp |
| Dataset split | TBD (classification: ablation-subset / full-train / unknown) |

**Code changes:**
| Item | Value |
|---|---|
| Changed files | TBD |
| Diff summary | TBD |
| Reversible? | yes / no / N/A |

**Outputs:**
| Item | Path |
|---|---|
| Checkpoint path | TBD |
| Log path | TBD |
| TensorBoard / W&B path | TBD |
| Prediction visualization path | TBD |

**Metrics:**
| Metric | Value | Split | Confidence | Notes |
|---|---:|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

**Resource usage:**
| Item | Value |
|---|---:|
| GPU count | TBD |
| Peak memory / GPU | TBD |
| Wall time | TBD |
| Throughput | TBD |
| Params | TBD |
| FLOPs/MACs (tool A) | TBD |
| FLOPs/MACs (tool B) | TBD |
| FLOPs reconciliation | within 15% / > 15% / manual fallback |

**Key observations:**
- TBD

**Failure / caveats:**
- TBD

**Decision:**
```text
continue / repeat / fix / kill / pivot
```

**Next action:**
- TBD
```

---

_(No experiment records yet. Append below this line.)_

---

## Code audit table (append-only rows; updatable cells within a row via str_replace)

| Component           | File:line | Class / Function | Role                | Input  | Output | Keep / Replace  | Stage to replace | Notes |
| ------------------- | --------- | ---------------- | ------------------- | ------ | ------ | --------------- | ---------------- | ----- |
| Config              | TBD       | TBD              | active config       | TBD    | TBD    | keep            | N/A              | TBD   |
| Train entry         | TBD       | TBD              | training entrypoint | TBD    | TBD    | keep            | N/A              | TBD   |
| Dataset builder     | TBD       | TBD              | create dataset      | TBD    | TBD    | keep            | N/A              | TBD   |
| Model builder       | TBD       | TBD              | build RECS model    | TBD    | TBD    | keep initially  | partial in V3    | TBD   |
| Visual encoder      | TBD       | TBD              | Swin backbone       | TBD    | TBD    | keep initially  | optional V3      | TBD   |
| Text encoder        | TBD       | TBD              | BERT text features  | TBD    | TBD    | keep            | N/A              | TBD   |
| Fusion              | TBD       | TBD              | CCFormer_fusion     | TBD    | TBD    | replace         | V3 → LCSF        | TBD   |
| LAGD / gate         | TBD       | TBD              | decoupling          | TBD    | TBD    | replace         | V2 → SGOR        | TBD   |
| Segmentation head   | TBD       | TBD              | mask prediction     | TBD    | TBD    | replace         | V1 → CFCR+RCR    | TBD   |
| Grounding head      | TBD       | TBD              | box prediction      | TBD    | TBD    | replace         | V1 → Rect(C_img) | TBD   |
| Loss                | TBD       | TBD              | RECS loss           | TBD    | TBD    | replace         | V3 → BRC         | TBD   |
| Eval                | TBD       | TBD              | eval metrics        | TBD    | TBD    | keep            | N/A              | TBD   |

---

## Tensor shape trace (updatable; all rows TBD until S1-E09)

| Module              | Input shape | Output shape | Batch | Image size | Text length | Evidence (file:line) | Notes |
| ------------------- | ----------- | ------------ | ----: | ---------- | ----------: | -------------------- | ----- |
| Image input         | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| BERT text features  | TBD         | TBD          |     1 | N/A        |          21 | TBD                  | TBD   |
| Swin stage 1        | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| Swin stage 2        | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| Swin stage 3        | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| Swin stage 4        | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| CCFormer_fusion     | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| LAGD / gate         | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| Mask decoder        | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| Box decoder         | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| Loss inputs         | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |
| Eval outputs        | TBD         | TBD          |     1 | 512×512    |          21 | TBD                  | TBD   |

---

## Loss audit (three-column rule — AGENTS.md §7.2)

Three columns must all be `yes` for `Effective?` to be `yes`. Grep alone is insufficient.

| Loss term                          | Defined in class? (file:line) | Called in forward? (file:line) | Weight > 0 in active config? (file:key=value) | Effective? | Notes |
| ---------------------------------- | ----------------------------- | ------------------------------ | --------------------------------------------- | ---------- | ----- |
| `loss_ris`                         | TBD                           | TBD                            | TBD                                           | TBD        | TBD   |
| `loss_vg`                          | TBD                           | TBD                            | TBD                                           | TBD        | TBD   |
| `loss_ris2vg`                      | TBD                           | TBD                            | TBD                                           | TBD        | TBD   |
| `RIS2VGLoss`                       | TBD                           | TBD                            | TBD                                           | TBD        | TBD   |
| `CCLoss` / cross-task collaborative | TBD                           | TBD                            | TBD                                           | TBD        | **Critical** — determines fair-baseline variant |
| Dice / BCE / Focal mask loss       | TBD                           | TBD                            | TBD                                           | TBD        | TBD   |
| L1 / GIoU box loss                 | TBD                           | TBD                            | TBD                                           | TBD        | TBD   |

---

## Profiling results (updatable; requires ≥2 tools per AGENTS.md §8.1)

| Module            | Params | Trainable params | FLOPs (tool A) | FLOPs (tool B) | Reconciliation | Peak memory | Latency | Confidence |
| ----------------- | -----: | ---------------: | -------------: | -------------: | -------------- | ----------: | ------: | ---------- |
| Full model        | TBD    | TBD              | TBD            | TBD            | TBD            | TBD         | TBD     | to verify  |
| Visual encoder    | TBD    | TBD              | TBD            | TBD            | TBD            | TBD         | TBD     | to verify  |
| Text encoder      | TBD    | TBD              | TBD            | TBD            | TBD            | TBD         | TBD     | to verify  |
| CCFormer_fusion   | TBD    | TBD              | TBD            | TBD            | TBD            | TBD         | TBD     | to verify  |
| LAGD / gate       | TBD    | TBD              | TBD            | TBD            | TBD            | TBD         | TBD     | to verify  |
| Mask branch       | TBD    | TBD              | TBD            | TBD            | TBD            | TBD         | TBD     | to verify  |
| Box branch        | TBD    | TBD              | TBD            | TBD            | TBD            | TBD         | TBD     | to verify  |
| Loss / eval overhead | TBD | TBD              | TBD            | TBD            | TBD            | TBD         | TBD     | to verify  |

Tool A, Tool B: chosen from {torchinfo, fvcore, thop, ptflops, torch.profiler}.
Reconciliation: `within 15%` / `> 15%, fallback used` / `manual estimate`.

---

## Dataset and split audit

Classification follows AGENTS.md §7.1 strictly.

| Split              | File path | Sample count | Image count | Category count | Has box? | Has mask? | Classification (§7.1) | Matches paper protocol? | Notes |
| ------------------ | --------- | -----------: | ----------: | -------------: | -------- | --------- | --------------------- | ----------------------- | ----- |
| train              | TBD       | TBD          | TBD         | TBD            | TBD      | TBD       | TBD                   | TBD                     | TBD   |
| val                | TBD       | TBD          | TBD         | TBD            | TBD      | TBD       | TBD                   | TBD                     | TBD   |
| test               | TBD       | TBD          | TBD         | TBD            | TBD      | TBD       | TBD                   | TBD                     | TBD   |
| ablation train (if any) | TBD  | TBD          | TBD         | TBD            | TBD      | TBD       | ablation-subset       | no                      | TBD   |
| ablation val (if any)   | TBD  | TBD          | TBD         | TBD            | TBD      | TBD       | ablation-subset       | no                      | TBD   |

---

## Ablation-subset sanity runs (append-only)

Any training on an ablation subset is logged here, **never** in the full-benchmark baseline table.

| Run ID | Config | Subset name | Epochs | Batch/GPU | GPUs | AMP | Metrics | Checkpoint | Confidence | Notes |
| ------ | ------ | ----------- | -----: | --------: | ---: | --- | ------- | ---------- | ---------- | ----- |
| TBD    | TBD    | TBD         | TBD    | TBD       | TBD  | TBD | TBD     | TBD        | sanity-only | TBD   |

---

## Baseline reproduction results (append-only, full-benchmark only)

This table contains **only** full-split results. Ablation-subset runs go in the table above.
Do not mix the two under any circumstances.

| Run ID | Config | Variant (as-found / fair-ccloss) | Split | Epochs | Batch/GPU | GPUs | AMP | mIoU | oIoU | P@0.5 | Checkpoint | Confidence | Notes |
| ------ | ------ | -------------------------------- | ----- | -----: | --------: | ---: | --- | ---: | ---: | ----: | ---------- | ---------- | ----- |
| TBD    | TBD    | TBD                              | TBD   | TBD    | TBD       | TBD  | TBD | TBD  | TBD  | TBD   | TBD        | to verify  | TBD   |

---

## Representation feasibility results (Stage 2, filled later)

Thresholds: AGENTS.md §11.3.

| Representation          | Sweep                                                 | Scalar count  | mIoU | oIoU | Boundary IoU | Boundary F | Chamfer | Hausdorff95 | Failure rate | Confidence |
| ----------------------- | ----------------------------------------------------- | ------------: | ---: | ---: | -----------: | ---------: | ------: | ----------: | -----------: | ---------- |
| dense mask oracle       | GT                                                    | 512×512       | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| box fill (HBB)          | axis-aligned                                          | 4             | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| rotated box (OBB)       | minAreaRect                                           | 5             | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| raw polygon             | N=20/40/80/120                                        | 2N            | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| CBCS polygon            | N=20/40/80/120                                        | 2N            | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| EFD                     | K=4/8/16/32/64                                        | 4K            | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| canonical contour       | bbox / rotated bbox / PCA / learned                   | 2N + transform | TBD | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| canonical EFD           | K=4/8/16/32/64                                        | 4K + transform | TBD | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| EFD + residual contour  | K ∈ {16, 32}, M ∈ {16, 32, 64}                        | 4K + 2M       | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| low-res SDF 32²         | occupancy                                             | 1024          | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |
| low-res SDF 64²         | occupancy                                             | 4096          | TBD  | TBD  | TBD          | TBD        | TBD     | TBD         | TBD          | TBD        |

---

## Replacement map

| Current CCFormer component           | Proposed replacement                              | Stage | Dependency                              | Risk                          | First test                              |
| ------------------------------------ | ------------------------------------------------- | ----- | --------------------------------------- | ----------------------------- | --------------------------------------- |
| Mask2Former-style mask predictor     | CFCR + RCR canonical Fourier-contour head         | V1 / Stage 3 | shape trace, mask-to-EFD converter      | mask fidelity drop            | 5% subset training                      |
| MSDeformAttnTransformerVG            | transform head + `Rect(C_img)`                    | V1 / Stage 3 | contour-to-box function                 | box P@0.5 drop                | box-from-GT-contour sanity              |
| LAGD / gate_decoupler                | SGOR: semantic-geometric routers                  | V2 / Stage 4 | V1 works                                | optimization instability      | unified vs routing ablation             |
| CCFormer_fusion / MCFM               | LCSF: language-conditioned scale fusion           | V3 / Stage 5 | V2 works                                | performance drop              | replace one scale first                 |
| RECSLoss / CCLoss                    | BRC: boundary-reconstruction consistency losses   | V3 / Stage 5 | Stage 2 metrics                         | loss instability              | staged loss schedule (A → B → C)        |

---

## Deviations (append-only; never-delete)

Every time Codex deviates from the planned protocol (e.g. AMP enabled to avoid OOM, image size reduced, `num_workers` changed to fit memory, a new package installed), append a row.

| Timestamp (UTC) | Experiment | Deviation | Reason | Effect on validity | Approved by |
| --------------- | ---------- | --------- | ------ | ------------------ | ----------- |
| _none yet_      | _fill in_  | _fill in_ | _fill in_ | _fill in_       | user / Codex |

---

## Installed packages (append-only; never-delete)

Every `pip install` must be logged here before or immediately after installation.

| Timestamp (UTC) | Package | Version | Reason | Installed by | Notes |
| --------------- | ------- | ------- | ------ | ------------ | ----- |
| _none yet_      | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

---

## Decisions made (append-only)

| Date | Decision | Evidence | Confidence | Revisit condition |
| ---- | -------- | -------- | ---------- | ----------------- |
| 2026-04-24 | Main route is shared canonical representation + task-specific routing (CanFoR-RRSECS) | user-provided research design | high | Stage 2 reconstruction study fails per AGENTS.md §11.3 |
| 2026-04-24 | Stage 1 does not modify model internals | engineering safety (AGENTS.md §6.2)      | high | Stage 1 complete |
| 2026-04-24 | Ablation split and full benchmark must be separated in all reporting | known protocol risk (AGENTS.md §7.1)     | high | never |
| 2026-04-24 | FLOPs must be reported with ≥2 tools and reconciled within 15% | known tool variance (AGENTS.md §8.1)     | high | never |
| 2026-04-24 | Loss effectiveness requires three-column evidence | known grep-hallucination risk (AGENTS.md §7.2) | high | never |

---

## Kill / pivot criteria (reference summary)

Canonical list: AGENTS.md §17. Stage 2 numerical thresholds: AGENTS.md §11.3.

| Criterion                                                                          | Kill / pivot action                                                                 |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| EFD K=32 reconstruction mIoU much worse than CBCS N=80 (≥ 5 pts)                   | Kill pure EFD; pivot to EFD + residual or canonical polygon                         |
| EFD K=64 still fails on slender / multi-part objects                               | Do not use EFD as sole representation                                               |
| Canonicalization causes orientation ambiguity rate > 10%                           | Drop PCA / rotated canonical; use axis-aligned or learned orientation               |
| Box from contour underperforms explicit box branch by > 3 pts P@0.5                | Strengthen transform loss or add temporary auxiliary box supervision                |
| V1 does not reduce decoder FLOPs                                                   | Verify Mask2Former and MSDeformAttn box decoder were actually removed from graph    |
| V2 routing does not improve over unified no-routing                                | Simplify routing; do not make it the main contribution                              |
| Full split contradicts ablation split                                              | Report full split only; use ablation only for debugging                             |

---

## Questions for GPT / human discussion (updatable)

To fill in as Stage 1 progresses:

- Is active `RECSLoss` truly missing `CCLoss`? Evidence: TBD.
- Which module dominates FLOPs? Evidence: TBD.
- Is the replacement map feasible without breaking eval? Evidence: TBD.
- Should the fair baseline include a patched `CCLoss`? Decision: TBD after S1-E03.
- Which split is the first paper table going to use? Decision: TBD after S1-E02 + S1-E12.
- Does Mask2Former pixel decoder OOM at 512×512 + batch 1 on V100-16GB? Answer: TBD after S1-E06.
