# Cross-Domain Pancreas Segmentation: Annotation Selection under Domain Shift

Reproducibility repository for the manuscript *"Does Difficulty Guide Case Selection? A Descriptive Retrospective Study in Cross-Domain Pancreas Segmentation"* (author: Yingjie Hu, Jiangsu University).

This is a **negative-result study**: it directly measures, case by case, whether prediction difficulty (D*) predicts single-case annotation utility (U1), and whether difficulty-guided selection beats random selection at fixed label budgets. The repository provides every script and frozen split needed to reproduce each reported number.

## What is here

```
code/               analysis scripts (training, selection, utility, calibration, reporting)
splits/             frozen patient-level splits (splits.json, splits_oop_pd.json,
                    splits_pairs_mcf.json, splits_pairs_mcfiu.json) — the single source of truth
outputs_upstream/   derived case-level tables documenting the reported numbers
results_seeded_v2.csv   main-matrix budget results (seeded_v2 protocol)
environment.yml     dependencies
复现流程.md          step-by-step run instructions (Chinese)
```

## Reproduce each number

| Reported figure | File / command |
|---|---|
| Budget curves (Table 5), MMD | `results_seeded_v2.csv` + `code/report.py` |
| 24-draw percentiles (§4.2) | `outputs_upstream/selection_repeat.json`, `selection_repeatability.csv` |
| Summed vs realised gains (§4.5, Table 9) | `outputs_upstream/utility_interaction.csv` |
| Single-case utility U1, 305 cases (§4.4) | `outputs_upstream/individual_utility{,_ct,_nyu}.csv`, `utility_multiseed_summary.json` |
| 11-pair domain grid, ρ = −0.78 (§4.7, Table 7) | `outputs_upstream/pair_grid_all.csv`, `pair_grid.csv` |
| Conformal coverage (§4.7) | `outputs_upstream/calibration_report.json` |
| Split generator / seed protocol | `code/protocol.py` (`SEED_PROTOCOL`, `select_nested_keys`), `code/prepare.py` (`build_splits`) |
| Selection & fine-tuning runs | `code/select_eval.py`, `code/train_eval.py` |

## Path placeholders

Machine-specific absolute paths have been replaced by placeholders for privacy: `<DATA_ROOT>` (source datasets: CrossPan, MSD Task07, AMOS22), `<WORKSPACE>` (this project directory), and `<HOME>`. Before running, either edit these in `code/config.py` (and the two files that declare `AMOS_ROOT`), or set the corresponding environment variables (`EXP_CT_ROOT`, `EXP_WORK`, `EXP_CACHE`, `EXP_RUNS`) which `config.py` already reads via `os.environ.get`.

## Determinism and limits

- `SEED_PROTOCOL = "seeded_v2"` fixes Python/NumPy/PyTorch CPU+CUDA RNGs and disables cuDNN benchmarking, so re-running with the same `splits/` and seeds reproduces the reported numbers up to minor GPU/driver-dependent differences.
- Reproducibility is of the **numbers**, not bit-exactness across machines; multi-seed variation (not bit reproduction) is the reporting unit (see the manuscript, Section 3.8).
- The nnU-Net (A6000) arm's 27 runs and the MedSAM arm are documented via their run metadata; their checkpoints are dataset-derived and are available from the author on request rather than redistributed.

## Data

Source images/masks are **not** redistributed. They come from: CrossPan (https://crosspan.netlify.app/, CC BY-NC) and MSD Task07 (https://registry.opendata.aws/msd/, CC BY-SA). Preprocessed caches and model weights are derived artifacts and are likewise not included; obtainable on request.

## Ethics / AI use

Public de-identified data, retrospective secondary analysis; no IRB required (per institutional guidance). Generative-AI tools (OpenAI GPT and DeepSeek) were used to draft code and edit text; data and analyses are the author's own.