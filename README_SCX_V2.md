# SCX v2 Experiments (CICIDS2017 Friday)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If `xgboost` fails on macOS with `libomp` missing:

```bash
brew install libomp
```

## Main runner

Script:

`scripts/scx_v2_experiment.py`

It runs:

- `baseline`: XGBoost trained only on initial labeled seed
- `scx_v2`: SOM pseudo-labeling + incremental XGBoost retraining

Outputs JSON with per-seed runs and aggregated summary.
It now also reports additional indicators: `balanced_accuracy`, `specificity`, `fpr`, `fnr`, `mcc`, `pr_auc`, and `roc_auc`.

## Example runs

Random stratified split (paper-style comparison):

```bash
PYTHONUNBUFFERED=1 .venv/bin/python scripts/scx_v2_experiment.py \
  --mode both \
  --seeds 42,43,44 \
  --split-strategy random_stratified \
  --threshold-tune accuracy \
  --calibration-ratio 0.10 \
  --attack-conf-threshold 0.05 \
  --benign-conf-threshold 0.30 \
  --output results/scx_v2_random_3seeds.json
```

Per-file temporal split (deployment-oriented):

```bash
PYTHONUNBUFFERED=1 .venv/bin/python scripts/scx_v2_experiment.py \
  --mode both \
  --seeds 42,43,44 \
  --split-strategy per_file_time \
  --threshold-tune accuracy \
  --calibration-ratio 0.10 \
  --attack-conf-threshold 0.05 \
  --benign-conf-threshold 0.30 \
  --output results/scx_v2_per_file_time_3seeds.json
```

Higher-labeled-data benchmark configuration (crosses 99.91% accuracy in this workspace):

```bash
PYTHONUNBUFFERED=1 .venv/bin/python scripts/scx_v2_experiment.py \
  --mode both \
  --seeds 42,43,44 \
  --split-strategy random_stratified \
  --seed-ratio 0.30 \
  --stream-ratio 0.45 \
  --test-ratio 0.25 \
  --n-estimators 900 \
  --max-depth 8 \
  --learning-rate 0.03 \
  --threshold-tune accuracy \
  --calibration-ratio 0.10 \
  --threshold-grid-size 299 \
  --attack-conf-threshold 0.05 \
  --benign-conf-threshold 0.30 \
  --output results/scx_v2_random_3seeds_seed30_noport_opt.json
```

## Notes

- Default split is `per_file_time`.
- The script drops `Destination Port` to reduce endpoint leakage.
- Default ratios are `10% seed`, `65% stream`, `25% test`.
- Use `--keep-port-features` only for sensitivity analysis; it can overstate generalization.
