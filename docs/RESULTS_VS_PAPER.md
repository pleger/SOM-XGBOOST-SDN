# Results Comparison: This Repo vs Paper

This document compares the paper's reported performance with the results reproduced in this repository.

## Paper numbers (SCX)

From the paper text:

- Initial model: Accuracy `99.89%`, F1 `99.91%`
- After continuous iterations (figure discussion): Accuracy up to `99.92%`, F1 up to `99.93%`
- Table II reports SCX Accuracy `99.91%`

Note: the paper contains minor inconsistencies across sections (99.89/99.90 vs 99.91/99.93).

## Our reproduced results (3 seeds mean)

All metrics below are means over seeds `42,43,44`.

| Setup | Model | Accuracy | F1 | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| `10/65/25`, random stratified, no-port | Baseline XGBoost | 99.8673% | 99.8384% | 99.8790% | 99.7979% | n/a |
| `10/65/25`, random stratified, no-port | SCX v2 | 99.8618% | 99.8316% | 99.9279% | 99.7356% | n/a |
| `10/65/25`, per-file temporal, no-port | Baseline XGBoost | 79.8577% | 50.2265% | 99.9627% | 33.5392% | n/a |
| `10/65/25`, per-file temporal, no-port | SCX v2 | 79.8359% | 50.1481% | 99.9533% | 33.4704% | n/a |
| `30/45/25`, random stratified, no-port, tune `recall@FPR<=0.05%` | Baseline XGBoost | 99.9044% | 99.8837% | 99.9307% | 99.8366% | 0.0483% |
| `30/45/25`, random stratified, no-port, tune `recall@FPR<=0.05%` | SCX v2 | 99.9003% | 99.8786% | 99.9312% | 99.8260% | 0.0479% |
| `30/45/25`, random stratified, no-port, tune `recall@FPR<=0.03%` | Baseline XGBoost | 99.9105% | 99.8910% | 99.9616% | 99.8205% | 0.0267% |
| `30/45/25`, random stratified, no-port, tune `recall@FPR<=0.03%` | SCX v2 | 99.9018% | 99.8804% | 99.9575% | 99.8034% | 0.0296% |
| `30/45/25`, random stratified, keep-port, tune `recall@FPR<=0.03%` | Baseline XGBoost | 99.9139% | 99.8952% | 99.9616% | 99.8288% | 0.0267% |
| `30/45/25`, random stratified, keep-port, tune `recall@FPR<=0.03%` | SCX v2 | 99.9113% | 99.8919% | 99.9713% | 99.8126% | 0.0200% |

## How we got these results

1. Dataset
- CICIDS2017 Friday CSV subset (downloaded from CIC dataset portal).
- Binary label mapping: `BENIGN=0`, all attacks=`1`.

2. Features
- Default mode removes endpoint-identifying port field (`Destination Port`) to reduce shortcut learning.
- A sensitivity run with `--keep-port-features` was also included.

3. Splits and seeds
- Main reproducible runs use three seeds: `42,43,44`.
- Ratios shown as `seed/stream/test`.
- Two split strategies were evaluated:
  - `random_stratified` (paper-like evaluation behavior)
  - `per_file_time` (harder temporal generalization)

4. SCX v2 loop
- Train initial XGBoost on seed data.
- Train SOM on seed data.
- Pseudo-label stream samples with confidence + support + quantization-error filtering.
- Retrain XGBoost in increments using accepted pseudo-labels.

5. Thresholding
- For constrained operating points, threshold is calibrated on a held-out stream calibration slice using:
  - `--threshold-tune recall_at_fpr`
  - `--fpr-ceiling 0.0005` (0.05%) or `0.0003` (0.03%)

6. Reproducibility
- All outputs are committed as JSON files in `results/`.
- Main runner: `scripts/scx_v2_experiment.py`

## Interpretation

- We can reach and exceed the paper's `99.91%` accuracy under favorable split/label-budget settings.
- Under stricter temporal conditions, performance is much lower for both methods.
- In this repository's current implementation, SCX v2 is usually close to baseline but does not consistently surpass baseline mean Accuracy/F1.
