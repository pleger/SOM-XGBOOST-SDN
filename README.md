# SOM-XGBOOST-SDN

Reproducible experiments for a SOM-based continuous XGBoost pipeline for DDoS detection on CICIDS2017 Friday traffic.

Main script:

- `scripts/scx_v2_experiment.py`

Detailed usage:

- `README_SCX_V2.md`

Results comparison with the paper (including summary table):

- `docs/RESULTS_VS_PAPER.md`

## Quick Comparison (Paper vs This Repo)

3-seed mean results (`42,43,44`):

| Scenario | Model | Accuracy | F1 |
|---|---:|---:|---:|
| Paper (reported best SCX) | SCX | 99.92% | 99.93% |
| This repo, `10/65/25`, random stratified, no-port | Baseline XGBoost | 99.8673% | 99.8384% |
| This repo, `10/65/25`, random stratified, no-port | SCX v2 | 99.8618% | 99.8316% |
| This repo, `30/45/25`, random stratified, no-port, `recall@FPR<=0.03%` | Baseline XGBoost | 99.9105% | 99.8910% |
| This repo, `30/45/25`, random stratified, no-port, `recall@FPR<=0.03%` | SCX v2 | 99.9018% | 99.8804% |
| This repo, `30/45/25`, random stratified, keep-port, `recall@FPR<=0.03%` | Baseline XGBoost | 99.9139% | 99.8952% |
| This repo, `30/45/25`, random stratified, keep-port, `recall@FPR<=0.03%` | SCX v2 | 99.9113% | 99.8919% |

Key takeaway: this reproduction can reach and exceed 99.91% accuracy under favorable split/label-budget settings, while temporal split performance is much lower for both methods.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On macOS, if `xgboost` complains about OpenMP:

```bash
brew install libomp
```

## Outputs

Experiment outputs are written as JSON files under `results/`, including:

- per-seed metrics
- aggregated means/std
- threshold values used after calibration
