# SOM-XGBOOST-SDN

Reproducible experiments for a SOM-based continuous XGBoost pipeline for DDoS detection on CICIDS2017 Friday traffic.

Main script:

- `scripts/scx_v2_experiment.py`

Detailed usage:

- `README_SCX_V2.md`

Results comparison with the paper (including summary table):

- `docs/RESULTS_VS_PAPER.md`

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
