# Dataset, licensing and generated artefacts

## Source data — ETDataset (not redistributed here)

| | |
|---|---|
| Dataset | **ETDataset** (Electricity Transformer Dataset), files `ETTm1.csv` and `ETTm2.csv` |
| Source | <https://github.com/zhouhaoyi/ETDataset> |
| Content | Two power transformers, 15-minute readings, 2016-07 → 2018-06: HV/MV/LV active and reactive load (HUFL, HULL, MUFL, MULL, LUFL, LULL) and oil temperature (OT) |
| License | **Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)**, per the `LICENSE` file in the ETDataset repository (checked 2026-10-08) |
| Required citation | Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021 ([arXiv:2012.07436](https://arxiv.org/abs/2012.07436)), as requested in the ETDataset README |

**What this repository does with it.** `make data` downloads the two CSV files from the ETDataset repository
into `data/raw/` on your machine. It then builds the simulated-failure benchmark locally and replays the held-out
period through the deployed engine. None of these row-level files are committed:

| Not committed (git-ignored, regenerated locally) | Why |
|---|---|
| `data/raw/*.csv` | original ETDataset files |
| `data/processed/*.parquet` | ETDataset values with simulated failure signatures injected (adapted material) |
| `reports/phase2/engine_log/` | per-step engine output including the (adapted) sensor values; feeds the UI timeline |
| `reports/stream/*.csv` | Phase-1 streaming logs including the (adapted) sensor values |
| `models/**/*.pt`, `models/*.onnx`, `models/preprocessor.pkl` | trained weights and fitted preprocessing (scaler means/stds, residual fits), learned from ETDataset |

CC BY-ND 4.0 permits sharing the original material with attribution but **not sharing adapted material**. The
simulated benchmark adds failure signatures to the real readings, so files holding those modified readings are
kept out of the repository and recreated deterministically by `make data`. This is a conservative reading; it
is not legal advice.

## What is committed

| Artefact | Content | Notes |
|---|---|---|
| Source code, configs, tests | original work | MIT License (`LICENSE`) |
| `data/processed/events.csv`, `dataset_card.json`, `data/splits/split_bounds.json` | metadata of the **simulated** events (type, timing, parameters) and split indices | no sensor values |
| `reports/**/*.json`, `*.csv`, `*.md` | metrics, aggregate statistics, evaluation tables | no row-level sensor values (checked by scanning every committed CSV/Parquet for sensor columns) |
| `reports/figures/*.png`, `docs/screenshots/*.png` | plots and UI screenshots that show sensor curves | visualisations, not the data; attribution to ETDataset given here and in the README |
| `models/model_metadata.json`, `models/phase2/deployment.json`, `models/phase2/calibrator_*.json` | model configuration, operating thresholds, isotonic calibration mapping (raw score → probability) | small JSON; no sensor values; needed so retrained models use the published operating point |

## Trained weights are not published

Whether trained model weights count as "adapted material" of a CC BY-ND dataset is **unresolved**: the licence
text does not address it and there is no settled interpretation. The conservative choice is made here: weights,
ONNX exports and the fitted preprocessor are **not** in the repository.

`make data` therefore trains the deployed models locally when no weights are present (`make models`):
- It retrains the GRU + attention model, its 3 seed-ensemble members and the preprocessing from the committed
  configuration (`experiments/best_config.json`, no hyper-parameter search), then exports ONNX.
- It takes about 20–30 minutes on a laptop CPU.
- On the development machine (Apple M2, PyTorch 2.14.1, CPU) the retrained weights were **bit-for-bit identical**
  (same SHA-256) to the ones used for every reported result, and the retrained preprocessing produced identical
  features.
- On other hardware or library builds, floating-point results may differ slightly. The app still runs, but metrics
  and the integrity tests that re-derive the published threshold may then deviate within numerical noise.

The Phase-2/final research checkpoints (`models/phase2/{architecture,features,horizon,cross_asset}/`) are only
needed to re-run those studies (`src/phase2/experiments.py`), not to run the app.

## Figures and screenshots

`reports/figures/*.png` and `docs/screenshots/*.png` include charts of sensor curves, some with simulated faults
injected. They are visualisations rather than the dataset itself, but whether a chart of licensed data is
"adapted material" is also not settled. They are kept, with attribution, because they document the results. If
you prefer the most conservative reading, remove the time-series figures (`eda_sensor_timeseries`,
`eda_example_events`, `eda_prefailure_signatures`, `eda_seasonality_drift`, `probability_timeline_*`) and the
screenshots.

This is not legal advice.

## Failure labels are simulated

ETDataset contains **no failure labels**. Failure events and harmless look-alike disturbances are generated by a
seeded, documented simulator (`src/data/simulation.py`, `configs/failure_simulation.yaml`). Real and simulated
values are kept in separate columns (`sim_*`), and tests prove the model never sees the simulator's metadata.
All results in this repository measure detection of these **simulated** signatures inside real operating data.
They are not real-world failure-prediction rates.
