# Dataset review (Phase 2 of the final upgrade)

**Question:** is there a public dataset that would materially improve the project: real transformer
failures, richer electrical measurements, weather, maintenance records?

| Candidate | What it has | Labels | Verdict |
|---|---|---|---|
| **ETDataset ETTm1/ETTm2** (Zhou et al., AAAI 2021) — *current* | 2 transformers, 15 min, 2 years: HV/MV/LV active & reactive load, oil temperature | none | **Kept.** Real, long, multi-asset, open licence, contains a genuine thermal health signal. |
| Kaggle "transformer fault" dataset (used in arXiv:2505.06295) | 1 transformer, 15 min, 2019-06-25 → 2020-04-14: phase voltages and currents, oil/winding/ambient temperature, oil level | "Faulty" = any of the transformer's own **binary alarm/trip indicators** (winding-temperature indicator, oil-temperature alarm and trip, magnetic oil-gauge alarm) | **Rejected.** The labels are alarm states derived from the same temperature sensors, so predicting them is close to circular. There are no recorded failures, a single asset, 10 months only, and it needs an account login. The paper's random 80/20 split after SMOTE is also not a temporal evaluation. |
| DGA / dissolved-gas datasets (IEC TC10, nuclear-plant transformer DGA) | gas concentrations; some time series | operating-mode / diagnosis classes | Rejected: a different sensing modality with no load or temperature stream; diagnosis, not early warning. |
| Simulated grid-event datasets (e.g. PSML, UCI grid stability) | PMU-level or tabular simulated data | simulated | Rejected: fully simulated, not asset health. |
| Utility outage reports (EIA OE-417) | real outage events | real but coarse (regional) | Rejected: no aligned per-asset sensor streams. |

**Conclusion:** no public dataset combines per-asset sensor streams with real failure labels. The project
keeps the real ETT sensor data, keeps simulated failure labels clearly separated from real measurements, and
invests instead in making the simulation documented, configurable and stress-tested
(`configs/failure_simulation.yaml`, `src/final/robustness.py`).

Sources: https://github.com/zhouhaoyi/ETDataset · https://arxiv.org/abs/2505.06295 ·
https://www.kaggle.com/datasets/pythonafroz/transformer-fault-analysis
