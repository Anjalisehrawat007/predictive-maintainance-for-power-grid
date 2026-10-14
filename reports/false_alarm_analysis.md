# False-alarm analysis (selected Phase-1 model)

> **HISTORICAL (Phase 1 / Phase 2).** Kept as a research record. Operating-point metrics here use the Phase-1 threshold (0.659) and the original validation events. Final results: [`final_project_quality_report.md`](final_project_quality_report.md) and `reports/final/*.json`.

A false alarm = an alarm episode (≥ 2 consecutive steps above the warning threshold, gaps ≤ 1 h merged) that does not overlap the 24 h window before any failure.

Categorisation order: (1) overlaps an injected *benign* event (±2 h) → (2) starts ≤ 6 h after an outage ended → (3) the **real** (pre-injection) signal shows an anomaly (|24 h z-score| ≥ 3 on any channel or |OT residual| ≥ 3 σ_train) → (4) otherwise *unknown*.

## val: 17 false-alarm episodes (0.082 per asset-day)

| category                            | interpretation                          |   episodes |
|:------------------------------------|:----------------------------------------|-----------:|
| load fluctuation (benign, injected) | meaningful anomaly (harmless)           |          1 |
| outage-like / post-outage recovery  | simulation artefact (outage blend-back) |          1 |
| seasonal / real-data anomaly        | meaningful anomaly in REAL data         |          3 |
| thermal bump (benign, injected)     | meaningful anomaly (harmless)           |         10 |
| unknown                             | model mistake (no anomaly found)        |          2 |

Median duration 1.25 h; 3 episodes were followed by a failure within 48 h (outside the 24 h scoring window: early but unscored warnings).

## test: 24 false-alarm episodes (0.115 per asset-day)

| category                            | interpretation                   |   episodes |
|:------------------------------------|:---------------------------------|-----------:|
| load fluctuation (benign, injected) | meaningful anomaly (harmless)    |          5 |
| seasonal / real-data anomaly        | meaningful anomaly in REAL data  |          9 |
| thermal bump (benign, injected)     | meaningful anomaly (harmless)    |          9 |
| unknown                             | model mistake (no anomaly found) |          1 |

Median duration 1.38 h; 9 episodes were followed by a failure within 48 h (outside the 24 h scoring window: early but unscored warnings).

## How meaningful is the 'real-data anomaly' category?

The same criterion applied to random *non-alarm, event-free* windows of equal length is met in **21%** (test) / 28% (val) of windows. Among test false alarms not explained by an injected event, 90% meet it. The model's alarms are therefore *enriched* for real anomalies, but the criterion is common; treat this category as 'unusual real operating condition', not as a confirmed fault.

## Conclusion

* 96% of test false alarms coincide with a genuine anomaly (an injected harmless disturbance or an anomaly in the real data); these are *meaningful anomaly detections* that are not failure precursors.
* 0% are caused by the simulator's outage blend-back (an artefact of the simulation layer, not of the real data).
* 4% have no identifiable anomaly → genuine model mistakes.

Episode list: `reports/phase2/false_alarms_categorised.csv`.