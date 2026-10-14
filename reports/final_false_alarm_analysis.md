# Final false-alarm analysis (deployed system, final operating point)

Test period, both transformers. Warning threshold (raw) 0.241; 52 false-alarm episodes = **0.250 per transformer-day** (1.75 per week). A false alarm is an alarm episode (2 consecutive steps over threshold, gaps ≤ 1 h merged) that does not overlap the 24 h before any (simulated) failure.

## Categories (assigned in this order)

1. **harmless simulated disturbance** — overlaps (±2 h) an injected benign thermal bump / load surge / oscillation
2. **sensor noise** — overlaps an injected sensor glitch, or a *real* channel that is normally never flat is flat-lined for ≥ 2 h (stuck sensor)
3. **unknown / simulation artefact** — starts ≤ 6 h after an outage ended (the simulator's blend-back)
4. **unusual real sensor behaviour** — the *real* (pre-injection) signal deviates ≥ 3 σ from its 24 h baseline on some channel
5. **seasonal / slow thermal behaviour** — no short-term deviation, but the real oil temperature is ≥ 3 σ above what load explains (slow ambient/seasonal heating)
6. **model error** — none of the above: no identifiable anomaly

| Category | Episodes | Share |
|---|---|---|
| harmless simulated disturbance | 24 | 46% |
| sensor noise | 1 | 2% |
| seasonal / slow thermal behaviour | 1 | 2% |
| unusual real sensor behaviour | 15 | 29% |
| model error | 11 | 21% |
| unknown / simulation artefact | 0 | 0% |

## How unusual are these conditions anyway? (base rates)

Share of random event-free 14-step windows in the test period meeting each criterion (n = 650):

| Criterion | Base rate |
|---|---|
| unusual real sensor behaviour | 22% |
| seasonal / slow thermal behaviour | 2% |
| sensor noise | 3% |

## Interpretation

* 79% of false alarms coincide with an identifiable anomaly (injected harmless disturbance, sensor noise, or unusual / slow real operating behaviour). These are anomaly detections that did not precede a simulated failure.
* 21% have no identifiable anomaly → model errors.
* Median duration 1.50 h; median confidence at peak 0.50 (vs. the alarm-confidence bins in `reports/final/final_metrics.json`).
* 16 episodes preceded a failure by 24–48 h (outside the scoring window).

Every episode (with timestamps, peak risk, confidence, real-signal deviations) is listed in `reports/final/false_alarms.json` and can be inspected in the UI (Failure Intelligence → False-alarm explorer).