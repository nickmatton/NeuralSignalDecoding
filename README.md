# Neural Signal Decoding

Decoding hand position and velocity from 95-channel motor-cortex **spike-count** data recorded during a reaching task. Compares MLP, 2D CNN, LSTM, and Transformer architectures on the same dataset (`contdata95.mat`, 50 ms bins).

**[Live demo →](https://nickmatton.com/demos/neural-decoding/index.html)**

The demo replays **real held-out predictions** on real spike data, with architecture diagrams and a comparison of test-set Pearson correlations across models.

## Results

Average Pearson `r` on the held-out test set, **mean ± std over 5 seeds** (Lambda A10 GPU sweep):

| Model       | Avg r (mean ± std) |
|-------------|--------------------|
| LSTM        | **0.989 ± 0.000**  |
| Transformer | 0.985 ± 0.001      |
| 2D CNN      | 0.960 ± 0.004      |
| MLP         | 0.928 ± 0.001      |

All four models clear 0.93 once the loss is balanced. Two findings drove the numbers: **(1)** z-scoring the kinematic targets (position and velocity differ ~4× in scale) so the MSE weights all four outputs equally — this alone lifted the 2D CNN **0.84 → 0.96** and the MLP **0.88 → 0.93** (their apparent "position weakness" was a loss-scaling artifact, not architecture); **(2)** the Transformer needs **sinusoidal positional encoding** to use temporal order (without it, attention treats the window as an unordered set and scores ~0.74). Sequence models still lead, but only modestly. Full per-metric breakdown in `sweep_results.json`.

## Validation on a public benchmark (Neural Latents Benchmark)

The `contdata95` numbers are a single session, where slowly-varying position is autocorrelated and inflates Pearson `r`. To confirm the decoders are genuinely good, the two sequence models were re-run on the **[Neural Latents Benchmark](https://neurallatents.github.io/)** — standardized datasets (NWB / DANDI), official held-out split, scored with **velocity R²**, on two different monkeys.

Velocity R², **mean ± std over 5 seeds** (Lambda A10 GPU; spike counts Gaussian-smoothed, 40 ms):

| Model       | MC_RTT (Indy, random-target) | MC_Maze (Jenkins, maze reach) |
|-------------|------------------------------|-------------------------------|
| LSTM        | 0.636 ± 0.013                | **0.881 ± 0.003**             |
| Transformer | **0.677 ± 0.012**            | 0.865 ± 0.003                 |

Both land at/near the top of the published ranges (RTT ≈ 0.60–0.70, Maze ≈ 0.88–0.91) — so the decoding skill is real, not an autocorrelation artifact. **Transformer wins RTT, LSTM wins Maze** (neither dominates). A simple **40 ms Gaussian smoothing** of the spike counts (a firing-rate estimate) lifted every number — e.g. Transformer RTT 0.58 → 0.68. Early stopping uses a contiguous temporal holdout (a random one leaks across stride-1 overlapping windows). Pipeline in `nlb_data.py` / `train_nlb.py`; results in `nlb_rtt_results.json` / `nlb_maze_results.json`; full runbook in `docs/nlb-experiment.md`.

## Repo layout

- `train_cnn.py` — MLP + 2D CNN; `train_all.py` — LSTM + Transformer
- `sweep.py` — multi-seed sweep → `sweep_results.json` + plots + demo data
- `nlb_data.py` / `train_nlb.py` — Neural Latents Benchmark loader + velocity-R² sweep
- `results/` — saved decode plots
- `web-demo/` — interactive demo (loads real exported predictions from `results.js`)
- `presentation/` — reveal.js interview slide deck (`open presentation/index.html`)
- `docs/nlb-experiment.md` — NLB experiment runbook

BME 517, University of Michigan.
