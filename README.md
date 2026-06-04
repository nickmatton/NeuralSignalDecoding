# Neural Signal Decoding

Decoding hand position and velocity from 95-channel motor-cortex **spike-count** data recorded during a reaching task. Compares MLP, 2D CNN, LSTM, and Transformer architectures on the same dataset (`contdata95.mat`, 50 ms bins).

**[Live demo →](https://nickmatton.com/demos/neural-decoding/index.html)**

The demo replays **real held-out predictions** on real spike data, with architecture diagrams and a comparison of test-set Pearson correlations across models.

## Results

Average Pearson `r` on the held-out test set, **mean ± std over 5 seeds** (Lambda A10 GPU sweep):

| Model       | Avg r (mean ± std) |
|-------------|--------------------|
| LSTM        | **0.986 ± 0.000**  |
| Transformer | 0.982 ± 0.001      |
| MLP         | 0.883 ± 0.006      |
| 2D CNN      | 0.836 ± 0.047      |

Sequence models with 1.6 s of temporal context decode kinematics almost perfectly. The Transformer required **sinusoidal positional encoding** to reach parity with the LSTM — without it, attention treats the window as an unordered set and scores only 0.74. Full per-metric breakdown in `sweep_results.json`.

## Repo layout

- `train_cnn.py` — MLP + 2D CNN; `train_all.py` — LSTM + Transformer
- `sweep.py` — multi-seed sweep → `sweep_results.json` + plots + demo data
- `results/` — saved decode plots
- `web-demo/` — interactive demo (loads real exported predictions from `results.js`)
- `presentation/` — reveal.js interview slide deck (`open presentation/index.html`)

BME 517, University of Michigan.
