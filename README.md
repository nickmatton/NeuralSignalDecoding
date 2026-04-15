# Neural Signal Decoding

Decoding hand position and velocity from 95-channel neural spike data recorded during a center-out reaching task. Compares MLP, 2D CNN, LSTM, and Transformer architectures on the same dataset (`contdata95.mat`, 50ms bins).

**[Live demo →](https://nickmatton.com/demos/neural-decoding/index.html)**

The demo includes a live decoding replay, architecture diagrams, and a comparison of test-set Pearson correlations across models.

## Results

| Model       | Avg Correlation |
|-------------|-----------------|
| LSTM        | 0.987           |
| MLP         | 0.876           |
| 2D CNN      | 0.788           |
| Transformer | 0.740           |

## Repo layout

- `train_all.py`, `train_cnn.py` — training scripts
- `results/` — saved model outputs and plots
- `web-demo/` — source for the hosted demo

BME 517, University of Michigan.
