"""
Multi-seed training sweep for defensible results.

Trains all four decoders (MLP, 2D CNN, LSTM, Transformer) across N seeds and
reports mean ± std per metric — instead of a single lucky run. The first seed
additionally produces the decode plots and the web-demo data so the demo shows
a real, representative model.

Outputs:
  sweep_results.json              — mean ± std over seeds (headline numbers)
  cnn_results.json                — representative-seed MLP + 2D CNN
  lstm_transformer_results.json   — representative-seed LSTM + Transformer
  results/*.png                   — representative-seed decode plots
  web-demo/results.js (+ .json)   — representative-seed demo data
"""

import json
import numpy as np
import torch

from train_cnn import (
    load_data, make_loader, train_model, evaluate, plot_decode,
    set_seed, save_demo_base, save_demo_model, SingleBinMLP, SpatioTemporalCNN,
)
from train_all import LSTMDecoder, TransformerDecoder

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100
LABELS = ['x_pos', 'y_pos', 'x_vel', 'y_vel']

# key, display name, constructor, data split, prep_fn, plot path
MODELS = [
    ('mlp', 'MLP', SingleBinMLP, 'single_bin', lambda y: y, 'results/single_bin_cnn_decode.png'),
    ('cnn2d', '2D CNN', SpatioTemporalCNN, 'windowed', lambda y: y.unsqueeze(1), 'results/spatio_temporal_cnn_decode.png'),
    ('lstm', 'LSTM', LSTMDecoder, 'windowed', lambda y: y, 'results/lstm_decode.png'),
    ('transformer', 'Transformer', TransformerDecoder, 'windowed', lambda y: y, 'results/transformer_decode.png'),
]


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    device = get_device()
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    single_bin, windowed = load_data('contdata95.mat')
    data = {'single_bin': single_bin, 'windowed': windowed}

    # Shared demo timeline (real spike data + actual kinematics).
    save_demo_base(single_bin['test'][0], single_bin['test'][1])

    rep_seed = SEEDS[0]
    acc = {key: [] for key, *_ in MODELS}  # acc[key] = list-over-seeds of [4 corrs]

    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"SEED {seed}")
        print("=" * 60)
        for key, name, ctor, dk, prep, plot in MODELS:
            set_seed(seed)
            d = data[dk]
            tr = make_loader(*d['train'], 64, shuffle=True)
            va = make_loader(*d['val'], 64)
            te = make_loader(*d['test'], 64)

            model = ctor()
            model = train_model(model, tr, va, prep, EPOCHS, 1e-3, device=device)
            corrs, pred, actual = evaluate(model, te, prep, device)
            acc[key].append(corrs)
            avg = float(np.mean(corrs))
            print(f"  seed {seed} | {name:12s} avg r = {avg:.3f}")

            if seed == rep_seed:
                plot_decode(pred, actual, corrs, f"{name} (avg r = {avg:.3f})", plot)
                save_demo_model(key, name, pred, corrs)

    # ── Aggregate mean ± std ────────────────────────────────────────────────
    sweep = {}
    for key, name, *_ in MODELS:
        arr = np.array(acc[key])  # (n_seeds, 4)
        metrics = {}
        for i, lab in enumerate(LABELS):
            col = arr[:, i]
            metrics[lab] = {
                'mean': round(float(col.mean()), 3),
                'std': round(float(col.std()), 3),
                'values': [round(float(v), 3) for v in col],
            }
        avgs = arr.mean(axis=1)
        metrics['average'] = {
            'mean': round(float(avgs.mean()), 3),
            'std': round(float(avgs.std()), 3),
            'values': [round(float(v), 3) for v in avgs],
        }
        sweep[key] = {'name': name, 'seeds': SEEDS, 'metrics': metrics}

    with open('sweep_results.json', 'w') as f:
        json.dump(sweep, f, indent=2)

    # ── Representative-seed JSONs (backwards compatible) ────────────────────
    def rep(key):
        c = acc[key][0]
        return {
            'x_pos': round(float(c[0]), 3), 'y_pos': round(float(c[1]), 3),
            'x_vel': round(float(c[2]), 3), 'y_vel': round(float(c[3]), 3),
            'average': round(float(np.mean(c)), 3),
        }

    avg_mlp = np.mean(acc['mlp'][0])
    avg_cnn2d = np.mean(acc['cnn2d'][0])
    with open('cnn_results.json', 'w') as f:
        json.dump({
            'best_model': 'single_bin' if avg_mlp >= avg_cnn2d else 'spatio_temporal',
            'correlations': rep('mlp') if avg_mlp >= avg_cnn2d else rep('cnn2d'),
            'all_results': {'single_bin': rep('mlp'), 'spatio_temporal': rep('cnn2d')},
        }, f, indent=2)
    with open('lstm_transformer_results.json', 'w') as f:
        json.dump({'lstm': rep('lstm'), 'transformer': rep('transformer')}, f, indent=2)

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"SWEEP SUMMARY — mean ± std over {len(SEEDS)} seeds")
    print("=" * 60)
    print(f"  {'Model':12s} | {'avg r (mean±std)':18s} | per-seed averages")
    for key, name, *_ in MODELS:
        m = sweep[key]['metrics']['average']
        vals = ', '.join(f"{v:.3f}" for v in m['values'])
        print(f"  {name:12s} | {m['mean']:.3f} ± {m['std']:.3f}      | {vals}")
    print("\nSaved sweep_results.json, cnn_results.json, lstm_transformer_results.json")


if __name__ == '__main__':
    main()
