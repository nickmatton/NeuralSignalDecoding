"""
Train the four decoders on a Neural Latents Benchmark dataset and report
velocity R² (mean ± std over seeds).

Reuses the architectures and training loop from the contdata95 code; only the
data (nlb_data.load_nlb), the output dim (2 = velocity), and the metric (vel R²
instead of Pearson r) change. Early stopping uses a holdout carved from the
train trials; the official `val` trials are reported (never used for selection).
"""

import argparse
import json
import numpy as np
import torch
from sklearn.metrics import r2_score

from nlb_data import load_nlb
from train_cnn import make_loader, train_model, set_seed, SingleBinMLP, SpatioTemporalCNN
from train_all import LSTMDecoder, TransformerDecoder

# key, name, constructor(n_channels, n_outputs), data split, prep_fn
MODELS = [
    ('mlp', 'MLP', lambda c, o: SingleBinMLP(c, o), 'single', lambda y: y),
    ('cnn2d', '2D CNN', lambda c, o: SpatioTemporalCNN(o), 'window', lambda y: y.unsqueeze(1)),
    ('lstm', 'LSTM', lambda c, o: LSTMDecoder(c, 128, o), 'window', lambda y: y),
    ('transformer', 'Transformer', lambda c, o: TransformerDecoder(c, 130, 10, o), 'window', lambda y: y),
]


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def vel_r2(pred, actual):
    """Per-axis R² and their average (NLB 'vel R²' = uniform average over x,y)."""
    per = r2_score(actual, pred, multioutput='raw_values')
    return float(np.mean(per)), [round(float(p), 4) for p in per]


def evaluate_r2(model, loader, prep, device):
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for yb, xb in loader:
            preds.append(model(prep(yb).to(device)).cpu().numpy())
            acts.append(xb.numpy())
    return vel_r2(np.concatenate(preds), np.concatenate(acts))


def carve_estop(X, y, frac=0.15, seed=0):
    """Random holdout from train for early stopping (not the official val set)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(n * frac)
    es, tr = perm[:k], perm[k:]
    return (X[tr], y[tr]), (X[es], y[es])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nwb', default='nlb_data/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb')
    ap.add_argument('--dataset', default='mc_rtt')
    ap.add_argument('--bin_ms', type=int, default=20)
    ap.add_argument('--seq_len', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--models', nargs='+', default=None,
                    help='subset of model keys to run (default: all)')
    ap.add_argument('--out', default='nlb_rtt_results.json')
    args = ap.parse_args()

    run_models = [m for m in MODELS if args.models is None or m[0] in args.models]

    device = get_device()
    print(f"Using device: {device}")
    print(f"\nLoading {args.dataset} ({args.bin_ms} ms bins, window {args.seq_len})...")
    single_bin, windowed, meta = load_nlb(args.nwb, args.dataset, args.bin_ms, args.seq_len)
    print(f"  {meta['n_channels']} channels, {meta['n_train']} train / {meta['n_val']} val samples, target {meta['vel_field']}")
    data = {'single': single_bin, 'window': windowed}
    C, O = meta['n_channels'], meta['n_outputs']

    acc = {key: [] for key, *_ in run_models}
    for seed in args.seeds:
        print("\n" + "=" * 56 + f"\nSEED {seed}\n" + "=" * 56)
        for key, name, ctor, dk, prep in run_models:
            set_seed(seed)
            X_tr, y_tr = data[dk]['train']
            (Xt, yt), (Xe, ye) = carve_estop(X_tr, y_tr, seed=seed)
            tr = make_loader(Xt, yt, 64, shuffle=True)
            es = make_loader(Xe, ye, 64)
            va = make_loader(*data[dk]['val'], 64)

            model = ctor(C, O)
            # grad clipping: recurrent/attention decoders diverge without it here
            model = train_model(model, tr, es, prep, args.epochs, 1e-3, device=device, grad_clip=1.0)
            avg, per = evaluate_r2(model, va, prep, device)
            acc[key].append(avg)
            print(f"  seed {seed} | {name:12s} vel R² = {avg:.3f}  (x {per[0]}, y {per[1]})")

    # Aggregate mean ± std
    results = {}
    print("\n" + "=" * 56 + f"\nSUMMARY — {args.dataset} vel R² (mean ± std, {len(args.seeds)} seeds)\n" + "=" * 56)
    for key, name, *_ in run_models:
        a = np.array(acc[key])
        results[key] = {'name': name, 'vel_r2_mean': round(float(a.mean()), 3),
                        'vel_r2_std': round(float(a.std()), 3), 'values': [round(float(v), 3) for v in a]}
        print(f"  {name:12s} | {a.mean():.3f} ± {a.std():.3f}   {results[key]['values']}")

    out = {'dataset': args.dataset, 'meta': meta, 'seeds': args.seeds, 'models': results}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == '__main__':
    main()
