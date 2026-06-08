"""
Export per-timestep NLB (MC_RTT) predictions for the web-demo tracking section.

The NLB models decode *velocity*, and nlb_*_results.json only store summary R².
A tracking visualization needs trajectories, so this script:

  1. trains the four decoders on one representative seed,
  2. picks a handful of contiguous held-out `val` reaches (each a clean,
     gap-free run of bins — the official val set is short, scattered trials),
  3. decodes velocity for every bin in each reach, then integrates BOTH the
     decoded and the true velocity into a 2D cursor path (the standard way to
     turn a velocity decoder into a visible trajectory; integrating both from a
     common origin keeps the comparison fair — see NLB cursor-trajectory plots),
  4. writes web-demo/nlb_results.js (window.NLB_DEMO_RESULTS) for nlb-demo.js.

Only real held-out reaches are exported (CLAUDE.md: the demo must use real
predictions, never synthetic data). The per-model headline vel R² shown in the
demo comes from nlb_rtt_results.json (5-seed mean ± std), not this single seed.
"""

import argparse
import json
import numpy as np
import torch

from nlb_data import load_binned, load_nlb
from train_cnn import make_loader, train_model, set_seed
from train_nlb import MODELS, get_device, carve_estop, vel_r2

DEMO_PATH = 'web-demo/nlb_results.js'


def contiguous_val_segments(split, finite_bin, seq_len, min_len, max_segs):
    """Longest contiguous runs of usable `val` bins, in time order.

    A bin is usable if it is labelled 'val' and its full `seq_len`-bin causal
    history is finite (so the window models have valid input). Returns a list of
    (start, length) for the `max_segs` longest runs, re-sorted by start time.
    """
    usable = np.zeros(len(split), dtype=bool)
    for i in range(seq_len - 1, len(split)):
        if split[i] == 'val' and finite_bin[i - seq_len + 1:i + 1].all():
            usable[i] = True

    runs, i = [], 0
    while i < len(usable):
        if usable[i]:
            j = i
            while j < len(usable) and usable[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j - i))
            i = j
        else:
            i += 1

    runs.sort(key=lambda r: r[1], reverse=True)
    runs = runs[:max_segs]
    runs.sort(key=lambda r: r[0])
    return runs


def predict_segment(model, spikes, dk, prep, start, length, seq_len, device):
    """Decode velocity (z-scored space) for every bin in a contiguous segment."""
    idx = np.arange(start, start + length)
    if dk == 'single':
        X = spikes[idx]                                              # (L, C)
    else:
        X = np.stack([spikes[i - seq_len + 1:i + 1] for i in idx])   # (L, seq_len, C)
    model.eval()
    with torch.no_grad():
        xb = prep(torch.from_numpy(X.astype(np.float32))).to(device)
        return model(xb).cpu().numpy()                              # (L, 2)


def integrate(vel, vmu, vsd, dt):
    """De-normalize z-scored velocity and integrate to a 2D path from the origin."""
    v = vel * vsd + vmu                       # back to native units
    pos = np.cumsum(v, axis=0) * dt           # (L, 2), starts near origin
    pos -= pos[0]                             # re-origin each reach at (0, 0)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nwb', default='nlb_data/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb')
    ap.add_argument('--dataset', default='mc_rtt')
    ap.add_argument('--bin_ms', type=int, default=20)
    ap.add_argument('--seq_len', type=int, default=12)
    ap.add_argument('--smooth_ms', type=float, default=40.0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0, help='representative seed to decode with')
    ap.add_argument('--min_len', type=int, default=50, help='min reach length in bins (~1s)')
    ap.add_argument('--max_segs', type=int, default=12, help='how many reaches to export')
    ap.add_argument('--metrics', default='nlb_rtt_results.json',
                    help='5-seed summary used for the headline vel R² per model')
    ap.add_argument('--out', default=DEMO_PATH)
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Time-ordered timeline (to integrate trajectories) + windowed tensors (to train).
    binned = load_binned(args.nwb, args.dataset, args.bin_ms, args.smooth_ms)
    single_bin, windowed, meta = load_nlb(args.nwb, args.dataset, args.bin_ms,
                                           args.seq_len, args.smooth_ms)
    spikes, split, finite_bin = binned['spikes'], binned['split'], binned['finite_bin']
    vel_z, vmu, vsd = binned['vel'], binned['vmu'], binned['vsd']
    C, O = meta['n_channels'], meta['n_outputs']
    dt = args.bin_ms / 1000.0
    data = {'single': single_bin, 'window': windowed}

    segs = contiguous_val_segments(split, finite_bin, args.seq_len, args.min_len, args.max_segs)
    print(f"Selected {len(segs)} held-out reaches "
          f"({sum(L for _, L in segs)} bins, "
          f"{sum(L for _, L in segs) * dt:.1f}s): lengths {[L for _, L in segs]}")

    # Headline vel R² (5-seed mean ± std) from the summary file.
    summary = json.load(open(args.metrics))['models']

    # Shared per-reach timeline: true cursor path + the z-scored spikes the
    # neural panel animates. (Integrated from the same true velocity the models
    # were trained against.)
    seg_meta, neural_cols = [], []
    actual = {'x': [], 'y': []}
    offset = 0
    for start, length in segs:
        seg_meta.append({'start': offset, 'length': int(length)})
        pos = integrate(vel_z[start:start + length], vmu, vsd, dt)
        actual['x'].extend([round(float(v), 4) for v in pos[:, 0]])
        actual['y'].extend([round(float(v), 4) for v in pos[:, 1]])
        neural_cols.append(spikes[start:start + length])             # (L, C)
        offset += length
    neural = np.concatenate(neural_cols, axis=0).T                   # (C, n_frames)

    out = {
        'dataset': args.dataset,
        'rep_seed': args.seed,
        'meta': {'n_channels': C, 'bin_ms': args.bin_ms, 'seq_len': args.seq_len,
                 'n_val': meta['n_val'], 'vel_field': meta['vel_field'],
                 'n_frames': int(offset), 'n_reaches': len(segs)},
        'segments': seg_meta,
        'neural': [[round(float(v), 3) for v in row] for row in neural],
        'actual': actual,
        'models': {},
    }

    for key, name, ctor, dk, prep in MODELS:
        set_seed(args.seed)
        X_tr, y_tr = data[dk]['train']
        gap = args.seq_len - 1 if dk == 'window' else 0
        (Xt, yt), (Xe, ye) = carve_estop(X_tr, y_tr, gap=gap)
        tr = make_loader(Xt, yt, 64, shuffle=True, drop_last=True)
        es = make_loader(Xe, ye, 64)

        model = ctor(C, O)
        print(f"\nTraining {name} (seed {args.seed})...")
        model = train_model(model, tr, es, prep, args.epochs, 1e-3, device=device, grad_clip=1.0)

        # Decode each reach and integrate to a path.
        px, py = [], []
        for start, length in segs:
            vel_pred = predict_segment(model, spikes, dk, prep, start, length, args.seq_len, device)
            pos = integrate(vel_pred, vmu, vsd, dt)
            px.extend([round(float(v), 4) for v in pos[:, 0]])
            py.extend([round(float(v), 4) for v in pos[:, 1]])

        # Report this seed's own val R² for a sanity check; headline uses summary.
        va = make_loader(*data[dk]['val'], 64)
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for yb, xb in va:
                preds.append(model(prep(yb).to(device)).cpu().numpy())
                acts.append(xb.numpy())
        seed_r2, _ = vel_r2(np.concatenate(preds), np.concatenate(acts))

        s = summary.get(key, {})
        out['models'][key] = {
            'name': name,
            'vel_r2_mean': s.get('vel_r2_mean'),
            'vel_r2_std': s.get('vel_r2_std'),
            'seed_r2': round(float(seed_r2), 3),
            'x': px, 'y': py,
        }
        print(f"  {name:12s} seed-{args.seed} val R² = {seed_r2:.3f} "
              f"(headline {s.get('vel_r2_mean')}±{s.get('vel_r2_std')})")

    # Global position scale so every reach fits the canvas the same way.
    allpos = [out['actual']['x'], out['actual']['y']]
    for m in out['models'].values():
        allpos += [m['x'], m['y']]
    out['pos_absmax'] = round(float(max(abs(v) for arr in allpos for v in arr)), 4)

    with open(args.out, 'w') as f:
        f.write('window.NLB_DEMO_RESULTS = ')
        json.dump(out, f)
        f.write(';\n')
    print(f"\nSaved {args.out}  ({out['meta']['n_frames']} frames, "
          f"{out['meta']['n_reaches']} reaches, scale {out['pos_absmax']})")


if __name__ == '__main__':
    main()
