"""
Train CNNs for Neural Decoding — Two Experiments
Experiment 1: Single-bin CNN with fixed hyperparams (dropout 0.3 instead of 0.8)
Experiment 2: 2D CNN with 32-step temporal context (spatio-temporal)
"""

import argparse
import json
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Models ──────────────────────────────────────────────────────────────────

class SingleBinCNN(nn.Module):
    """Original-style architecture: single time bin, fixed dropout."""
    def __init__(self, n_channels=95, n_outputs=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_outputs),
        )

    def forward(self, x):
        return self.net(x)


class SpatioTemporalCNN(nn.Module):
    """2D CNN treating (time, channels) as a grayscale image."""
    def __init__(self, n_outputs=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        # x: [batch, 1, 32, 95]
        f = self.features(x).flatten(1)
        return self.head(f)


# ── Data ────────────────────────────────────────────────────────────────────

def load_data(data_path, seq_len=32):
    mat = scipy.io.loadmat(data_path)
    X = mat['X'].astype(np.float32)  # (T, 4) kinematics
    Y = mat['Y'].astype(np.float32)  # (T, 95) neural

    T, n_channels = Y.shape

    # Temporal split
    n_train = int(T * 0.7)
    n_val = int(T * 0.15)

    # Z-score normalize neural data using training stats
    y_mean = Y[:n_train].mean(axis=0)
    y_std = Y[:n_train].std(axis=0)
    y_std[y_std == 0] = 1.0
    Y = (Y - y_mean) / y_std

    # Single-bin data
    single_bin = {
        'train': (Y[:n_train], X[:n_train]),
        'val': (Y[n_train:n_train+n_val], X[n_train:n_train+n_val]),
        'test': (Y[n_train+n_val:], X[n_train+n_val:]),
    }

    # Windowed data for 2D CNN
    def make_windows(y, x, sl):
        n = len(y) - sl + 1
        windows = np.stack([y[i:i+sl] for i in range(n)])  # (n, seq_len, 95)
        targets = x[sl-1:sl-1+n]  # target at end of window
        return windows, targets

    win_train_y, win_train_x = make_windows(Y[:n_train], X[:n_train], seq_len)
    win_val_y, win_val_x = make_windows(
        Y[n_train-seq_len+1:n_train+n_val],
        X[n_train-seq_len+1:n_train+n_val],
        seq_len,
    )
    win_test_y, win_test_x = make_windows(
        Y[n_train+n_val-seq_len+1:],
        X[n_train+n_val-seq_len+1:],
        seq_len,
    )

    windowed = {
        'train': (win_train_y, win_train_x),
        'val': (win_val_y, win_val_x),
        'test': (win_test_y, win_test_x),
    }

    return single_bin, windowed


def make_loader(y, x, batch_size, shuffle=False):
    ds = TensorDataset(torch.from_numpy(y), torch.from_numpy(x))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── Training ────────────────────────────────────────────────────────────────

def pearson_corr(pred, actual):
    """Per-column Pearson correlation."""
    corrs = []
    for i in range(pred.shape[1]):
        p, a = pred[:, i], actual[:, i]
        corr = np.corrcoef(p, a)[0, 1]
        corrs.append(corr if not np.isnan(corr) else 0.0)
    return corrs


def train_model(model, train_loader, val_loader, prep_fn, epochs=100, lr=1e-3, patience=15, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for yb, xb in train_loader:
            yb, xb = prep_fn(yb).to(device), xb.to(device)
            pred = model(yb)
            loss = criterion(pred, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for yb, xb in val_loader:
                yb, xb = prep_fn(yb).to(device), xb.to(device)
                pred = model(yb)
                val_loss += criterion(pred, xb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, test_loader, prep_fn, device='cpu'):
    model.eval()
    all_pred, all_actual = [], []
    with torch.no_grad():
        for yb, xb in test_loader:
            yb = prep_fn(yb).to(device)
            pred = model(yb)
            all_pred.append(pred.cpu().numpy())
            all_actual.append(xb.numpy())
    pred = np.concatenate(all_pred)
    actual = np.concatenate(all_actual)
    corrs = pearson_corr(pred, actual)
    return corrs, pred, actual


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_decode(pred, actual, corrs, title, filename, n_points=500):
    labels = ['X Position', 'Y Position', 'X Velocity', 'Y Velocity']
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14)
    t = np.arange(n_points) * 0.05  # seconds

    for i, ax in enumerate(axes):
        ax.plot(t, actual[:n_points, i], 'b-', alpha=0.7, label='Actual')
        ax.plot(t, pred[:n_points, i], 'r-', alpha=0.7, label='Predicted')
        ax.set_ylabel(labels[i])
        ax.set_title(f'{labels[i]} — r = {corrs[i]:.3f}')
        ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved plot: {filename}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='contdata95.mat')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\nLoading data...")
    single_bin, windowed = load_data(args.data_path)

    results = {}

    # ── Experiment 1: Single-bin CNN ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Single-bin CNN (fixed dropout 0.3)")
    print("=" * 60)

    sb_train = make_loader(*single_bin['train'], args.batch_size, shuffle=True)
    sb_val = make_loader(*single_bin['val'], args.batch_size)
    sb_test = make_loader(*single_bin['test'], args.batch_size)

    model1 = SingleBinCNN()
    prep_single = lambda y: y  # no reshape needed
    model1 = train_model(model1, sb_train, sb_val, prep_single, args.epochs, args.lr, device=device)
    corrs1, pred1, actual1 = evaluate(model1, sb_test, prep_single, device)

    labels = ['X pos', 'Y pos', 'X vel', 'Y vel']
    print("\n  Single-bin CNN results:")
    for l, c in zip(labels, corrs1):
        print(f"    {l}: {c:.3f}")
    avg1 = np.mean(corrs1)
    print(f"    Average: {avg1:.3f}")

    plot_decode(pred1, actual1, corrs1, 'Single-bin CNN (Dropout 0.3)', 'single_bin_cnn_decode.png')
    results['single_bin'] = {'corrs': corrs1, 'avg': avg1}

    # ── Experiment 2: 2D CNN ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: 2D Spatio-temporal CNN (32 x 95)")
    print("=" * 60)

    w_train = make_loader(*windowed['train'], args.batch_size, shuffle=True)
    w_val = make_loader(*windowed['val'], args.batch_size)
    w_test = make_loader(*windowed['test'], args.batch_size)

    model2 = SpatioTemporalCNN()
    prep_2d = lambda y: y.unsqueeze(1)  # [batch, 32, 95] -> [batch, 1, 32, 95]
    model2 = train_model(model2, w_train, w_val, prep_2d, args.epochs, args.lr, device=device)
    corrs2, pred2, actual2 = evaluate(model2, w_test, prep_2d, device)

    print("\n  2D CNN results:")
    for l, c in zip(labels, corrs2):
        print(f"    {l}: {c:.3f}")
    avg2 = np.mean(corrs2)
    print(f"    Average: {avg2:.3f}")

    plot_decode(pred2, actual2, corrs2, '2D Spatio-temporal CNN', 'spatio_temporal_cnn_decode.png')
    results['spatio_temporal'] = {'corrs': corrs2, 'avg': avg2}

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Single-bin CNN avg correlation:     {avg1:.3f}")
    print(f"  2D Spatio-temporal CNN avg corr:    {avg2:.3f}")
    print(f"  Original CNN (demo, dropout 0.8):   0.100")

    # Save best results for demo integration
    best_name = 'spatio_temporal' if avg2 > avg1 else 'single_bin'
    best = results[best_name]
    output = {
        'best_model': best_name,
        'correlations': {
            'x_pos': round(best['corrs'][0], 3),
            'y_pos': round(best['corrs'][1], 3),
            'x_vel': round(best['corrs'][2], 3),
            'y_vel': round(best['corrs'][3], 3),
            'average': round(best['avg'], 3),
        },
        'all_results': {
            'single_bin': {
                'x_pos': round(results['single_bin']['corrs'][0], 3),
                'y_pos': round(results['single_bin']['corrs'][1], 3),
                'x_vel': round(results['single_bin']['corrs'][2], 3),
                'y_vel': round(results['single_bin']['corrs'][3], 3),
                'average': round(results['single_bin']['avg'], 3),
            },
            'spatio_temporal': {
                'x_pos': round(results['spatio_temporal']['corrs'][0], 3),
                'y_pos': round(results['spatio_temporal']['corrs'][1], 3),
                'x_vel': round(results['spatio_temporal']['corrs'][2], 3),
                'y_vel': round(results['spatio_temporal']['corrs'][3], 3),
                'average': round(results['spatio_temporal']['avg'], 3),
            },
        },
    }

    with open('cnn_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to cnn_results.json")


if __name__ == '__main__':
    main()
