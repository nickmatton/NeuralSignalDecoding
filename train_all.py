"""
Train LSTM and Transformer on neural decoding data
and generate matching-style plots for all models.

Skips MLP and 2D CNN (already have plots from train_cnn.py).
"""

import json
import numpy as np
import torch
import torch.nn as nn
from train_cnn import (
    load_data, make_loader, train_model, evaluate, plot_decode,
)


class LSTMDecoder(nn.Module):
    def __init__(self, n_channels=95, hidden=128, n_outputs=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_channels)
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        b, t, c = x.shape
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


class TransformerDecoder(nn.Module):
    def __init__(self, n_channels=95, d_model=130, nhead=5, n_outputs=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_channels)
        self.proj = nn.Linear(n_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.3, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        b, t, c = x.shape
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\nLoading data...")
    _, windowed = load_data('contdata95.mat')

    w_train = make_loader(*windowed['train'], 64, shuffle=True)
    w_val = make_loader(*windowed['val'], 64)
    w_test = make_loader(*windowed['test'], 64)

    prep_seq = lambda y: y  # [batch, seq, channels]
    results = {}

    # ── Train LSTM ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING LSTM")
    print("=" * 60)
    model_lstm = LSTMDecoder()
    model_lstm = train_model(model_lstm, w_train, w_val, prep_seq, 100, 1e-3, device=device)
    corrs, pred, actual = evaluate(model_lstm, w_test, prep_seq, device)
    avg = np.mean(corrs)
    print(f"\n  LSTM: avg={avg:.3f} | X pos={corrs[0]:.3f}, Y pos={corrs[1]:.3f}, X vel={corrs[2]:.3f}, Y vel={corrs[3]:.3f}")
    plot_decode(pred, actual, corrs, f"LSTM (avg r = {avg:.3f})", "../../demos/lstm_decode.png")
    results['lstm'] = {'corrs': corrs, 'avg': avg}

    # ── Train Transformer ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING TRANSFORMER")
    print("=" * 60)
    model_trans = TransformerDecoder()
    model_trans = train_model(model_trans, w_train, w_val, prep_seq, 100, 1e-3, device=device)
    corrs, pred, actual = evaluate(model_trans, w_test, prep_seq, device)
    avg = np.mean(corrs)
    print(f"\n  Transformer: avg={avg:.3f} | X pos={corrs[0]:.3f}, Y pos={corrs[1]:.3f}, X vel={corrs[2]:.3f}, Y vel={corrs[3]:.3f}")
    plot_decode(pred, actual, corrs, f"Transformer (avg r = {avg:.3f})", "../../demos/transformer_decode.png")
    results['transformer'] = {'corrs': corrs, 'avg': avg}

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name in ['lstm', 'transformer']:
        r = results[name]
        print(f"  {name:12s} | avg: {r['avg']:.3f}")

    with open('lstm_transformer_results.json', 'w') as f:
        json.dump({
            name: {
                'x_pos': round(r['corrs'][0], 3),
                'y_pos': round(r['corrs'][1], 3),
                'x_vel': round(r['corrs'][2], 3),
                'y_vel': round(r['corrs'][3], 3),
                'average': round(float(r['avg']), 3),
            } for name, r in results.items()
        }, f, indent=2)
    print("\nSaved lstm_transformer_results.json")


if __name__ == '__main__':
    main()
