"""
Load a Neural Latents Benchmark dataset (NWB) for supervised velocity decoding.

Mirrors the contdata95 pipeline: binned spike counts -> windows -> kinematics,
z-scored with training statistics only. Here the target is hand/finger VELOCITY
(2 outputs) and the metric is velocity R². Train/val come from the official
NLB `trial_info['split']` (the EvalAI leaderboard is closed, so we evaluate on
the `val` trials locally).

Returns the same shape of object as train_cnn.load_data:
    single_bin = {'train': (spikes, vel), 'val': (spikes, vel)}   # spikes (N, C)
    windowed   = {'train': (windows, vel), 'val': (windows, vel)} # windows (N, seq_len, C)
where the tuple is (model_input, target) to match make_loader(*split, ...).
"""

import numpy as np
from nlb_tools.nwb_interface import NWBDataset

# Which behavioral velocity field each dataset exposes.
VEL_FIELD = {'mc_rtt': 'finger_vel', 'mc_maze': 'hand_vel'}


def _to_seconds(series):
    """trial_info times -> float seconds, whether Timedelta or numeric."""
    if hasattr(series, 'dt'):
        return series.dt.total_seconds().to_numpy()
    return series.to_numpy(dtype=float)


def _bin_split(bin_times_s, trial_info):
    """Per-bin split label ('train'/'val'/'none') from trial start/end times."""
    split = np.full(len(bin_times_s), 'none', dtype=object)
    st = _to_seconds(trial_info['start_time'])
    et = _to_seconds(trial_info['end_time'])
    sp = trial_info['split'].to_numpy()
    for s, e, lab in zip(st, et, sp):
        split[(bin_times_s >= s) & (bin_times_s < e)] = lab
    return split


def load_nlb(nwb_path, dataset='mc_rtt', bin_ms=20, seq_len=12, smooth_sigma_ms=0.0):
    """Load + bin a dataset for velocity decoding.

    bin_ms: resample width (native is 1 ms). seq_len: window length (bins) for
    the sequence/CNN models. smooth_sigma_ms: if >0, Gaussian-smooth each channel's
    spike-count time series (a firing-rate estimate; denoises the input).
    """
    vel_field = VEL_FIELD[dataset]
    ds = NWBDataset(nwb_path, split_heldout=True)
    # nlb_tools.resample() rebins the data, then sets a cosmetic index.freq which
    # pandas 2.x rejects when the index has gaps (e.g. MC_Maze's inter-trial gaps).
    # The rebinning has already happened by then, so the freq error is safe to skip.
    try:
        ds.resample(bin_ms)
    except ValueError:
        pass
    df = ds.data

    spikes = df['spikes'].to_numpy(np.float32)          # (T, C)
    vel = df[vel_field].to_numpy(np.float32)[:, :2]     # (T, 2) -> x, y velocity
    times_s = df.index.total_seconds().to_numpy()       # bin start times (s)
    split = _bin_split(times_s, ds.trial_info)

    # Recording gaps show up as NaN in spikes and/or velocity (~0.05% of bins).
    # Exclude them as samples; windows overlapping them are dropped below.
    bad_bin = np.isnan(vel).any(axis=1) | ~np.isfinite(spikes).all(axis=1)
    split[bad_bin] = 'none'
    finite_bin = ~bad_bin

    # Optional Gaussian smoothing of the spike counts (firing-rate estimate).
    # Gap bins are zeroed first so NaNs don't spread across the kernel; the gap
    # exclusion above already used the raw finiteness, so it's unaffected.
    if smooth_sigma_ms and smooth_sigma_ms > 0:
        from scipy.ndimage import gaussian_filter1d
        spikes = gaussian_filter1d(
            np.nan_to_num(spikes, nan=0.0), sigma=smooth_sigma_ms / bin_ms, axis=0
        ).astype(np.float32)

    # Z-score spikes AND velocity targets using TRAIN bins only. Normalizing the
    # targets keeps MSE well-conditioned (raw finger velocity is large-magnitude
    # and makes the LSTM/Transformer diverge to NaN); R² is invariant to this
    # affine rescaling, so the reported metric is unchanged.
    train_mask = split == 'train'
    mu = spikes[train_mask].mean(axis=0)
    sd = spikes[train_mask].std(axis=0)
    sd[sd == 0] = 1.0
    spikes = (spikes - mu) / sd

    vmu = vel[train_mask].mean(axis=0)
    vsd = vel[train_mask].std(axis=0)
    vsd[vsd == 0] = 1.0
    vel = (vel - vmu) / vsd

    def collect(label):
        # Causal windows of past `seq_len` bins ending at bin i; single bin = spikes[i].
        idx = np.where(split == label)[0]
        idx = idx[idx >= seq_len - 1]                   # need full history
        # drop windows that reach back into a gap (non-finite) bin
        if len(idx):
            keep = np.array([finite_bin[i - seq_len + 1:i + 1].all() for i in idx])
            idx = idx[keep]
        single = spikes[idx]                            # (N, C)
        windows = np.stack([spikes[i - seq_len + 1:i + 1] for i in idx])  # (N, seq_len, C)
        targets = vel[idx]                              # (N, 2)
        return single, windows, targets

    s_tr, w_tr, y_tr = collect('train')
    s_va, w_va, y_va = collect('val')

    single_bin = {'train': (s_tr, y_tr), 'val': (s_va, y_va)}
    windowed = {'train': (w_tr, y_tr), 'val': (w_va, y_va)}

    meta = {'n_channels': spikes.shape[1], 'n_outputs': 2, 'bin_ms': bin_ms,
            'seq_len': seq_len, 'vel_field': vel_field,
            'n_train': len(y_tr), 'n_val': len(y_va)}
    return single_bin, windowed, meta


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        'nlb_data/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb'
    sb, win, meta = load_nlb(path, dataset='mc_rtt')
    print('meta:', meta)
    print('single train:', sb['train'][0].shape, '-> ', sb['train'][1].shape)
    print('window train:', win['train'][0].shape, '-> ', win['train'][1].shape)
    print('window val:  ', win['val'][0].shape, '-> ', win['val'][1].shape)
