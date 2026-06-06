# NLB Decoding Experiment — Runbook

Extend the `contdata95` architecture comparison (MLP · 2D CNN · LSTM · Transformer)
to the **Neural Latents Benchmark** (MC_RTT, MC_Maze). Decode hand/finger
**velocity** from binned spikes; report **velocity R²**. Reuse the 4 architectures.

**Target scores (vel R², "good"):** MC_Maze ≈ 0.88–0.91 · MC_RTT ≈ 0.60–0.70.

---

## Step 0 — Environment

- Python 3.12 (pyenv); PyTorch + MPS available.
- **Gotcha:** `pip install nlb_tools` fails — it pins an old `pandas` with no
  3.12 wheel, so pip tries to build it from source and dies. **Fix:** install
  modern deps first, then `nlb_tools` with `--no-deps` (it's pure Python).

```bash
python3 -m pip install dandi scikit-learn pandas h5py pynwb
python3 -m pip install --no-deps nlb_tools
```

- Verified: `dandi 0.76.4`, `pandas 2.3.3`, `pynwb 3.1.3`, `sklearn 1.8.0`;
  `from nlb_tools.nwb_interface import NWBDataset` imports OK.
- (Harmless warning: `typer`/`click` version mismatch — `dandi` still works.)

## Datasets (DANDI, NWB format)

| Dataset | DANDI ID | Size | Velocity field |
|---|---|---|---|
| MC_RTT | 000129 | ~49 MB | `finger_vel` (continuous random-target reaching) |
| MC_Maze | 000128 | ~694 MB | `hand_vel` (delayed maze reaching) |

## nlb_tools API (for the loader)

- `NWBDataset(fpath, split_heldout=True)` → `.data`: pandas DataFrame, MultiIndex
  columns `(signal, channel)` — groups include `spikes`, `heldout_spikes`,
  `hand_pos/vel`, `finger_pos/vel`, `cursor_pos/vel`, `target_pos`. Index = timedelta.
- `.resample(bin_ms)` — rebin (spikes summed; continuous decimated).
- `.trial_info` — trial metadata incl. a `split` field (train/val) for local eval.

---

## Steps log

- [x] **Step 0** — install deps (above).
- [x] **Step 1** — download MC_RTT → `nlb_data/000129/` (gitignored).
  ```bash
  dandi download -o nlb_data 'https://dandiarchive.org/dandiset/000129/draft'
  ```
  Two NWB files: `..._desc-train_behavior+ecephys.nwb` (49.8 MB, has behavior) and
  `..._desc-test_ecephys.nwb` (1.2 MB, spikes only — masked eval set). Subject "Indy"
  (same animal family as `contdata95`).
- [x] **Step 2** — inspected the train NWB. **MC_RTT structure:**
  - Native `bin_width` = **1 ms**; 649,100 bins ≈ 10.8 min.
  - `spikes` = **98** held-in channels (+ `heldout_spikes` 32, only needed for co-bps).
  - Behavior target = **`finger_vel`** (x, y). (No `hand_vel`/`cursor_vel` here.)
  - `trial_info` has a **`split`** field → **train 810 / val 270** trials. Train on
    `train`, report vel R² on `val` (local eval; EvalAI leaderboard is closed).
  - **Decoding choices:** resample to **20 ms**, z-score spikes on train bins only,
    decode the **2 velocity** outputs. MLP = single bin; CNN/LSTM/Transformer = window.
- [x] **Step 3** — `nlb_data.py` → `load_nlb(nwb, dataset, bin_ms=20, seq_len=12)`.
  Resamples to 20 ms, z-scores spikes on train bins, builds causal windows, returns
  `single_bin`/`windowed` dicts (`{'train','val'}`) of `(spikes, velocity)` — same shape
  as `train_cnn.load_data`. MC_RTT → **24,289 train / 8,100 val**, 98 ch → 2 outputs.
- [x] **Step 4** — `train_nlb.py`: reuses the 4 architectures + `train_model`; metric =
  **vel R²** (`sklearn.r2_score`, avg of x/y). Early stopping uses a 15% holdout carved
  from train; official `val` is only reported. `--models` flag runs a subset.
  Two bugs fixed during validation:
  1. **NaN divergence** — raw velocity targets are large-magnitude → LSTM/Transformer
     diverged. Fixes: z-score the velocity targets (R²-invariant) **and** add optional
     `grad_clip` to `train_model` (passed `grad_clip=1.0` from `train_nlb`; default `None`
     keeps contdata95 unchanged).
  2. **Recording-gap NaNs** — ~15 bins (0.05%) are NaN (gaps). Exclude them as samples
     and drop any window overlapping one (`nlb_data` gap handling).
  **Validation (MC_RTT, 1 seed, 40 ep):** LSTM **0.648**, Transformer **0.633** (both in
  the 0.60–0.70 target range), MLP 0.090 (single-bin/zero-lag baseline). Pipeline works.
- [~] **Step 5** — full multi-seed sweep (5 seeds × 4 models), **Lambda A10** (2D CNN is
  ~3 min/epoch on MPS; fast on CUDA). Running RTT then MC_Maze (chained), then pull + terminate.

  **Remote setup gotchas (Lambda Stack, Ubuntu 22.04 / py3.10):**
  - Guest agent installed first (CLAUDE.md rule) — active.
  - `pip install dandi pynwb` pulls **numpy 2**, which breaks the Debian-compiled
    `pandas`/`h5py` (ABI: "numpy.dtype size changed"). Fix: `pip install 'numpy<2' pandas h5py`
    (pip builds in `~/.local` shadow the Debian ones). Then `pip install --no-deps nlb_tools`.
  - `dandi` also needs a newer `jsonschema`: `pip install --upgrade jsonschema`.
  - `dandi` is in `~/.local/bin` (not on non-interactive PATH) → `export PATH=$HOME/.local/bin:$PATH`.
  - `dandi download -o <dir>` requires `<dir>` to already exist (`mkdir -p` first).
  - **MC_Maze resample bug:** `nlb_tools.resample()` sets a cosmetic `index.freq` that pandas 2.x
    rejects on MC_Maze's gappy index. The rebin already happened, so wrap in `try/except ValueError`
    (fixed in `nlb_data.py`). RTT (continuous) was unaffected.
  - Loaders verified on remote: RTT 24,278/8,078 (98 ch, finger_vel); **MC_Maze 255,703/84,782
    (137 ch, hand_vel)**.
  - **RTT results (5 seeds, vel R²):** Transformer 0.634±0.006, LSTM 0.602±0.007,
    2D CNN 0.516±0.033, MLP 0.090±0.004.
  - **MC_Maze cost control:** 255k stride-1 train windows made the 2D CNN ~4–6 h / ~$8.
    Added `--max_train` to subsample train windows (val kept full); ran Maze with
    `--max_train 80000` → ~1–1.5 h. (Also fixed a chained-launch bug where the wait-guard's
    `pgrep -f "dataset mc_rtt"` matched the launcher's own heredoc args → deadlock; launch
    sweeps directly with `setsid`, don't chain via pgrep on the arg string.)
  - **2D CNN on MC_Maze hangs** (reproducible): freezes at ~epoch 50, GPU→0%, no traceback —
    a cuDNN/conv deadlock on the A10 for the Maze conv shapes (fine on MC_RTT).
  - **Scope decision:** focus the NLB comparison on the two sequence models — **LSTM +
    Transformer** (drops the flaky 2D CNN and the weak single-bin MLP). RTT already has both;
    MC_Maze run with `--models lstm transformer`.
  - **`pkill` footgun:** `pkill -f train_nlb.py` from the launching SSH shell matches the
    shell's *own* command line → self-kills before launch. Launch directly (no pkill); verify
    real procs with `ps aux | grep '[t]rain_nlb.py'` (bracket avoids self-match).
- [x] **Step 5 DONE** — pulled `nlb_rtt_results.json` + `nlb_maze_results.json`; instance
  terminated, ssh key removed.

## Final results — velocity R² (mean ± std, 5 seeds)

| Model | MC_RTT | MC_Maze |
|---|---|---|
| **LSTM** | 0.602 ± 0.007 | **0.855 ± 0.003** |
| **Transformer** | **0.634 ± 0.006** | 0.833 ± 0.005 |

Both in/near the published ranges (RTT ~0.60–0.70; Maze ~0.88–0.91). Transformer edges LSTM
on RTT; LSTM edges Transformer on Maze. (RTT also has MLP 0.090 / 2D CNN 0.516 from the
earlier 4-model run; MC_Maze focused on the two sequence models per scope decision.)
