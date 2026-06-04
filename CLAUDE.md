# Project: Neural Signal Decoding

Decode hand kinematics (X/Y position + velocity) from 95-channel motor-cortex
spike counts (`contdata95.mat`). Models: MLP, 2D CNN, LSTM, Transformer.

- `train_cnn.py` — MLP + 2D CNN
- `train_all.py` — LSTM + Transformer (imports shared helpers from `train_cnn.py`)
- `sweep.py` — multi-seed sweep (mean ± std) + exports demo data and plots
- `web-demo/` — interactive demo; loads real exported predictions from `web-demo/results.js`
- `presentation/` — reveal.js interview slide deck

## Lambda Cloud GPU workflow

The 2D CNN is the bottleneck (~8 h on CPU, ~90 min on local MPS, ~minutes on a
CUDA GPU). For sweeps, train on a Lambda Cloud A10 (`gpu_1x_a10`).

API key lives in `.env` as `LAMBDA_API_KEY` (gitignored). Use the REST API at
`https://cloud.lambdalabs.com/api/v1/` with `curl -u "$LAMBDA_API_KEY:"`.

### ALWAYS install the guest agent after launching a new instance

Right after a new Lambda instance becomes `active` and SSH is up, install the
Lambda guest agent so GPU/VRAM + system metrics show up in the Cloud console.
Ref: https://docs.lambda.ai/public-cloud/guest-agent/

As good practice for any root-level install, download the script, read it, and
then run the saved file (rather than piping a network download straight into a
privileged shell):

```bash
# on the instance (ubuntu@<IP>):
curl -fsSL https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh -o /tmp/lga_install.sh
cat /tmp/lga_install.sh          # review: it just adds Lambda's signed APT repo + installs the pkg
sudo bash /tmp/lga_install.sh
sudo systemctl --no-pager status lambda-guest-agent.service   # verify: active (running), telegraf
```

The agent runs as `lambda-guest-agent.service` (telegraf, inputs include
`nvidia_smi`). Metrics appear in the console within a few minutes.

### Teardown

Always terminate the instance when the run finishes (it bills per hour):

```bash
curl -u "$LAMBDA_API_KEY:" -X POST \
  https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -H "Content-Type: application/json" -d '{"instance_ids":["<ID>"]}'
```

## Conventions

- Temporal train/val/test split (70/15/15) — never shuffle across time (leakage).
- Z-score neural data using **train statistics only**.
- Seed everything via `set_seed()` for reproducibility.
- The web demo must use **real** exported predictions — never synthetic/faked data.
