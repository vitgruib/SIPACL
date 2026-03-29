# SipACL

```bash
git clone https://github.com/<your-org>/SipACL
cd SipACL
```

**Install** (from repo root):

```bash
pip install -r reqs.txt
```

**Training** — from `Work/`, run `policy/ppo.py` with `tyro` CLI args (e.g. `--replay-resample-prob`, `--plr-stale-coef`, `--sampler-type random`). Checkpoints go under `Work/runs/`; PLR buffers default to `Work/buffer_runs/…` per hyperparameters (see `policy/ppo.py`).

**Five-run baselines** (bash, from `Work/`):

- `run_no_plr_5x.sh` — PLR off (`--replay-resample-prob -1`), seeds 1–5.
- `run_plr_5x.sh` — PLR on with hardcoded `REPLAY_P=0.5` and `PLR_STALE_COEF=0.01`, seeds 1–5.
