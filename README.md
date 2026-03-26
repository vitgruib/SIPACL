# SipACL

```bash
git clone https://github.com/<your-org>/SipACL
cd SipACL
```

**Install** (from repo root):

```bash
pip install -r reqs.txt
```

**Hyperparameter sweep** (from `Work/`). VerifAI / Scenic sampling is **`random` only** (`--sampler-type random`). Each combination gets a unique buffer under `Work/buffer_runs/` (see `policy/ppo.py` defaults).

### Supercomputer / HPC (four bash scripts, sequential, no extra libraries)

Submit **four** separate jobs if you like (one script per `replay_resample_prob` value). Together they cover the full grid (**12 runs** = 4 `p` × 3 `stale`). Staleness sweep: `plr_stale_coef` ∈ `{0, 0.01, 0.05}`.

| Script | What it runs |
|--------|----------------|
| `run_resample_sweep_part1.sh` | `random`, `p` = -1 × all `stale` (3 runs) |
| `run_resample_sweep_part2.sh` | `random`, `p` = 0.25 × all `stale` (3 runs) |
| `run_resample_sweep_part3.sh` | `random`, `p` = 0.5 × all `stale` (3 runs) |
| `run_resample_sweep_part4.sh` | `random`, `p` = 0.75 × all `stale` (3 runs) |

`p` = `--replay-resample-prob`, `stale` = `--plr-stale-coef`. Example: `cd Work && bash run_resample_sweep_part1.sh`
