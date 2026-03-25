# SipACL

```bash
git clone https://github.com/<your-org>/SipACL
cd SipACL
```

**Install** (from repo root):

```bash
pip install -r reqs.txt
```

**Hyperparameter sweep** (from `Work/`). Both scripts run the **same** grid of `policy/ppo.py` jobs **in parallel** (unique buffers under `Work/buffer_runs/` per combo):

- **`run_resample_sweep.ps1`** — requires **PowerShell 7+** (`ForEach-Object -Parallel`). Example: `pwsh ./run_resample_sweep.ps1` or `-MaxParallel 8`.
- **`run_resample_sweep.sh`** — requires **GNU parallel**; concurrency via `J` (default 4). Example: `J=4 bash run_resample_sweep.sh`.

On Windows without GNU parallel, use the `.ps1` script. On Linux/macOS/Git Bash with `parallel` installed, use the `.sh` script.
