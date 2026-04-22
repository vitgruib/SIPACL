# SipACL

```bash
git clone https://github.com/<your-org>/SipACL
cd SipACL
```

## Install

From the **repository root** (where `Scenic/`, `metadrive/`, and `Work/` sit):

```bash
pip install -r reqs.txt
```

That installs the vendored `Scenic` and `metadrive` packages in editable mode (`-e ./Scenic`, `-e ./metadrive`) and then the pinned versions from `Work/reqs.txt` (`requirements-dev.txt` is the same two steps; `reqs.txt` just includes it). Run the command from the repo root so those paths resolve. If you only install `Work/reqs.txt`, you get the pins but not the local Scenic/MetaDrive trees.

## Training

```bash
cd Work
python policy/ppo.py
```

## Batch runs (30 seeds, 4 terminals)

Scripts live in **`Work/to_run/sh/`** (Bash) and **`Work/to_run/ps1/`** (PowerShell). Each script switches to **`Work/`** for you.

Four jobs split **by replay setting and seed band**: **`pn1_`** = `--replay-resample-prob -1`; **`p0p5_`** = `--replay-resample-prob 0.5`. Everything else uses **`Args` defaults** in `policy/ppo.py` (e.g. `sampler-type random`).

From **repo root** (examples below), or after **`cd Work`** use the same filenames with `to_run/...` instead of `Work/to_run/...`.

| Terminal | Bash | PowerShell |
|----------|------|------------|
| 1 | `bash Work/to_run/sh/pn1_seeds_1_to_15.sh` | `.\Work\to_run\ps1\pn1_seeds_1_to_15.ps1` |
| 2 | `bash Work/to_run/sh/pn1_seeds_16_to_30.sh` | `.\Work\to_run\ps1\pn1_seeds_16_to_30.ps1` |
| 3 | `bash Work/to_run/sh/p0p5_seeds_1_to_15.sh` | `.\Work\to_run\ps1\p0p5_seeds_1_to_15.ps1` |
| 4 | `bash Work/to_run/sh/p0p5_seeds_16_to_30.sh` | `.\Work\to_run\ps1\p0p5_seeds_16_to_30.ps1` |


To save a log of a run, put the file under **`Work/logs/`** so it stays next to other logs, not mixed with source files.
