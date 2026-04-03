# SipACL

```bash
git clone https://github.com/<your-org>/SipACL
cd SipACL
```

## Install

```bash
pip install -r Work/reqs.txt
```

## Training

```bash
cd Work
python policy/ppo.py
```

## Batch runs (10 seeds, 4 terminals)

Scripts live in **`Work/to_run/sh/`** (Bash) and **`Work/to_run/ps1/`** (PowerShell). Each script switches to **`Work/`** for you.

From **repo root** (examples below), or after **`cd Work`** use the same filenames with `to_run/...` instead of `Work/to_run/...`.

| Terminal | Bash | PowerShell |
|----------|------|------------|
| 1 | `bash Work/to_run/sh/pn1_srandom_st0_seeds_1_to_5.sh` | `.\Work\to_run\ps1\pn1_srandom_st0_seeds_1_to_5.ps1` |
| 2 | `bash Work/to_run/sh/pn1_srandom_st0_seeds_6_to_10.sh` | `.\Work\to_run\ps1\pn1_srandom_st0_seeds_6_to_10.ps1` |
| 3 | `bash Work/to_run/sh/p0p5_srandom_st0p01_seeds_1_to_5.sh` | `.\Work\to_run\ps1\p0p5_srandom_st0p01_seeds_1_to_5.ps1` |
| 4 | `bash Work/to_run/sh/p0p5_srandom_st0p01_seeds_6_to_10.sh` | `.\Work\to_run\ps1\p0p5_srandom_st0p01_seeds_6_to_10.ps1` |

- Scripts starting with **`pn1_`** turn PLR off; **`p0p5_`** turn it on with the project’s default replay settings.


To save a log of a run, put the file under **`Work/logs/`** so it stays next to other logs, not mixed with source files.
