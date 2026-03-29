# Work

Run scripts from this folder (`Work/`).

## Training (four terminals)

Use **four separate terminals**, each `cd` into `Work/` (Git Bash or WSL on Windows).

Run **`run_no_plr_5x` in two terminals** and **`run_plr_5x` in two terminals** (each script twice, one run per terminal):

| Terminal | Bash | PowerShell (Windows) |
|----------|------|---------------------|
| 1 | `bash run_no_plr_5x.sh` | `.\run_no_plr_5x.ps1` |
| 2 | `bash run_no_plr_5x.sh` | `.\run_no_plr_5x.ps1` |
| 3 | `bash run_plr_5x.sh` | `.\run_plr_5x.ps1` |
| 4 | `bash run_plr_5x.sh` | `.\run_plr_5x.ps1` |

That is **four terminals total**: two for no-PLR batches and two for PLR batches.

If PowerShell blocks scripts, run once per session: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
