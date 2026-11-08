"""Run all planned Norman GEARS / Set Transformer runs, skipping finished ones."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = Path(r"C:\Users\is2891\Desktop\Emily")
DATA = WORK / "gears_data"
OUT = WORK / "gears_runs"
LOG = WORK / "train_log.txt"

RUNS = [f"gears_split{s}_train101" for s in range(1, 6)] + \
       [f"gears_set_transformer_split{s}_train101" for s in range(1, 6)]

# make the conda env's git and DLLs visible when run from Task Scheduler
env_dir = Path(sys.executable).parent
env = dict(os.environ)
env["PATH"] = os.pathsep.join([str(env_dir), str(env_dir / "Library" / "bin"),
                               str(env_dir / "Scripts"), env.get("PATH", "")])

with open(LOG, "a", buffering=1) as log:
    for run in RUNS:
        if (OUT / "outputs" / "norman-benchmark" / run / "results.json").exists():
            log.write(f"===== skip {run} (done)\n")
            continue
        log.write(f"===== start {run}\n")
        code = subprocess.run(
            [sys.executable, "-W", "ignore", "scripts/norman_train.py", run,
             "--data-path", str(DATA), "--output-root", str(OUT), "--device", "cuda"],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"===== end {run} exit={code}\n")
    log.write("===== all done\n")