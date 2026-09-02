"""Unattended plan driver: P1 -> (P2 -> D1 -> D2 once their code lands).

Run once and leave the machine on:

    python -u run_plan.py

- Stages run sequentially (one GPU job at a time).
- After each training stage a health check runs; an abnormality STOPS the driver
  and leaves results/plan_state.json + results/plan_run.log for review.
- Re-run `python -u run_plan.py` to resume; finished fold-seed work is skipped
  by the trainers themselves, finished stages by this driver.
- A stage whose code is not written yet stops the driver cleanly with a marker.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATE = Path("results/plan_state.json")
LOG = Path("results/plan_run.log")
PY = sys.executable

# name, command, health-check command (or None), required-file (or None)
STAGES = [
    ("p1a_r2_lopo",
     [PY, "-u", "train_multipath_lopo.py", "--mode", "kfold", "--v1", "--rhythm",
      "--k", "5", "--epochs", "25", "--seeds", "42", "123", "--out-dir", "results/mp_lopo"],
     [PY, "-m", "validation.check_lopo_health", "results/mp_lopo", "vrce", "5", "2"],
     None),
    ("p1b_flat_lopo",
     [PY, "-u", "train_lopo_cv.py", "--arch", "flat", "--k", "5", "--epochs", "25",
      "--seeds", "42", "123", "--out-dir", "results/lopo_cv"],
     [PY, "-m", "validation.check_lopo_health", "results/lopo_cv", "flat", "5", "2"],
     None),
    ("p2_selective",
     [PY, "-u", "-m", "selective.run_p2"], None, "selective/run_p2.py"),
    ("p1_gate",
     [PY, "-u", "-m", "validation.analyze_p1_gate"], None, "validation/analyze_p1_gate.py"),
    ("d1_ds12_r2",
     [PY, "-u", "-m", "validation.run_d1"], None, "validation/run_d1.py"),
    # d2_v1_ablation is NOT auto-run: +V1-only LOPO is 33 GPU-h, worth it only
    # if the P1 gate shows R2 beats flat on S but V1-vs-rhythm attribution is
    # still open. Launch manually if so:
    #   python -u train_multipath_lopo.py --mode kfold --v1 --k 5 --epochs 25 \
    #     --seeds 42 123 --out-dir results/mp_lopo
]


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for name, cmd, health, needs in STAGES:
        if state.get(name) == "done":
            log(f"SKIP {name} (already done)")
            continue
        if needs and not Path(needs).exists():
            log(f"STOP: stage {name} needs {needs} (not written yet). "
                f"Add the code, then re-run `python -u run_plan.py`.")
            state[name] = "pending_code"
            STATE.write_text(json.dumps(state, indent=2))
            return 0

        log(f"==== START {name} ====")
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = (time.time() - t0) / 3600
        if rc != 0:
            log(f"FAIL {name}: exit {rc} after {dt:.1f} h. Driver stopped.")
            state[name] = f"failed(exit {rc})"
            STATE.write_text(json.dumps(state, indent=2))
            return 1
        if health:
            log(f"---- health check {name} ----")
            if subprocess.run(health).returncode != 0:
                log(f"ABNORMAL {name}: health check failed. Driver stopped for review.")
                state[name] = "abnormal"
                STATE.write_text(json.dumps(state, indent=2))
                return 1
        log(f"==== DONE {name} in {dt:.1f} h ====")
        state[name] = "done"
        STATE.write_text(json.dumps(state, indent=2))

    log("ALL STAGES DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
