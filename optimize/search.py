"""optimize/search.py — patch a track's train.py constants, run, grep, log a leaderboard.

The optimizer NEVER edits ``prepare.py`` (fixed eval/splits) and never edits ``train.py``
on disk — it writes a patched copy ``_sweep_train.py`` in the track dir, runs it under the
track's own fixed time budget, and records ``primary_metric`` + extras. Fully comparable to
hand-run experiments.

Usage:
    python -m optimize.search --track classification --data-dir ./data --trials 20
    python -m optimize.search --track generative --data-dir ./data --trials 10 --timeout 2400
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from optimize.space import SPACES   # noqa: E402

TRACK_DIR = {t: os.path.join(_ROOT, "tracks", t) for t in SPACES}


def patch_source(src: str, overrides: dict) -> str:
    for name, val in overrides.items():
        pat = re.compile(rf"^{re.escape(name)}\s*=.*$", re.MULTILINE)
        new = f"{name} = {val!r}"
        src, n = pat.subn(new, src, count=1)
        if n == 0:
            raise KeyError(f"constant {name} not found in train.py (cannot sweep it)")
    return src


def parse_metrics(stdout: str, keys) -> dict:
    out = {}
    for line in stdout.splitlines():
        m = re.match(r"^([A-Za-z0-9_@]+):\s+(-?[0-9.]+|nan)\s*$", line.strip())
        if m and m.group(1) in keys:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def sample_configs(space: dict, trials: int, grid: bool, seed: int):
    keys = list(space.keys())
    if grid:
        combos = list(itertools.product(*[space[k] for k in keys]))
        random.Random(seed).shuffle(combos)
        combos = combos[:trials] if trials > 0 else combos
        return [dict(zip(keys, c)) for c in combos]
    rng = random.Random(seed)
    seen, configs = set(), []
    while len(configs) < trials and len(seen) < 5000:
        cfg = {k: rng.choice(space[k]) for k in keys}
        sig = json.dumps(cfg, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(cfg)
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(SPACES))
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=0, help="hard kill per trial (s); 0=rely on TIME_BUDGET")
    args = ap.parse_args()

    spec = SPACES[args.track]
    primary_key = "primary_metric"
    keys = [primary_key] + spec["extra_greps"]
    higher_better = spec["higher_better"]
    track_dir = TRACK_DIR[args.track]
    data_dir = os.path.abspath(args.data_dir)

    runs_dir = os.path.join(_ROOT, "runs", args.track)
    os.makedirs(runs_dir, exist_ok=True)
    sweep_tsv = os.path.join(runs_dir, "sweep.tsv")
    new_file = not os.path.isfile(sweep_tsv)

    with open(os.path.join(track_dir, "train.py"), encoding="utf-8") as f:
        base_src = f.read()

    configs = sample_configs(spec["space"], args.trials, args.grid, args.seed)
    print(f"[{args.track}] {len(configs)} trials  data={data_dir}  higher_better={higher_better}")

    best = None
    tmp_path = os.path.join(track_dir, "_sweep_train.py")
    with open(sweep_tsv, "a", encoding="utf-8") as log:
        if new_file:
            log.write("trial\t" + primary_key + "\t" + "\t".join(spec["extra_greps"])
                      + "\tstatus\tconfig\n")
        for i, cfg in enumerate(configs):
            try:
                patched = patch_source(base_src, cfg)
            except KeyError as e:
                print(f"  trial {i}: {e}"); continue
            with open(tmp_path, "w", encoding="utf-8") as tf:
                tf.write(patched)
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, "_sweep_train.py", "--data-dir", data_dir],
                    cwd=track_dir, capture_output=True, text=True,
                    timeout=(args.timeout or None))
                stdout = proc.stdout + "\n" + proc.stderr
                status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
            except subprocess.TimeoutExpired as e:
                stdout = (e.stdout or "") + "\n[TIMEOUT]"
                status = "timeout"
            with open(os.path.join(runs_dir, f"trial_{i:03d}.log"), "w") as lf:
                lf.write(stdout)
            mets = parse_metrics(stdout, keys)
            pm = mets.get(primary_key, float("nan"))
            row = [str(i), f"{pm:.6f}"] + [f"{mets.get(k, float('nan')):.6f}"
                                           for k in spec["extra_greps"]]
            row += [status, json.dumps(cfg)]
            log.write("\t".join(row) + "\n"); log.flush()
            improved = (best is None or (pm == pm and (
                (pm > best[0]) if higher_better else (pm < best[0]))))
            if pm == pm and improved:
                best = (pm, cfg, i)
            print(f"  trial {i:3d}: {primary_key}={pm:.4f} [{status}] {time.time()-t0:.0f}s "
                  f"{'<= BEST' if (pm==pm and improved) else ''}")
    if os.path.isfile(tmp_path):
        os.remove(tmp_path)
    if best:
        print(f"\nBEST {args.track}: {primary_key}={best[0]:.4f} (trial {best[2]})\n  {best[1]}")
    print(f"log: {sweep_tsv}")


if __name__ == "__main__":
    main()
