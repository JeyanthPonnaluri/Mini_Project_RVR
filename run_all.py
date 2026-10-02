"""
Re-run every experiment shown in the Streamlit app and write results/final/*.json.

    python run_all.py              # full run (20 seeds; ~1 h on 2 CPU cores)
    python run_all.py --quick      # 3 seeds, for a fast end-to-end check
    python run_all.py --only mia   # single experiment
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from fl_study import experiments as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "final")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--processes", type=int, default=2)
    a = ap.parse_args()
    seeds = list(range(3)) if a.quick else E.SEEDS
    P = a.processes
    jobs = {
        "cohort": lambda: E.cohort_facts(),
        "base_paper": lambda: E.base_paper_replication(seeds, P),
        "centralized": lambda: E.centralized_baselines(n_repeats=2 if a.quick else 5),
        "drift": lambda: E.drift_experiment(seeds=seeds, processes=P),
        "privacy_grid": lambda: E.privacy_grid(seeds=seeds, processes=P),
        "personalization": lambda: E.personalization(seeds=seeds, processes=P),
        "shapley": lambda: E.shapley_study(seeds=seeds, processes=P,
                                           R_by_K={3: 2, 5: 2, 10: 1} if a.quick else None),
        "mia": lambda: E.mia_experiment(seeds, P),
        "shift": lambda: E.shift_experiment(seeds),
        "ablation": lambda: E.ablation(seeds, P),
    }
    os.makedirs(OUT, exist_ok=True)
    meta_path = os.path.join(OUT, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    for name, fn in jobs.items():
        if a.only and name not in a.only:
            continue
        t = time.time()
        print(f"[run_all] {name} ...", flush=True)
        res = fn()
        with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(_clean(res), f, indent=1)
        meta[name] = {"seconds": round(time.time() - t, 1), "seeds": len(seeds), "quick": a.quick,
                      "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"[run_all] {name} done in {meta[name]['seconds']} s", flush=True)
    meta["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                           "numpy": np.__version__}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
