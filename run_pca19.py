"""
Run every experiment on the base paper's own 19-hospital cohort (PCa-19) and write
results/final/pca19_*.json.

    python run_pca19.py                    # full run (20 seeds; ~40 min on 2 CPU cores; set OMP_NUM_THREADS=1)
    python run_pca19.py --quick            # 3 seeds, small Monte-Carlo counts
    python run_pca19.py --only loco mia    # selected experiments
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from fl_study import multisite as M
from run_all import _clean

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--repro_repeats", type=int, default=50)   # must match the stored authors_code CSVs
    a = ap.parse_args()
    seeds = list(range(3)) if a.quick else M.SEEDS
    P = a.processes
    rr = 3 if a.quick else a.repro_repeats
    jobs = {
        "cohort": lambda: M.cohort(),
        "strategies": lambda: M.strategies(seeds, P),
        "privacy": lambda: M.privacy(seeds, P),
        "personalisation": lambda: M.personalisation(seeds, P),
        "mia": lambda: M.mia(seeds, P),
        "loco": lambda: M.loco(seeds, P),
        "shapley": lambda: M.shapley(seeds, P, n_perm=2 if a.quick else 3),
        "strategies_converged": lambda: M.strategies(seeds, P, overrides={"rounds": 40, "epochs": 10}),
        "baselines": lambda: M.baselines(M.VAL_SEEDS[:3] if a.quick else M.VAL_SEEDS, P),
        "baselines_tcga": lambda: M.baselines_tcga(seeds, P),
        "subsampling": lambda: M.subsampling(M.VAL_SEEDS[:3] if a.quick else M.VAL_SEEDS, P),
        "private_valuation": lambda: M.private_valuation(M.VAL_SEEDS[:3] if a.quick else M.VAL_SEEDS, P),
        "private_valuation_tcga": lambda: M.private_valuation_tcga(seeds, P),
        "valuation_mlp": lambda: M.valuation_mlp(M.VAL_SEEDS[:2] if a.quick else M.VAL_SEEDS, P),
        "valuation_plus": lambda: M.valuation_plus(M.VAL_SEEDS[:3] if a.quick else M.VAL_SEEDS, P),
        "valuation_proxies": lambda: M.valuation_proxies(M.VAL_SEEDS[:3] if a.quick else M.VAL_SEEDS, P),
        "heart_valuation": lambda: __import__("fl_study.heart", fromlist=["run"]).run(processes=P),
        "nu_sweep": lambda: __import__("fl_study.nu_sweep", fromlist=["run"]).run(processes=P),
        "reproduction_noh": lambda: M.reproduce_base_paper("noh", rr),
        "reproduction_ettala": lambda: M.reproduce_base_paper("ettala", rr),
    }
    os.makedirs(OUT, exist_ok=True)
    meta_path = os.path.join(OUT, "pca19_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    for name, fn in jobs.items():
        if a.only and name not in a.only:
            continue
        t = time.time()
        print(f"[pca19] {name} ...", flush=True)
        res = fn()
        with open(os.path.join(OUT, f"pca19_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(_clean(res), f, indent=1)
        meta[name] = {"seconds": round(time.time() - t, 1), "seeds": len(seeds), "quick": a.quick,
                      "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"[pca19] {name} done in {meta[name]['seconds']} s", flush=True)
    meta["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
