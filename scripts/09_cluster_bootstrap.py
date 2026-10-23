"""Stage 9 - Post-hoc sensitivity check: cluster bootstrap of D over filename prefixes.

Not pre-registered (PREREGISTRATION.md §11, D4). It checks the independence assumption of the
image bootstrap: images that share a filename prefix (possibly the same scene or video) are
resampled together. The confirmatory decision does not use this script.

Input: results/main/evaluation/test/<condition>/{matches.npz, predictions.json} from Stage 6,
       results/dataset/sequence_groups.csv from Stage 3 (both under results/results/ if it exists).
Image ids follow scripts/03_prepare_dataset.py: enumerate(sorted(file names), start=1).
Cluster = file stem before the first "_" (same rule as sequence_prefix in Stage 3).
Contrasts: FP32 vs INT8 seed 0 (the primary comparison) and FP32 vs FP16 (the control).

Output: results/main/analysis/test/cluster_bootstrap.json

Usage (from the project root):
    python scripts/09_cluster_bootstrap.py [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import ap_weighted, load_match_groups  # noqa: E402

RES = ROOT / "results" / "results" if (ROOT / "results" / "results").is_dir() else ROOT / "results"
EVAL = RES / "main" / "evaluation" / "test"
BASE, TREATMENTS = "fp32", ["int8_seed0", "fp16"]
CONDS = [BASE] + TREATMENTS
B, SEED, LEVEL, METRIC = 2000, 0, 0.95, "AP50_95"

_groups: dict = {}
_cluster: np.ndarray | None = None


def _init(cluster: np.ndarray) -> None:
    global _groups, _cluster
    _groups = {c: load_match_groups(EVAL / c / "matches.npz") for c in CONDS}
    _cluster = cluster


def d_values(cluster_weights: np.ndarray) -> list[float]:
    """D for every treatment under one set of cluster weights (each condition computed once)."""
    w = cluster_weights.astype(np.float64)[_cluster]
    ap = {c: {b: ap_weighted(_groups[c], w, b)[METRIC] for b in ("small", "non_small")} for c in CONDS}
    rel = lambda t, b: (ap[BASE][b] - ap[t][b]) / ap[BASE][b]  # noqa: E731
    return [rel(t, "small") - rel(t, "non_small") for t in TREATMENTS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    img_ids = [int(i) for i in load_match_groups(EVAL / BASE / "matches.npz")["meta"]["imgIds"]]
    names = set()
    for c in CONDS:
        names |= {p["file_name"] for p in json.loads((EVAL / c / "predictions.json").read_text(encoding="utf-8"))}
    names = sorted(names)
    if len(names) != len(img_ids):
        print(f"STOP: {len(names)} file names with predictions, {len(img_ids)} test images.")
        return 1
    prefix_of = {i: Path(f).stem.split("_")[0] for i, f in enumerate(names, start=1)}

    ref = {r["prefix"]: int(r["images"]) for r in csv.DictReader(open(RES / "dataset" / "sequence_groups.csv"))
           if r["split"] == "test"}
    if dict(Counter(prefix_of[i] for i in img_ids)) != ref:
        print("STOP: prefix counts differ from sequence_groups.csv.")
        return 1
    prefixes = sorted(ref)
    cluster = np.array([prefixes.index(prefix_of[i]) for i in img_ids])
    n_cl = len(prefixes)

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_cl, size=(B, n_cl))
    weights = [np.bincount(d, minlength=n_cl) for d in draws]

    with Pool(args.workers, initializer=_init, initargs=(cluster,)) as pool:
        point = pool.apply(d_values, (np.ones(n_cl, dtype=np.int64),))
        boot = np.array(pool.map(d_values, weights, chunksize=20))  # [B, len(TREATMENTS)]

    out = {"note": "post-hoc sensitivity analysis, not pre-registered", "metric": METRIC,
           "n_images": len(img_ids), "n_clusters": n_cl,
           "largest_clusters": sorted((int(v) for v in ref.values()), reverse=True)[:5],
           "bootstrap_reps": B, "seed": SEED, "ci_level": LEVEL, "D": {}}
    for j, t in enumerate(TREATMENTS):
        x = boot[:, j][np.isfinite(boot[:, j])]
        lo, hi = np.percentile(x, [100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2])
        out["D"][f"{BASE} vs {t}"] = {"point": point[j], "ci_low": float(lo), "ci_high": float(hi),
                                      "se": float(x.std(ddof=1)), "p_le_0": float((x <= 0).mean()),
                                      "valid_resamples": int(x.size)}
        print(f"{BASE} vs {t:11s} D = {point[j] * 100:+.2f} pp   95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}]"
              f"   SE {x.std(ddof=1) * 100:.2f}")
    (RES / "main" / "analysis" / "test" / "cluster_bootstrap.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
