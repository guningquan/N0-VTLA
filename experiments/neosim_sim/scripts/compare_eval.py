#!/usr/bin/env python
"""S8 statistics: success rates with Wilson 95% CIs, and an exact McNemar test on common seeds.

usage: compare_eval.py <results_A.jsonl> <results_B.jsonl> [--names A B]
Each line of results.jsonl: {"seed": int, "success": bool, "error": bool, ...} (written by
eval_n0vtla_dual_joint_cl.py). Errored seeds are excluded; the paired test uses only seeds that
finished without error in both runs.
"""
import argparse
import json
import math


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["seed"])] = r
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: binomial test on the b + c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    args = ap.parse_args()
    runs = [load(args.a), load(args.b)]
    for name, rows in zip(args.names, runs):
        ok = [r for r in rows.values() if not r.get("error")]
        k = sum(bool(r["success"]) for r in ok)
        lo, hi = wilson(k, len(ok))
        print(f"{name}: {k}/{len(ok)} = {k / max(1, len(ok)):.1%}  95% CI [{lo:.1%}, {hi:.1%}]  "
              f"(errored seeds excluded: {len(rows) - len(ok)})")
    common = sorted(s for s in runs[0] if s in runs[1]
                    and not runs[0][s].get("error") and not runs[1][s].get("error"))
    b = sum(bool(runs[0][s]["success"]) and not runs[1][s]["success"] for s in common)
    c = sum(not runs[0][s]["success"] and bool(runs[1][s]["success"]) for s in common)
    ka = sum(bool(runs[0][s]["success"]) for s in common)
    kb = sum(bool(runs[1][s]["success"]) for s in common)
    n = len(common)
    print(f"paired on {n} common seeds: {args.names[0]} {ka}/{n}, {args.names[1]} {kb}/{n}, "
          f"diff {(kb - ka) / max(1, n):+.1%}")
    print(f"discordant: only {args.names[0]} succeeded = {b}, only {args.names[1]} succeeded = {c}; "
          f"exact McNemar p = {mcnemar_exact(b, c):.4f}")


if __name__ == "__main__":
    main()
