
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    samples = list(ms)
    if not samples:
        raise AgreementError("no samples.")

    finite = []
    for i, x in enumerate(samples):
        value = _finite(x, f"latency sample[{i}]")
        if value < 0.0:
            raise AgreementError(f"latency sample[{i}] is {value}, which is negative.")
        finite.append(value)

    n = len(finite)
    mean = sum(finite) / n
    worst = max(finite)
    p50 = _percentile(finite, 50)
    p95 = _percentile(finite, 95)
    p99 = _percentile(finite, 99)
    jitter = p95 - p50

    return {
        "n": n,
        "mean": mean,
        "worst": worst,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "jitter": jitter,
    }


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    base = _finite(base_ms, "base_ms")
    other = _finite(other_ms, "other_ms")
    if base <= 0.0:
        raise AgreementError(f"base_ms is {base}, which is not positive.")
    if other <= 0.0:
        raise AgreementError(f"other_ms is {other}, which is not positive.")

    factor = base / other
    return {
        "factor": factor,
        "faster": factor > 1.0,
        "saved_ms": base - other,
    }


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    if len(a) != len(b):
        raise AgreementError(f"length mismatch: {len(a)} != {len(b)}.")
    if not a:
        raise AgreementError("no entries to compare.")

    values_a = []
    values_b = []
    for i, (x, y) in enumerate(zip(a, b)):
        values_a.append(_finite(x, f"a[{i}]"))
        values_b.append(_finite(y, f"b[{i}]"))

    best_i = 0
    best = abs(values_a[0] - values_b[0])
    for i in range(1, len(values_a)):
        diff = abs(values_a[i] - values_b[i])
        if diff > best:
            best = diff
            best_i = i

    return {
        "max_abs": best,
        "index": best_i,
        "a": values_a[best_i],
        "b": values_b[best_i],
        "n": len(values_a),
    }


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    if len(a) != len(b):
        raise AgreementError(f"length mismatch: {len(a)} != {len(b)}.")
    if not a:
        raise AgreementError("no entries to compare.")

    floor_value = _finite(floor, "floor")
    if floor_value <= 0.0:
        raise AgreementError(f"floor is {floor_value}, which is not positive.")

    values_a = []
    values_b = []
    for i, (x, y) in enumerate(zip(a, b)):
        values_a.append(_finite(x, f"a[{i}]"))
        values_b.append(_finite(y, f"b[{i}]"))

    best = 0.0
    best_i = 0
    floored = 0
    denom_min = float("inf")
    for i, b_val in enumerate(values_b):
        denom = abs(b_val)
        denom_min = min(denom_min, denom)
        if denom < floor_value:
            floored += 1
        rel = abs(values_a[i] - b_val) / max(denom, floor_value)
        if rel > best:
            best = rel
            best_i = i

    return {
        "max_rel": best,
        "index": best_i,
        "floored": floored,
        "denom_min": denom_min if denom_min != float("inf") else 0.0,
        "n": len(values_a),
    }


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    rows_a_list = list(rows_a)
    rows_b_list = list(rows_b)
    if not rows_a_list or not rows_b_list:
        raise AgreementError("no rows to compare.")
    if len(rows_a_list) != len(rows_b_list):
        raise AgreementError(f"row-count mismatch: {len(rows_a_list)} != {len(rows_b_list)}.")

    agree = 0
    first_disagreement = None
    n = len(rows_a_list)
    for row_index, (row_a, row_b) in enumerate(zip(rows_a_list, rows_b_list)):
        if not row_a or not row_b:
            raise AgreementError(f"row {row_index} is empty.")
        if len(row_a) != len(row_b):
            raise AgreementError(f"row {row_index} width mismatch: {len(row_a)} != {len(row_b)}.")
        for idx, (x, y) in enumerate(zip(row_a, row_b)):
            _finite(x, f"rows_a[{row_index}][{idx}]")
            _finite(y, f"rows_b[{row_index}][{idx}]")

        best_a = 0
        best_a_val = row_a[0]
        for i in range(1, len(row_a)):
            if row_a[i] > best_a_val:
                best_a_val = row_a[i]
                best_a = i

        best_b = 0
        best_b_val = row_b[0]
        for i in range(1, len(row_b)):
            if row_b[i] > best_b_val:
                best_b_val = row_b[i]
                best_b = i

        if best_a == best_b:
            agree += 1
        elif first_disagreement is None:
            first_disagreement = row_index

    return {
        "n": n,
        "agree": agree,
        "disagree": n - agree,
        "rate": agree / n,
        "first_disagreement": first_disagreement,
    }


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    rows_list = list(rows)
    if not rows_list:
        raise AgreementError("no rows to compare.")
    tol_value = _finite(tol, "tol")
    if tol_value < 0.0:
        raise AgreementError(f"tol is {tol_value}, which is negative.")

    margins = []
    for row_index, row in enumerate(rows_list):
        if len(row) < 2:
            raise AgreementError(f"row {row_index} has insufficient width for a margin.")
        for idx, value in enumerate(row):
            _finite(value, f"rows[{row_index}][{idx}]")
        sorted_row = sorted(row, reverse=True)
        margins.append(sorted_row[0] - sorted_row[1])

    min_margin = min(margins)
    argmin = min(range(len(margins)), key=lambda i: margins[i])
    median = _percentile(margins, 50)
    mean = sum(margins) / len(margins)
    below_tol = sum(1 for margin in margins if margin < tol_value)
    safe = below_tol == 0

    return {
        "n": len(margins),
        "min": min_margin,
        "median": median,
        "mean": mean,
        "argmin": argmin,
        "below_tol": below_tol,
        "safe": safe,
    }

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()
    