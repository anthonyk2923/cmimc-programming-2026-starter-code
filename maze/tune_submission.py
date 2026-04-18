#!/usr/bin/env python3
"""Parameter sweep tool for constants in submission.py.

Usage examples:
  python tune_submission.py
  python tune_submission.py --params BASE_THRESHOLD MIN_THRESHOLD
  python tune_submission.py --param BASE_THRESHOLD=30,40,50,60
  python tune_submission.py --mode grid --param BASE_THRESHOLD=40,50 --param MIN_THRESHOLD=8,12
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from engine import MazeEngine
import submission


DEFAULT_SWEEP_VALUES: Dict[str, List[int]] = {
    "BASE_THRESHOLD": [30, 40, 50, 60, 70],
    "MIN_THRESHOLD": [4, 8, 10, 12, 16],
    "BOT_FALLBACK_STEP": [140, 180, 220, 260, 320],
    "BOT_FALLBACK_DEPTH": [8, 10, 13, 16, 20],
    "GHOST_GOOD_SLOT": [16, 20, 25, 30, 36],
    "GHOST_LATE_STEP": [120, 160, 190, 220, 260],
    "GHOST_DEEP_EXTRA": [0, 1, 2, 3, 4],
}


@dataclass
class EvalResult:
    assignment: Dict[str, int]
    mean: float
    stdev: float
    minimum: float
    maximum: float
    per_style_mean: Tuple[float, float, float, float]
    failures: int


def _parse_param_override(raw: str) -> Tuple[str, List[int]]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --param value '{raw}'. Use NAME=v1,v2,v3 format."
        )
    name, values_raw = raw.split("=", 1)
    name = name.strip()
    if name not in DEFAULT_SWEEP_VALUES:
        valid = ", ".join(sorted(DEFAULT_SWEEP_VALUES))
        raise ValueError(f"Unknown parameter '{name}'. Valid options: {valid}")
    values = []
    for token in values_raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"No values provided for parameter '{name}'.")
    return name, values


def _current_defaults() -> Dict[str, int]:
    return {name: int(getattr(submission, name)) for name in DEFAULT_SWEEP_VALUES}


def _set_constants(values: Dict[str, int]) -> None:
    for name, value in values.items():
        setattr(submission, name, int(value))


def _run_scores(
    engine: MazeEngine,
    seeds: Sequence[int],
    slots_style: int,
) -> Tuple[List[float], List[List[float]], int]:
    scores: List[float] = []
    style_scores: List[List[float]] = [[], [], [], []]
    failures = 0
    for seed in seeds:
        for style in range(4):
            try:
                coins = engine.grade(
                    submission.SubmissionBot,
                    submission.SubmissionGhost,
                    style,
                    slots_style,
                    seed,
                ).coins
                scores.append(float(coins))
                style_scores[style].append(float(coins))
            except Exception:
                failures += 1
    return scores, style_scores, failures


def _summarize(
    assignment: Dict[str, int],
    scores: List[float],
    style_scores: List[List[float]],
    failures: int,
) -> EvalResult:
    mean = statistics.fmean(scores) if scores else 0.0
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    minimum = min(scores) if scores else 0.0
    maximum = max(scores) if scores else 0.0
    per_style_mean = tuple(
        statistics.fmean(v) if v else float("nan") for v in style_scores
    )
    return EvalResult(
        assignment=dict(assignment),
        mean=mean,
        stdev=stdev,
        minimum=minimum,
        maximum=maximum,
        per_style_mean=per_style_mean,
        failures=failures,
    )


def _product_size(grid: Dict[str, Sequence[int]]) -> int:
    result = 1
    for values in grid.values():
        result *= len(values)
    return result


def _iter_grid(grid: Dict[str, Sequence[int]]) -> Iterable[Dict[str, int]]:
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield {k: int(v) for k, v in zip(keys, combo)}


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:,.2f}"


def _describe_assignment(assignment: Dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in assignment.items())


def _print_sweep_table(param: str, results: Sequence[EvalResult], baseline: EvalResult) -> None:
    print()
    print(f"=== Sweep: {param} ===")
    print(
        "value | mean | delta_vs_base | stdev | min | max | "
        "style0 | style1 | style2 | style3 | fails"
    )
    for result in sorted(results, key=lambda r: r.mean, reverse=True):
        value = result.assignment[param]
        delta = result.mean - baseline.mean
        print(
            f"{value:>5} | {_fmt(result.mean):>8} | {delta:>+13.2f} | "
            f"{_fmt(result.stdev):>6} | {_fmt(result.minimum):>6} | {_fmt(result.maximum):>6} | "
            f"{_fmt(result.per_style_mean[0]):>6} | {_fmt(result.per_style_mean[1]):>6} | "
            f"{_fmt(result.per_style_mean[2]):>6} | {_fmt(result.per_style_mean[3]):>6} | "
            f"{result.failures:>5}"
        )


def _print_grid_top(results: Sequence[EvalResult], baseline: EvalResult, top_k: int) -> None:
    print()
    print("=== Grid Search Top Results ===")
    print("rank | mean | delta_vs_base | stdev | fails | constants")
    for i, result in enumerate(sorted(results, key=lambda r: r.mean, reverse=True)[:top_k], start=1):
        delta = result.mean - baseline.mean
        print(
            f"{i:>4} | {_fmt(result.mean):>8} | {delta:>+13.2f} | "
            f"{_fmt(result.stdev):>6} | {result.failures:>5} | {_describe_assignment(result.assignment)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune constants in submission.py by sweeping values and benchmarking "
            "average coins across seeds and graph styles."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "grid"],
        default="sweep",
        help="sweep = one parameter at a time, grid = cartesian product.",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        choices=sorted(DEFAULT_SWEEP_VALUES.keys()),
        help="Only tune these parameters. Defaults to all known parameters.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override sweep values using NAME=v1,v2,v3 (can be repeated).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Random seeds per candidate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible candidate seed generation.",
    )
    parser.add_argument(
        "--slots-style",
        type=int,
        default=1,
        choices=[1, 2],
        help="Slot generation style passed to MazeEngine.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top results to print in grid mode.",
    )
    args = parser.parse_args()

    rng_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 10**9)
    rng = random.Random(rng_seed)
    seeds = [rng.randrange(1, 1_000_000_000) for _ in range(args.seeds)]

    defaults = _current_defaults()
    chosen_params = list(args.params) if args.params else list(
        DEFAULT_SWEEP_VALUES.keys())

    sweep_values: Dict[str, List[int]] = {
        name: list(DEFAULT_SWEEP_VALUES[name])
        for name in chosen_params
    }
    for raw in args.param:
        name, values = _parse_param_override(raw)
        if name in sweep_values:
            sweep_values[name] = values

    engine = MazeEngine()

    # Baseline first.
    _set_constants(defaults)
    baseline_scores, baseline_style_scores, baseline_failures = _run_scores(
        engine,
        seeds,
        args.slots_style,
    )
    baseline = _summarize(defaults, baseline_scores,
                          baseline_style_scores, baseline_failures)

    print("Maze Submission Constant Tuner")
    print(f"Mode: {args.mode}")
    print(f"Seeds per candidate: {args.seeds}")
    print("Graph styles per seed: 4")
    print(f"Games per candidate: {args.seeds * 4}")
    print(f"Slot style: {args.slots_style}")
    print(f"Random seed: {rng_seed}")
    print("Baseline constants:")
    print(f"  {_describe_assignment(defaults)}")
    print(
        f"Baseline mean={_fmt(baseline.mean)} stdev={_fmt(baseline.stdev)} "
        f"fails={baseline.failures}"
    )

    try:
        if args.mode == "sweep":
            for param in chosen_params:
                results: List[EvalResult] = []
                for value in sweep_values[param]:
                    assignment = dict(defaults)
                    assignment[param] = int(value)
                    _set_constants(assignment)
                    scores, style_scores, failures = _run_scores(
                        engine,
                        seeds,
                        args.slots_style,
                    )
                    results.append(_summarize(
                        assignment, scores, style_scores, failures))
                _print_sweep_table(param, results, baseline)

        else:
            grid = {k: sweep_values[k] for k in chosen_params}
            candidates = _product_size(grid)
            print()
            print(
                f"Grid candidates: {candidates} "
                f"({args.seeds * 4} games each, total {candidates * args.seeds * 4} games)"
            )
            results: List[EvalResult] = []
            for idx, assignment in enumerate(_iter_grid(grid), start=1):
                full_assignment = dict(defaults)
                full_assignment.update(assignment)
                _set_constants(full_assignment)
                scores, style_scores, failures = _run_scores(
                    engine,
                    seeds,
                    args.slots_style,
                )
                results.append(
                    _summarize(full_assignment, scores, style_scores, failures)
                )
                if idx % 10 == 0 or idx == candidates:
                    print(f"  completed {idx}/{candidates}")

            _print_grid_top(results, baseline, args.top)

    finally:
        _set_constants(defaults)


if __name__ == "__main__":
    main()
