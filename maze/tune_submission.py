#!/usr/bin/env python3
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


STYLE_NAMES = ["SPARSE", "DENSE", "CLUSTER", "HALIN"]
STYLE_PARAM_NAMES = [
    "BOT_WAIT",
    "BOT_TARGET_DEPTH",
    "BOT_STOP_THRESHOLD",
    "BOT_LATE_STEP",
    "BOT_LATE_THRESHOLD",
    "GHOST_TARGET_DEPTH",
    "GHOST_GOOD_SLOT",
    "GHOST_DEEP_EXTRA",
    "GHOST_DEEP_STOP",
    "GHOST_LATE_STEP",
]
SCALAR_PARAM_NAMES = ["GHOST_SAMPLE_LATE_ONLY"]


DEFAULT_SWEEP_VALUES: Dict[str, List[int | bool]] = {
    "BOT_WAIT_0": [0, 2, 4, 6, 8, 12],
    "BOT_WAIT_1": [4, 8, 12],
    "BOT_WAIT_2": [4, 8, 12],
    "BOT_WAIT_3": [8, 12, 16, 20],
    "BOT_TARGET_DEPTH_0": [10, 12, 14, 16],
    "BOT_TARGET_DEPTH_1": [4, 5, 6, 7],
    "BOT_TARGET_DEPTH_2": [4, 5, 6, 7, 8],
    "BOT_TARGET_DEPTH_3": [8, 10, 12, 14],
    "BOT_STOP_THRESHOLD_0": [35, 40, 45, 50, 60],
    "BOT_STOP_THRESHOLD_1": [20, 25, 30, 40, 50],
    "BOT_STOP_THRESHOLD_2": [20, 25, 30, 40, 50],
    "BOT_STOP_THRESHOLD_3": [20, 25, 30, 40, 50],
    "BOT_LATE_STEP_0": [160, 200, 220, 260, 320],
    "BOT_LATE_STEP_1": [120, 160, 220],
    "BOT_LATE_STEP_2": [120, 160, 220],
    "BOT_LATE_STEP_3": [120, 160, 220],
    "BOT_LATE_THRESHOLD_0": [2, 3, 4, 5],
    "BOT_LATE_THRESHOLD_1": [2, 3, 4, 5],
    "BOT_LATE_THRESHOLD_2": [2, 3, 4, 5],
    "BOT_LATE_THRESHOLD_3": [2, 3, 4, 5],
    "GHOST_TARGET_DEPTH_0": [10, 12, 14, 16],
    "GHOST_TARGET_DEPTH_1": [4, 5, 6, 7],
    "GHOST_TARGET_DEPTH_2": [4, 5, 6, 7, 8],
    "GHOST_TARGET_DEPTH_3": [6, 8, 10, 12],
    "GHOST_GOOD_SLOT_0": [20, 25, 30, 35],
    "GHOST_GOOD_SLOT_1": [15, 20, 25, 30],
    "GHOST_GOOD_SLOT_2": [15, 20, 25, 30],
    "GHOST_GOOD_SLOT_3": [15, 20, 25, 30],
    "GHOST_DEEP_EXTRA_0": [1, 2, 3, 4],
    "GHOST_DEEP_EXTRA_1": [1, 2, 3, 4],
    "GHOST_DEEP_EXTRA_2": [1, 2, 3, 4],
    "GHOST_DEEP_EXTRA_3": [1, 2, 3, 4],
    "GHOST_DEEP_STOP_0": [1, 3, 5, 7, 10],
    "GHOST_DEEP_STOP_1": [1, 3, 5, 7, 10],
    "GHOST_DEEP_STOP_2": [1, 3, 5, 7, 10],
    "GHOST_DEEP_STOP_3": [1, 3, 5, 7, 10],
    "GHOST_LATE_STEP_0": [60, 80, 100, 120, 140],
    "GHOST_LATE_STEP_1": [40, 50, 60, 70, 80],
    "GHOST_LATE_STEP_2": [20, 25, 30, 35, 40],
    "GHOST_LATE_STEP_3": [40, 50, 60, 70, 80],
    "GHOST_SAMPLE_LATE_ONLY": [False, True],
}


@dataclass
class EvalResult:
    assignment: Dict[str, int | bool]
    mean: float
    stdev: float
    minimum: float
    maximum: float
    per_style_mean: Tuple[float, float, float, float]
    failures: int


def _all_param_names() -> List[str]:
    names: List[str] = []
    for base in STYLE_PARAM_NAMES:
        for style_idx in range(4):
            names.append(f"{base}_{style_idx}")
    names.extend(SCALAR_PARAM_NAMES)
    return names


ALL_PARAM_NAMES = _all_param_names()


def _current_defaults() -> Dict[str, int | bool]:
    values: Dict[str, int | bool] = {}
    for base in STYLE_PARAM_NAMES:
        tup = getattr(submission, base)
        for style_idx in range(4):
            values[f"{base}_{style_idx}"] = tup[style_idx]
    for base in SCALAR_PARAM_NAMES:
        values[base] = getattr(submission, base)
    return values


def _set_constants(values: Dict[str, int | bool]) -> None:
    grouped: Dict[str, List[int]] = {}
    for base in STYLE_PARAM_NAMES:
        grouped[base] = list(getattr(submission, base))
    for name, value in values.items():
        if name in SCALAR_PARAM_NAMES:
            setattr(submission, name, value)
            continue
        base, style_idx_raw = name.rsplit("_", 1)
        grouped[base][int(style_idx_raw)] = int(value)
    for base, items in grouped.items():
        setattr(submission, base, tuple(items))


def _parse_param_override(raw: str) -> Tuple[str, List[int | bool]]:
    if "=" not in raw:
        raise ValueError(f"Invalid --param value '{raw}'. Use NAME=v1,v2,v3 format.")
    name, values_raw = raw.split("=", 1)
    name = name.strip()
    if name not in DEFAULT_SWEEP_VALUES:
        valid = ", ".join(sorted(DEFAULT_SWEEP_VALUES))
        raise ValueError(f"Unknown parameter '{name}'. Valid options: {valid}")
    values: List[int | bool] = []
    for token in values_raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "true":
            values.append(True)
        elif token.lower() == "false":
            values.append(False)
        else:
            values.append(int(token))
    if not values:
        raise ValueError(f"No values provided for parameter '{name}'.")
    return name, values


def _run_scores(engine: MazeEngine, seeds: Sequence[int], slots_style: int) -> Tuple[List[float], List[List[float]], int]:
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
    assignment: Dict[str, int | bool],
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


def _iter_grid(grid: Dict[str, Sequence[int | bool]]) -> Iterable[Dict[str, int | bool]]:
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield {k: v for k, v in zip(keys, combo)}


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:,.2f}"


def _describe_assignment(assignment: Dict[str, int | bool]) -> str:
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
            f"{str(value):>5} | {_fmt(result.mean):>8} | {delta:>+13.2f} | "
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


def _local_search(
    engine: MazeEngine,
    defaults: Dict[str, int | bool],
    sweep_values: Dict[str, List[int | bool]],
    seeds: Sequence[int],
    slots_style: int,
    iterations: int,
    rng_seed: int,
) -> EvalResult:
    rng = random.Random(rng_seed)
    current = dict(defaults)
    _set_constants(current)
    scores, style_scores, failures = _run_scores(engine, seeds, slots_style)
    best = _summarize(current, scores, style_scores, failures)
    frontier: List[EvalResult] = [best]
    seen = {tuple(sorted(current.items()))}
    print()
    print("=== Local Search ===")
    print(f"start mean={_fmt(best.mean)}")

    param_names = list(sweep_values.keys())
    for i in range(iterations):
        parent = rng.choice(frontier).assignment
        candidate = dict(parent)
        for _ in range(rng.randint(1, min(4, len(param_names)))):
            name = rng.choice(param_names)
            candidate[name] = rng.choice(sweep_values[name])
        signature = tuple(sorted(candidate.items()))
        if signature in seen:
            continue
        seen.add(signature)
        _set_constants(candidate)
        scores, style_scores, failures = _run_scores(engine, seeds, slots_style)
        result = _summarize(candidate, scores, style_scores, failures)
        if result.mean > best.mean:
            best = result
            frontier.append(result)
            frontier = sorted(frontier, key=lambda r: r.mean, reverse=True)[:20]
            print(
                f"improved {i:>4}: mean={_fmt(result.mean)} "
                f"styles={[round(x, 2) for x in result.per_style_mean]} "
                f"{_describe_assignment(result.assignment)}"
            )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the current style-specific submission parameters."
    )
    parser.add_argument("--mode", choices=["sweep", "grid", "search"], default="sweep")
    parser.add_argument("--params", nargs="*", choices=sorted(ALL_PARAM_NAMES))
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--slots-style", type=int, default=1, choices=[1, 2])
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--fixed-seeds", action="store_true", help="Use seeds 0..N-1 instead of random seeds.")
    args = parser.parse_args()

    defaults = _current_defaults()
    chosen_params = list(args.params) if args.params else list(DEFAULT_SWEEP_VALUES.keys())

    sweep_values: Dict[str, List[int | bool]] = {
        name: list(DEFAULT_SWEEP_VALUES[name])
        for name in chosen_params
    }
    for raw in args.param:
        name, values = _parse_param_override(raw)
        if name in sweep_values:
            sweep_values[name] = values

    if args.fixed_seeds:
        seeds = list(range(args.seeds))
    else:
        rng = random.Random(args.seed)
        seeds = [rng.randrange(1, 1_000_000_000) for _ in range(args.seeds)]

    engine = MazeEngine()
    _set_constants(defaults)
    baseline_scores, baseline_style_scores, baseline_failures = _run_scores(engine, seeds, args.slots_style)
    baseline = _summarize(defaults, baseline_scores, baseline_style_scores, baseline_failures)

    print("Maze Tuner")
    print(f"Mode: {args.mode}")
    print(f"Seeds: {seeds}")
    print(f"Slots style: {args.slots_style}")
    print("Baseline:")
    print(f"  mean={_fmt(baseline.mean)} stdev={_fmt(baseline.stdev)} fails={baseline.failures}")
    print(f"  styles={[round(x, 2) for x in baseline.per_style_mean]}")

    try:
        if args.mode == "sweep":
            for param in chosen_params:
                results: List[EvalResult] = []
                for value in sweep_values[param]:
                    candidate = dict(defaults)
                    candidate[param] = value
                    _set_constants(candidate)
                    scores, style_scores, failures = _run_scores(engine, seeds, args.slots_style)
                    results.append(_summarize(candidate, scores, style_scores, failures))
                _print_sweep_table(param, results, baseline)
        elif args.mode == "grid":
            results: List[EvalResult] = []
            for candidate in _iter_grid(sweep_values):
                merged = dict(defaults)
                merged.update(candidate)
                _set_constants(merged)
                scores, style_scores, failures = _run_scores(engine, seeds, args.slots_style)
                results.append(_summarize(merged, scores, style_scores, failures))
            _print_grid_top(results, baseline, args.top)
        else:
            best = _local_search(
                engine=engine,
                defaults=defaults,
                sweep_values=sweep_values,
                seeds=seeds,
                slots_style=args.slots_style,
                iterations=args.iterations,
                rng_seed=args.seed,
            )
            print()
            print("=== Best Found ===")
            print(f"mean={_fmt(best.mean)} delta={best.mean - baseline.mean:+.2f}")
            print(f"styles={[round(x, 2) for x in best.per_style_mean]}")
            print(_describe_assignment(best.assignment))
    finally:
        _set_constants(defaults)


if __name__ == "__main__":
    main()
