#!/usr/bin/env python3
import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import List

import config
from engine import MazeEngine


GRAPH_STYLE_NAMES = {
    0: "Sparse Random",
    1: "Dense Random",
    2: "Clusters",
    3: "Halin",
}


@dataclass
class TrialResult:
    index: int
    seeds: List[int]
    average: float
    total: float
    fails: int
    style_averages: List[float]
    style_fails: List[int]


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _summarize(values: List[float]) -> dict:
    ordered = sorted(values)
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    pop_std = statistics.pstdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * sample_std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    mad = statistics.fmean(abs(v - mean) for v in values)
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "range": ordered[-1] - ordered[0],
        "stddev": sample_std,
        "pop_stddev": pop_std,
        "mad": mad,
        "p10": _percentile(ordered, 0.10),
        "p25": _percentile(ordered, 0.25),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "cv": sample_std / mean if mean else 0.0,
        "ci95": ci95,
    }


def _fmt(value: float) -> str:
    return f"{value:,.1f}"


def _run_trial(
    engine: MazeEngine,
    seeds: List[int],
    bot,
    ghost,
    slots_style: int,
    trial_index: int,
) -> TrialResult:
    total = 0.0
    fails = 0
    style_totals = [0.0, 0.0, 0.0, 0.0]
    style_counts = [0, 0, 0, 0]
    style_fails = [0, 0, 0, 0]

    for seed in seeds:
        for style in range(4):
            try:
                coins = engine.grade(bot, ghost, style, slots_style, seed).coins
                total += coins
                style_totals[style] += coins
                style_counts[style] += 1
            except Exception:
                fails += 1
                style_fails[style] += 1

    games = sum(style_counts)
    style_averages = [
        style_totals[i] / style_counts[i] if style_counts[i] else float("nan")
        for i in range(4)
    ]

    return TrialResult(
        index=trial_index,
        seeds=seeds,
        average=total / games if games else 0.0,
        total=total,
        fails=fails,
        style_averages=style_averages,
        style_fails=style_fails,
    )


def _print_summary(title: str, values: List[float]) -> None:
    stats = _summarize(values)
    print(title)
    print(f"  mean:      {_fmt(stats['mean'])}")
    print(f"  median:    {_fmt(stats['median'])}")
    print(f"  min/max:   {_fmt(stats['min'])} / {_fmt(stats['max'])}")
    print(f"  range:     {_fmt(stats['range'])}")
    print(f"  std dev:   {_fmt(stats['stddev'])}")
    print(f"  mad:       {_fmt(stats['mad'])}")
    print(f"  p10/p90:   {_fmt(stats['p10'])} / {_fmt(stats['p90'])}")
    print(f"  p25/p75:   {_fmt(stats['p25'])} / {_fmt(stats['p75'])}")
    print(f"  cv:        {stats['cv'] * 100:.2f}%")
    print(
        f"  95% CI:    {_fmt(stats['mean'])} +/- {_fmt(stats['ci95'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated competition-style Maze evaluations and print "
            "reliability statistics."
        )
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="How many independent full-run batches to evaluate.",
    )
    parser.add_argument(
        "--seeds-per-trial",
        type=int,
        default=20,
        help="How many random seeds to use inside each batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible benchmark batches.",
    )
    parser.add_argument(
        "--slots-style",
        type=int,
        default=config.slots_style,
        help="Slot generation style to use. Defaults to config.slots_style.",
    )
    parser.add_argument(
        "--show-trials",
        action="store_true",
        help="Print every batch result, not just summary stats.",
    )
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 10**9)
    rng = random.Random(base_seed)
    engine = MazeEngine()
    results: List[TrialResult] = []

    for trial in range(1, args.trials + 1):
        seeds = [rng.randrange(1, 100_000_000) for _ in range(args.seeds_per_trial)]
        result = _run_trial(
            engine,
            seeds,
            config.bot,
            config.ghost,
            args.slots_style,
            trial,
        )
        results.append(result)

    averages = [result.average for result in results]
    total_fails = sum(result.fails for result in results)
    best = max(results, key=lambda result: result.average)
    worst = min(results, key=lambda result: result.average)

    print("Maze Benchmark Stats")
    print(f"Bot: {config.bot.__name__}")
    print(f"Ghost: {config.ghost.__name__}")
    print(f"Trials: {args.trials}")
    print(f"Seeds per trial: {args.seeds_per_trial}")
    print(f"Games per trial: {args.seeds_per_trial * 4}")
    print(f"Slots style: {args.slots_style}")
    print(f"Benchmark RNG seed: {base_seed}")
    print(f"Total failures: {total_fails}")
    print()

    _print_summary("Competition-Style Batch Average", averages)
    print()

    for style in range(4):
        style_values = [result.style_averages[style] for result in results]
        style_failures = sum(result.style_fails[style] for result in results)
        _print_summary(
            f"Graph Style {style} ({GRAPH_STYLE_NAMES[style]})",
            style_values,
        )
        print(f"  failures:   {style_failures}")
        print()

    print("Best Batch")
    print(f"  trial:      {best.index}")
    print(f"  average:    {_fmt(best.average)}")
    print(f"  seeds:      {best.seeds}")
    print(f"  per-style:  {[round(value, 1) for value in best.style_averages]}")
    print()

    print("Worst Batch")
    print(f"  trial:      {worst.index}")
    print(f"  average:    {_fmt(worst.average)}")
    print(f"  seeds:      {worst.seeds}")
    print(f"  per-style:  {[round(value, 1) for value in worst.style_averages]}")
    print()

    print("Spread")
    print(f"  best - worst: {_fmt(best.average - worst.average)}")
    print(f"  best / worst: {best.average / worst.average:.3f}x" if worst.average else "  best / worst: inf")

    if args.show_trials:
        print()
        print("Per-Trial Results")
        for result in results:
            print(
                f"  trial {result.index:>2}: avg={_fmt(result.average)} "
                f"fails={result.fails} "
                f"styles={[round(value, 1) for value in result.style_averages]}"
            )


if __name__ == "__main__":
    main()
