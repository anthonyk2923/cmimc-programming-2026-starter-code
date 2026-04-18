#!/usr/bin/env python3
"""Random-search tuner for the Lemon Tycoon submission strategy."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Type

from benchmark import benchmark_player
from submission import SubmissionStrategy


SEARCH_SPACE = {
    "initial_soft_cap": [3, 4, 5],
    "war_soft_cap": [1, 2, 3],
    "crowd_penalty": [0.12, 0.15, 0.18, 0.21, 0.24],
    "threat_penalty": [1.2, 1.45, 1.65, 1.9, 2.2],
    "cooldown_penalty": [1.4, 1.7, 1.95, 2.2],
    "own_stack_penalty": [0.16, 0.20, 0.24, 0.28],
    "safe_bonus": [0.16, 0.20, 0.24, 0.30],
    "emergency_top_bonus": [0.35, 0.50, 0.65, 0.80],
    "sabotage_threshold_calm": [24.0, 28.0, 32.0, 36.0],
    "sabotage_threshold_danger": [8.0, 12.0, 14.0, 18.0],
    "sabotage_threshold_emergency": [-2.0, 0.0, 2.0, 4.0],
    "min_sabotage_round": [6, 7, 8, 9],
    "early_sabotage_gap": [220.0, 260.0, 320.0],
    "active_leader_floor": [320.0, 420.0, 520.0],
    "leader_advantage_floor": [100.0, 140.0, 180.0],
    "mid_sabotage_gap": [180.0, 220.0, 280.0],
    "small_bankroll_skip_mult": [1.0, 1.5, 2.0, 2.5],
    "small_bankroll_gap": [100.0, 140.0, 180.0],
    "double_sabotage_gap": [80.0, 100.0, 130.0],
    "double_sabotage_lead": [140.0, 170.0, 210.0],
    "double_sabotage_second_margin": [-6.0, -4.0, -1.0, 2.0],
}


def sample_config(rng: random.Random) -> Dict[str, float]:
    cfg = dict(SubmissionStrategy.CONFIG)
    for key, values in SEARCH_SPACE.items():
        cfg[key] = rng.choice(values)
    return cfg


def make_candidate(cfg: Dict[str, float], name: str) -> Type[SubmissionStrategy]:
    return type(name, (SubmissionStrategy,), {"CONFIG": cfg})


def score(stats: Dict[str, float]) -> Tuple[float, float, float]:
    # Prioritize winning and placement, then use lemons as a weaker tiebreak.
    return (-stats["win_rate"], stats["avg_rank"], -stats["avg_lemons"])


def main() -> None:
    rng = random.Random(20260418)
    coarse_rounds = 36
    finalists = 6
    fine_repetitions = 6

    tested: List[Tuple[Tuple[float, float, float], Dict[str, float], Dict[str, float]]] = []

    base_stats = benchmark_player(SubmissionStrategy, 2)
    tested.append((score(base_stats), dict(SubmissionStrategy.CONFIG), base_stats))
    print("Base strategy:", base_stats)

    for idx in range(coarse_rounds):
        cfg = sample_config(rng)
        candidate = make_candidate(cfg, f"TunedSubmission{idx}")
        stats = benchmark_player(candidate, 1)
        tested.append((score(stats), cfg, stats))

    tested.sort(key=lambda item: item[0])
    shortlisted = tested[:finalists]

    rescored: List[Tuple[Tuple[float, float, float], Dict[str, float], Dict[str, float]]] = []
    for _, cfg, _ in shortlisted:
        candidate = make_candidate(cfg, "FinalistSubmission")
        stats = benchmark_player(candidate, fine_repetitions)
        rescored.append((score(stats), cfg, stats))
        print("Finalist:", stats, cfg)

    rescored.sort(key=lambda item: item[0])
    best_score, best_cfg, best_stats = rescored[0]
    print("\nBest stats:", best_stats)
    print("Best config:")
    for key in sorted(best_cfg):
        if best_cfg[key] != SubmissionStrategy.CONFIG.get(key):
            print(f"  {key}: {best_cfg[key]}")


if __name__ == "__main__":
    main()
