#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Dict, List, Sequence, Tuple

from engine import MazeEngine


SEEDS = list(range(20))

STYLE_SPARSE = 0
STYLE_DENSE = 1
STYLE_CLUSTER = 2
STYLE_HALIN = 3


def _bit_get(blob: bytearray, idx: int) -> int:
    return (blob[idx >> 3] >> (idx & 7)) & 1


def _bit_set(blob: bytearray, idx: int) -> None:
    blob[idx >> 3] |= 1 << (idx & 7)


def _classify(root_degree: int, dense_cutoff: int, cluster_cutoff: int) -> int:
    if root_degree <= 2:
        return STYLE_SPARSE
    if root_degree >= dense_cutoff:
        return STYLE_DENSE
    if root_degree >= cluster_cutoff:
        return STYLE_CLUSTER
    return STYLE_HALIN


def _order_neighbors(
    neighbors: List[int],
    mode: int,
    pos: int,
    step: int,
) -> List[int]:
    if mode == 0:
        return list(neighbors)
    if mode == 1:
        return sorted(neighbors)
    if mode == 2:
        return sorted(neighbors, reverse=True)
    if mode == 3:
        pivot = (pos * 31 + step * 17) % 100
        return sorted(neighbors, key=lambda n: ((n - pivot) % 100, n))
    pivot = (pos * 17 + step * 31) % 100
    return sorted(neighbors, key=lambda n: ((pivot - n) % 100, n))


@dataclass(frozen=True)
class Params:
    dense_cutoff: int
    cluster_cutoff: int
    bot_wait: Tuple[int, int, int, int]
    bot_depth: Tuple[int, int, int, int]
    bot_stop: Tuple[int, int, int, int]
    bot_late_step: Tuple[int, int, int, int]
    bot_late_stop: Tuple[int, int, int, int]
    ghost_depth: Tuple[int, int, int, int]
    ghost_good: Tuple[int, int, int, int]
    ghost_deep_extra: Tuple[int, int, int, int]
    ghost_deep_stop: Tuple[int, int, int, int]
    ghost_late_step: Tuple[int, int, int, int]
    ghost_sample_late_only: bool
    bot_order: int
    ghost_order: int


def _baseline_params() -> Params:
    return Params(
        dense_cutoff=8,
        cluster_cutoff=5,
        bot_wait=(28, 8, 8, 16),
        bot_depth=(12, 5, 5, 8),
        bot_stop=(50, 50, 50, 50),
        bot_late_step=(220, 220, 220, 220),
        bot_late_stop=(3, 3, 3, 3),
        ghost_depth=(12, 5, 5, 8),
        ghost_good=(25, 25, 25, 25),
        ghost_deep_extra=(2, 2, 2, 2),
        ghost_deep_stop=(5, 5, 5, 5),
        ghost_late_step=(60, 60, 60, 60),
        ghost_sample_late_only=True,
        bot_order=0,
        ghost_order=0,
    )


def build_players(params: Params) -> Tuple[Callable[..., Tuple[int, Any]], Callable[..., Tuple[int, Any]]]:
    def bot(
        step: int,
        total_steps: int,
        pos: int,
        last_pos: int,
        neighbors: List[int],
        has_slot: bool,
        slot_coins: int,
        data: Any,
    ) -> Tuple[int, Any]:
        if step == 1:
            last_pos = -1

        state = bytearray(17) if data is None else bytearray(data)
        if data is None:
            style = _classify(len(neighbors), params.dense_cutoff, params.cluster_cutoff)
            state[15] = params.bot_wait[style]
            state[16] = style

        style = state[16]
        _bit_set(state, pos)
        depth = state[13]
        mode = state[14]

        if mode == 2:
            return -1, bytes(state)

        if mode == 0 and state[15] > 0:
            state[15] -= 1
            return -1, bytes(state)

        state[14] = 1

        if has_slot and (
            slot_coins >= params.bot_stop[style]
            or (depth >= params.bot_depth[style] and slot_coins >= 8)
            or (step >= params.bot_late_step[style] and slot_coins >= params.bot_late_stop[style])
        ):
            state[14] = 2
            return -1, bytes(state)

        ordered = _order_neighbors(neighbors, params.bot_order, pos, step)
        for n in ordered:
            if not _bit_get(state, n):
                state[13] = depth + 1 if depth < 255 else 255
                return n, bytes(state)

        for n in ordered:
            if n != last_pos:
                return n, bytes(state)

        if last_pos != -1 and last_pos in neighbors:
            state[13] = depth - 1 if depth else 0
            return last_pos, bytes(state)

        return -1, bytes(state)

    def ghost(
        step: int,
        total_steps: int,
        pos: int,
        last_pos: int,
        neighbors: List[int],
        has_slot: bool,
        slot_coins: int,
        data: Any,
    ) -> Tuple[int, Any]:
        if step == 1:
            last_pos = -1

        state = bytearray(21) if data is None else bytearray(data)
        if data is None:
            style = _classify(len(neighbors), params.dense_cutoff, params.cluster_cutoff)
            state[18] = params.ghost_depth[style]
            state[19] = style
            state[20] = 0

        _bit_set(state, pos)
        depth = state[16]
        mode = state[17]
        style = state[19]
        target_depth = state[18]

        if mode == 1:
            return -1, bytes(state)

        if mode == 2:
            if slot_coins > 0 or step >= params.ghost_late_step[style] + 120:
                state[17] = 1
                return -1, bytes(state)
            state[17] = 0

        if has_slot and depth >= target_depth and slot_coins >= params.ghost_good[style]:
            state[17] = 1
            return -1, bytes(state)

        if has_slot and depth >= target_depth + params.ghost_deep_extra[style] and slot_coins >= params.ghost_deep_stop[style]:
            state[17] = 1
            return -1, bytes(state)

        if has_slot and not params.ghost_sample_late_only and depth >= target_depth and slot_coins == 0:
            state[17] = 2
            return -1, bytes(state)

        if has_slot and step >= params.ghost_late_step[style]:
            if slot_coins > 0:
                state[17] = 1
            else:
                state[17] = 2
            return -1, bytes(state)

        ordered = _order_neighbors(neighbors, params.ghost_order, pos, step)
        for n in ordered:
            if not _bit_get(state, n):
                state[16] = depth + 1 if depth < 255 else 255
                return n, bytes(state)

        for n in ordered:
            if n != last_pos:
                return n, bytes(state)

        if last_pos != -1 and last_pos in neighbors:
            state[16] = depth - 1 if depth else 0
            return last_pos, bytes(state)

        return -1, bytes(state)

    return bot, ghost


def evaluate(engine: MazeEngine, params: Params) -> Tuple[float, Tuple[float, float, float, float]]:
    bot, ghost = build_players(params)
    style_scores = [[], [], [], []]
    for style in range(4):
        for seed in SEEDS:
            style_scores[style].append(engine.grade(bot, ghost, style, 1, seed).coins)
    per_style = tuple(fmean(values) for values in style_scores)
    overall = fmean([score for values in style_scores for score in values])
    return overall, per_style


def mutate_tuple(rng: random.Random, values: Tuple[int, ...], candidates: Sequence[int], idxs: Sequence[int]) -> Tuple[int, ...]:
    items = list(values)
    for idx in idxs:
        items[idx] = rng.choice(candidates)
    return tuple(items)


def mutate(rng: random.Random, params: Params) -> Params:
    choice = rng.randrange(10)
    if choice == 0:
        return Params(
            dense_cutoff=rng.choice([7, 8, 9, 10]),
            cluster_cutoff=rng.choice([4, 5, 6]),
            bot_wait=params.bot_wait,
            bot_depth=params.bot_depth,
            bot_stop=params.bot_stop,
            bot_late_step=params.bot_late_step,
            bot_late_stop=params.bot_late_stop,
            ghost_depth=params.ghost_depth,
            ghost_good=params.ghost_good,
            ghost_deep_extra=params.ghost_deep_extra,
            ghost_deep_stop=params.ghost_deep_stop,
            ghost_late_step=params.ghost_late_step,
            ghost_sample_late_only=params.ghost_sample_late_only,
            bot_order=params.bot_order,
            ghost_order=params.ghost_order,
        )
    if choice == 1:
        return Params(
            dense_cutoff=params.dense_cutoff,
            cluster_cutoff=params.cluster_cutoff,
            bot_wait=mutate_tuple(rng, params.bot_wait, [0, 4, 8, 12, 16, 24, 32], [rng.randrange(4)]),
            bot_depth=params.bot_depth,
            bot_stop=params.bot_stop,
            bot_late_step=params.bot_late_step,
            bot_late_stop=params.bot_late_stop,
            ghost_depth=params.ghost_depth,
            ghost_good=params.ghost_good,
            ghost_deep_extra=params.ghost_deep_extra,
            ghost_deep_stop=params.ghost_deep_stop,
            ghost_late_step=params.ghost_late_step,
            ghost_sample_late_only=params.ghost_sample_late_only,
            bot_order=params.bot_order,
            ghost_order=params.ghost_order,
        )
    if choice == 2:
        return Params(
            dense_cutoff=params.dense_cutoff,
            cluster_cutoff=params.cluster_cutoff,
            bot_wait=params.bot_wait,
            bot_depth=mutate_tuple(rng, params.bot_depth, [4, 5, 6, 8, 10, 12, 14], [rng.randrange(4)]),
            bot_stop=params.bot_stop,
            bot_late_step=params.bot_late_step,
            bot_late_stop=params.bot_late_stop,
            ghost_depth=params.ghost_depth,
            ghost_good=params.ghost_good,
            ghost_deep_extra=params.ghost_deep_extra,
            ghost_deep_stop=params.ghost_deep_stop,
            ghost_late_step=params.ghost_late_step,
            ghost_sample_late_only=params.ghost_sample_late_only,
            bot_order=params.bot_order,
            ghost_order=params.ghost_order,
        )
    if choice == 3:
        return Params(
            dense_cutoff=params.dense_cutoff,
            cluster_cutoff=params.cluster_cutoff,
            bot_wait=params.bot_wait,
            bot_depth=params.bot_depth,
            bot_stop=mutate_tuple(rng, params.bot_stop, [15, 20, 25, 30, 35, 40, 50], [rng.randrange(4)]),
            bot_late_step=params.bot_late_step,
            bot_late_stop=params.bot_late_stop,
            ghost_depth=params.ghost_depth,
            ghost_good=params.ghost_good,
            ghost_deep_extra=params.ghost_deep_extra,
            ghost_deep_stop=params.ghost_deep_stop,
            ghost_late_step=params.ghost_late_step,
            ghost_sample_late_only=params.ghost_sample_late_only,
            bot_order=params.bot_order,
            ghost_order=params.ghost_order,
        )
    if choice == 4:
        return Params(
            dense_cutoff=params.dense_cutoff,
            cluster_cutoff=params.cluster_cutoff,
            bot_wait=params.bot_wait,
            bot_depth=params.bot_depth,
            bot_stop=params.bot_stop,
            bot_late_step=mutate_tuple(rng, params.bot_late_step, [120, 160, 200, 220, 260, 320], [rng.randrange(4)]),
            bot_late_stop=mutate_tuple(rng, params.bot_late_stop, [2, 3, 4, 5, 6], [rng.randrange(4)]),
            ghost_depth=params.ghost_depth,
            ghost_good=params.ghost_good,
            ghost_deep_extra=params.ghost_deep_extra,
            ghost_deep_stop=params.ghost_deep_stop,
            ghost_late_step=params.ghost_late_step,
            ghost_sample_late_only=params.ghost_sample_late_only,
            bot_order=params.bot_order,
            ghost_order=params.ghost_order,
        )
    if choice == 5:
        return Params(
            dense_cutoff=params.dense_cutoff,
            cluster_cutoff=params.cluster_cutoff,
            bot_wait=params.bot_wait,
            bot_depth=params.bot_depth,
            bot_stop=params.bot_stop,
            bot_late_step=params.bot_late_step,
            bot_late_stop=params.bot_late_stop,
            ghost_depth=mutate_tuple(rng, params.ghost_depth, [3, 4, 5, 6, 8, 10, 12, 14], [rng.randrange(4)]),
            ghost_good=params.ghost_good,
            ghost_deep_extra=params.ghost_deep_extra,
            ghost_deep_stop=params.ghost_deep_stop,
            ghost_late_step=params.ghost_late_step,
            ghost_sample_late_only=params.ghost_sample_late_only,
            bot_order=params.bot_order,
            ghost_order=params.ghost_order,
        )
    if choice == 6:
        return Params(
            dense_cutoff=params.dense_cutoff,
            cluster_cutoff=params.cluster_cutoff,
            bot_wait=params.bot_wait,
            bot_depth=params.bot_depth,
            bot_stop=params.bot_stop,
            bot_late_step=params.bot_late_step,
            bot_late_stop=params.bot_late_stop,
            ghost_depth=params.ghost_depth,
            ghost_good=mutate_tuple(rng, params.ghost_good, [10, 15, 20, 25, 30, 35], [rng.randrange(4)]),
            ghost_deep_extra=mutate_tuple(rng, params.ghost_deep_extra, [1, 2, 3, 4], [rng.randrange(4)]),
            ghost_deep_stop=mutate_tuple(rng, params.ghost_deep_stop, [1, 3, 5, 7, 10], [rng.randrange(4)]),
            ghost_late_step=params.ghost_late_step,
            ghost_sample_late_only=params.ghost_sample_late_only,
            bot_order=params.bot_order,
            ghost_order=params.ghost_order,
        )
    if choice == 7:
        return Params(
            dense_cutoff=params.dense_cutoff,
            cluster_cutoff=params.cluster_cutoff,
            bot_wait=params.bot_wait,
            bot_depth=params.bot_depth,
            bot_stop=params.bot_stop,
            bot_late_step=params.bot_late_step,
            bot_late_stop=params.bot_late_stop,
            ghost_depth=params.ghost_depth,
            ghost_good=params.ghost_good,
            ghost_deep_extra=params.ghost_deep_extra,
            ghost_deep_stop=params.ghost_deep_stop,
            ghost_late_step=mutate_tuple(rng, params.ghost_late_step, [30, 40, 60, 80, 100, 120], [rng.randrange(4)]),
            ghost_sample_late_only=rng.choice([False, True]),
            bot_order=params.bot_order,
            ghost_order=params.ghost_order,
        )
    if choice == 8:
        return Params(
            dense_cutoff=params.dense_cutoff,
            cluster_cutoff=params.cluster_cutoff,
            bot_wait=params.bot_wait,
            bot_depth=params.bot_depth,
            bot_stop=params.bot_stop,
            bot_late_step=params.bot_late_step,
            bot_late_stop=params.bot_late_stop,
            ghost_depth=params.ghost_depth,
            ghost_good=params.ghost_good,
            ghost_deep_extra=params.ghost_deep_extra,
            ghost_deep_stop=params.ghost_deep_stop,
            ghost_late_step=params.ghost_late_step,
            ghost_sample_late_only=params.ghost_sample_late_only,
            bot_order=rng.randrange(5),
            ghost_order=params.ghost_order,
        )
    return Params(
        dense_cutoff=params.dense_cutoff,
        cluster_cutoff=params.cluster_cutoff,
        bot_wait=params.bot_wait,
        bot_depth=params.bot_depth,
        bot_stop=params.bot_stop,
        bot_late_step=params.bot_late_step,
        bot_late_stop=params.bot_late_stop,
        ghost_depth=params.ghost_depth,
        ghost_good=params.ghost_good,
        ghost_deep_extra=params.ghost_deep_extra,
        ghost_deep_stop=params.ghost_deep_stop,
        ghost_late_step=params.ghost_late_step,
        ghost_sample_late_only=params.ghost_sample_late_only,
        bot_order=params.bot_order,
        ghost_order=rng.randrange(5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    engine = MazeEngine()
    best = _baseline_params()
    best_score, best_styles = evaluate(engine, best)
    print("baseline", round(best_score, 2), tuple(round(x, 2) for x in best_styles), best, flush=True)

    seen: Dict[Params, float] = {best: best_score}
    frontier = [best]

    for i in range(args.iterations):
        parent = rng.choice(frontier)
        cand = mutate(rng, parent)
        if cand in seen:
            continue
        score, styles = evaluate(engine, cand)
        seen[cand] = score
        if score > best_score:
            best = cand
            best_score = score
            best_styles = styles
            print("improved", i, round(score, 2), tuple(round(x, 2) for x in styles), cand, flush=True)
            frontier.append(cand)
            frontier = sorted(frontier, key=lambda p: seen[p], reverse=True)[:20]

    print("best", round(best_score, 2), tuple(round(x, 2) for x in best_styles), best, flush=True)


if __name__ == "__main__":
    main()
