#!/usr/bin/env python3
"""Benchmark SubmissionStrategy against simple reference opponents.

Usage:
  python benchmark.py
  python benchmark.py --matches 200
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Type

from engine import GameEngine
from players.player import Player
from submission import SubmissionPlayer


GAME_PARAMS: Dict[str, Any] = {
    "num_players": 4,
    "factory_bit_width": 4,
    "sell_price": 5.0,
    "buy_price": 15.0,
    "sabotage_cost": 15.0,
    "initial_lemons": 30.0,
    "goal_lemons": 2000.0,
    "max_rounds": 200,
}


class RandomBuyer(Player):
    def __init__(
        self,
        player_id: int,
        num_players: int,
        factory_bit_width: int,
        sell_price: float,
        buy_price: float,
        sabotage_cost: float,
        initial_lemons: float,
        goal_lemons: float,
        max_rounds: int,
    ):
        self.max_factory_id = 2 ** factory_bit_width
        self.buy_price = buy_price
        self.rng = random.Random(1009 + player_id * 37)

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ):
        n = int(your_lemons // self.buy_price)
        buys = [self.rng.randint(1, self.max_factory_id) for _ in range(n)]
        return buys, [], []


class GreedyHighID(Player):
    def __init__(
        self,
        player_id: int,
        num_players: int,
        factory_bit_width: int,
        sell_price: float,
        buy_price: float,
        sabotage_cost: float,
        initial_lemons: float,
        goal_lemons: float,
        max_rounds: int,
    ):
        self.best_id = 2 ** factory_bit_width
        self.buy_price = buy_price

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ):
        n = int(your_lemons // self.buy_price)
        return [self.best_id] * n, [], []


class HeavySaboteur(Player):
    def __init__(
        self,
        player_id: int,
        num_players: int,
        factory_bit_width: int,
        sell_price: float,
        buy_price: float,
        sabotage_cost: float,
        initial_lemons: float,
        goal_lemons: float,
        max_rounds: int,
    ):
        self.max_factory_id = 2 ** factory_bit_width
        self.buy_price = buy_price
        self.sabotage_cost = sabotage_cost

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ):
        # Buy some high IDs, but reserve for repeated sabotage on top IDs.
        reserve = 2 * self.sabotage_cost
        budget = max(0.0, your_lemons - reserve)
        n = int(budget // self.buy_price)
        buys = [self.max_factory_id] * n

        sabotages: List[int] = []
        remaining = your_lemons - n * self.buy_price
        if remaining >= self.sabotage_cost:
            sabotages.append(self.max_factory_id)
            remaining -= self.sabotage_cost
        if remaining >= self.sabotage_cost:
            sabotages.append(max(1, self.max_factory_id - 1))
        return buys, [], sabotages


@dataclass
class Result:
    rank: int
    lemons: float


def run_one_match(player_ctors: Sequence[Type[Player]]) -> List[Result]:
    engine = GameEngine(player_ctors, GAME_PARAMS)
    while not engine.is_game_over():
        engine.step()

    state = engine.get_state()
    rankings = engine.get_rankings()
    rank_of = {pid: idx + 1 for idx, pid in enumerate(rankings)}
    return [Result(rank=rank_of[pid], lemons=float(state["lemons"][pid])) for pid in range(GAME_PARAMS["num_players"])]


def benchmark(matches: int) -> None:
    opponents: List[Type[Player]] = [RandomBuyer, GreedyHighID, HeavySaboteur]

    total_rank = 0.0
    total_lemons = 0.0
    wins = 0

    for m in range(matches):
        random.shuffle(opponents)
        # Rotate seat to reduce ID-position artifacts.
        my_seat = m % GAME_PARAMS["num_players"]
        table: List[Type[Player]] = [opponents[0], opponents[1], opponents[2], GreedyHighID]
        table[my_seat] = SubmissionPlayer

        results = run_one_match(table)
        mine = results[my_seat]

        total_rank += mine.rank
        total_lemons += mine.lemons
        if mine.rank == 1:
            wins += 1

    avg_rank = total_rank / matches
    avg_lemons = total_lemons / matches
    win_rate = wins / matches

    print("Benchmark complete")
    print(f"Matches: {matches}")
    print(f"Average rank: {avg_rank:.3f}")
    print(f"Average final lemons: {avg_lemons:.2f}")
    print(f"Win rate: {100.0 * win_rate:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Lemon Tycoon submission strategy")
    parser.add_argument("--matches", type=int, default=100, help="number of matches to run")
    args = parser.parse_args()

    if args.matches <= 0:
        raise SystemExit("--matches must be positive")

    benchmark(args.matches)


if __name__ == "__main__":
    main()
