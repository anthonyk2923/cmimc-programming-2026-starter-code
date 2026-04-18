#!/usr/bin/env python3
"""Benchmark Lemon Tycoon strategies against a tougher local gauntlet."""

from __future__ import annotations

import argparse
import itertools
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


class DiversifiedInvestor(Player):
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
        self.buy_price = buy_price
        self.top_ids = [2 ** factory_bit_width - i for i in range(5)]

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ):
        budget = int(your_lemons // self.buy_price)
        temp = your_factories[:]
        buys: List[int] = []
        for _ in range(budget):
            fid = min(self.top_ids, key=lambda x: (temp[x - 1], -x))
            buys.append(fid)
            temp[fid - 1] += 1
        return buys, [], []


class CautiousSaver(Player):
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
        self.buy_price = buy_price
        self.goal_lemons = goal_lemons
        self.best_id = 2 ** factory_bit_width
        self.best_prod = 2.0 * factory_bit_width

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ):
        current_prod = your_factories[self.best_id - 1] * self.best_prod
        if current_prod > 0 and your_lemons + current_prod >= self.goal_lemons:
            return [], [], []
        if current_prod > 0 and math.ceil(max(0.0, self.goal_lemons - your_lemons) / current_prod) <= 2:
            return [], [], []
        n = int(your_lemons // self.buy_price)
        return [self.best_id] * n, [], []


class LeaderPunisher(Player):
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
        self.player_id = player_id
        self.num_players = num_players
        self.sell_price = sell_price
        self.buy_price = buy_price
        self.sabotage_cost = sabotage_cost
        self.goal_lemons = goal_lemons
        self.top_ids = [2 ** factory_bit_width - i for i in range(4)]

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ):
        richest_other = max(
            all_lemons[pid]
            for pid in range(self.num_players)
            if pid != self.player_id
        )

        sabotages: List[int] = []
        sells: List[int] = []
        cash = your_lemons

        if richest_other >= self.goal_lemons - 180.0 or richest_other > your_lemons + 150.0:
            target = self.top_ids[0]
            own = your_factories[target - 1]
            if own > 0:
                sells = [target] * own
                cash += own * self.sell_price
            if cash >= self.sabotage_cost:
                sabotages.append(target)
                cash -= self.sabotage_cost
            if richest_other >= self.goal_lemons - 90.0 and cash >= self.sabotage_cost:
                sabotages.append(self.top_ids[1])
                cash -= self.sabotage_cost

        budget = int(cash // self.buy_price)
        buys = [self.top_ids[i % len(self.top_ids)] for i in range(budget)]
        return buys, sells, sabotages


class CycleSaboteur(Player):
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
        self.player_id = player_id
        self.num_players = num_players
        self.sell_price = sell_price
        self.buy_price = buy_price
        self.sabotage_cost = sabotage_cost
        self.goal_lemons = goal_lemons
        self.top_ids = [2 ** factory_bit_width - i for i in range(5)]

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ):
        sells: List[int] = []
        sabotages: List[int] = []
        cash = your_lemons

        richest_other = max(
            all_lemons[pid]
            for pid in range(self.num_players)
            if pid != self.player_id
        )

        threatened = set()
        for pid, ids in enumerate(sabotages_by_player):
            if pid != self.player_id:
                threatened.update(ids)

        if richest_other >= self.goal_lemons - 220.0 or destroyed_factory_counts:
            target = self.top_ids[0]
            if threatened:
                target = max(threatened, key=lambda fid: (fid, -your_factories[fid - 1]))
            own = your_factories[target - 1]
            if own > 0:
                sells = [target] * own
                cash += own * self.sell_price
            if cash >= self.sabotage_cost:
                sabotages.append(target)
                cash -= self.sabotage_cost

        budget = int(cash // self.buy_price)
        buys: List[int] = []
        for i in range(budget):
            fid = self.top_ids[(round_number + i) % len(self.top_ids)]
            buys.append(fid)
        return buys, sells, sabotages


OPPONENT_POOL: List[Type[Player]] = [
    RandomBuyer,
    GreedyHighID,
    DiversifiedInvestor,
    CautiousSaver,
    LeaderPunisher,
    CycleSaboteur,
]


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
    return [
        Result(rank=rank_of[pid], lemons=float(state["lemons"][pid]))
        for pid in range(GAME_PARAMS["num_players"])
    ]


def benchmark_player(
    player_ctor: Type[Player],
    repetitions: int,
) -> Dict[str, float]:
    total_rank = 0.0
    total_lemons = 0.0
    wins = 0
    matches = 0

    combos = list(itertools.combinations(OPPONENT_POOL, 3))
    for _ in range(repetitions):
        for combo in combos:
            for my_seat in range(GAME_PARAMS["num_players"]):
                table = list(combo)
                table.insert(my_seat, player_ctor)
                results = run_one_match(table)
                mine = results[my_seat]
                matches += 1
                total_rank += mine.rank
                total_lemons += mine.lemons
                if mine.rank == 1:
                    wins += 1

    return {
        "matches": float(matches),
        "avg_rank": total_rank / matches,
        "avg_lemons": total_lemons / matches,
        "win_rate": wins / matches,
    }


def benchmark(repetitions: int) -> None:
    stats = benchmark_player(SubmissionPlayer, repetitions)
    print("Benchmark complete")
    print(f"Tables per repetition: {len(list(itertools.combinations(OPPONENT_POOL, 3))) * GAME_PARAMS['num_players']}")
    print(f"Repetitions: {repetitions}")
    print(f"Matches: {int(stats['matches'])}")
    print(f"Average rank: {stats['avg_rank']:.3f}")
    print(f"Average final lemons: {stats['avg_lemons']:.2f}")
    print(f"Win rate: {100.0 * stats['win_rate']:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Lemon Tycoon submission strategy")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="number of full gauntlet repetitions",
    )
    args = parser.parse_args()

    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")

    benchmark(args.repetitions)


if __name__ == "__main__":
    main()
