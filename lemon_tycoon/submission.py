from players.player import Player
from typing import Dict, List, Tuple
import math


class SubmissionStrategy(Player):
    """Rush the strongest factory ID, but avoid wasting lemons late."""

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
        self.buy_price = int(buy_price)
        self.sabotage_cost = sabotage_cost
        self.goal_lemons = goal_lemons
        self.max_rounds = max_rounds

        self.max_factory_id = 2 ** factory_bit_width
        self.best_id = self.max_factory_id
        self.best_prod = 2 * factory_bit_width

        self.production_rates = [
            2.0 * math.log2(factory_id)
            for factory_id in range(1, self.max_factory_id + 1)
        ]

        self.id_popularity = [0.0] * (self.max_factory_id + 1)
        self.id_pressure = [0.0] * (self.max_factory_id + 1)

        for factory_id in range(1, self.max_factory_id + 1):
            # Strong prior that high IDs are popular in competitive play.
            self.id_popularity[factory_id] = 0.12 * factory_id

    def _update_beliefs(
        self,
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> None:
        for factory_id in range(1, self.max_factory_id + 1):
            self.id_popularity[factory_id] *= 0.92
            self.id_pressure[factory_id] *= 0.82

        for factory_id, destroyed in destroyed_factory_counts.items():
            if 1 <= factory_id <= self.max_factory_id:
                self.id_popularity[factory_id] = max(
                    self.id_popularity[factory_id],
                    float(destroyed),
                )

        hit_count = [0] * (self.max_factory_id + 1)
        for pid, ids in enumerate(sabotages_by_player):
            if pid == self.player_id:
                continue
            for factory_id in set(ids):
                if 1 <= factory_id <= self.max_factory_id:
                    hit_count[factory_id] += 1

        denom = max(1, self.num_players - 1)
        for factory_id in range(1, self.max_factory_id + 1):
            if hit_count[factory_id] > 0:
                self.id_pressure[factory_id] += 0.6 * (hit_count[factory_id] / denom)

    def _choose_buy_count(self, your_lemons: float, your_factories: List[int]) -> int:
        lemons = int(round(your_lemons))
        factories = your_factories[self.best_id - 1]

        if factories <= 0:
            return lemons // self.buy_price

        # If the current stack can finish within two production steps, stop buying
        # and preserve lemons for the winner tie-break.
        no_buy_rounds = math.ceil(
            max(0.0, self.goal_lemons - lemons) / (self.best_prod * factories)
        )
        if no_buy_rounds <= 2:
            return 0

        return lemons // self.buy_price

    def _pick_sabotages(
        self,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        rounds_left: int,
    ) -> List[int]:
        current_production = self.best_prod * your_factories[self.best_id - 1]
        if your_lemons + current_production >= self.goal_lemons:
            return []
        if your_lemons < self.sabotage_cost:
            return []

        richest_other = max(
            all_lemons[player_id]
            for player_id in range(self.num_players)
            if player_id != self.player_id
        )
        if richest_other < max(900.0, your_lemons + 180.0):
            return []

        horizon = 3 if rounds_left <= 6 else 2
        best_score = float("-inf")
        best_factory_id = self.best_id

        for factory_id in range(max(1, self.max_factory_id - 3), self.max_factory_id + 1):
            production = self.production_rates[factory_id - 1]
            own_count = your_factories[factory_id - 1]
            estimated_opp_count = max(0.0, self.id_popularity[factory_id] - own_count)
            score = (
                estimated_opp_count * production * horizon
                - own_count * production * horizon
                - self.sabotage_cost
                - 8.0 * self.id_pressure[factory_id]
            )
            if score > best_score:
                best_score = score
                best_factory_id = factory_id

        threshold = 70.0
        if rounds_left <= 6:
            threshold = 35.0
        if richest_other >= 1500.0:
            threshold -= 10.0

        if best_score > threshold:
            return [best_factory_id]

        return []

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> Tuple[List[int], List[int], List[int]]:
        rounds_left = self.max_rounds - round_number
        self._update_beliefs(destroyed_factory_counts, sabotages_by_player)

        sabotages = self._pick_sabotages(
            your_lemons,
            your_factories,
            all_lemons,
            rounds_left,
        )

        remaining_lemons = your_lemons - len(sabotages) * self.sabotage_cost
        if remaining_lemons < 0:
            sabotages = []
            remaining_lemons = your_lemons

        buy_count = self._choose_buy_count(remaining_lemons, your_factories)
        buys = [self.best_id] * buy_count

        return buys, [], sabotages


SubmissionPlayer = SubmissionStrategy
