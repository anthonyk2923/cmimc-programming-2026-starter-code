from players.player import Player
from typing import Dict, List, Tuple
import math
import random


class SubmissionStrategy(Player):
    """Adaptive strategy for Lemon Tycoon.

    Key ideas:
    - Buy according to expected value over remaining rounds.
    - Diversify IDs when sabotage pressure is high.
    - Sabotage only when estimated denial value exceeds cost.
    - Use small per-player stochasticity to avoid mirror ties in self-play.
    """

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
        self.max_factory_id = 2 ** factory_bit_width
        self.sell_price = sell_price
        self.buy_price = buy_price
        self.sabotage_cost = sabotage_cost
        self.goal_lemons = goal_lemons
        self.max_rounds = max_rounds

        self.production_rates = [2.0 * math.log2(i) for i in range(1, self.max_factory_id + 1)]

        # Estimated opponent+global concentration per ID, learned from sabotages.
        self.id_popularity = [0.0] * (self.max_factory_id + 1)

        # Recent sabotage pressure per ID in [0, 1].
        self.id_sabotage_pressure = [0.0] * (self.max_factory_id + 1)

        # Player-specific RNG to break symmetry in mirrors.
        self.rng = random.Random(7919 * (player_id + 1) + 17)

    def _update_beliefs(
        self,
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> None:
        # Decay memory.
        for i in range(1, self.max_factory_id + 1):
            self.id_popularity[i] *= 0.92
            self.id_sabotage_pressure[i] *= 0.82

        # Direct evidence from last-round sabotages.
        for factory_id, destroyed in destroyed_factory_counts.items():
            if 1 <= factory_id <= self.max_factory_id:
                self.id_popularity[factory_id] = max(self.id_popularity[factory_id], float(destroyed))

        # How many distinct opponents targeted each ID last round.
        hit_count = [0] * (self.max_factory_id + 1)
        for pid, ids in enumerate(sabotages_by_player):
            if pid == self.player_id:
                continue
            seen = set()
            for factory_id in ids:
                if 1 <= factory_id <= self.max_factory_id and factory_id not in seen:
                    hit_count[factory_id] += 1
                    seen.add(factory_id)

        denom = max(1, self.num_players - 1)
        for factory_id in range(1, self.max_factory_id + 1):
            if hit_count[factory_id] > 0:
                frac = hit_count[factory_id] / denom
                self.id_sabotage_pressure[factory_id] += 0.6 * frac

    def _effective_roi(self, factory_id: int, rounds_left: int, own_count: int) -> float:
        """Expected net value of buying one factory of ID factory_id now."""
        if rounds_left <= 1:
            # No production time left; only liquidation value.
            return self.sell_price - self.buy_price

        production = self.production_rates[factory_id - 1]
        prod_rounds = rounds_left - 1  # bought factories start producing next round

        # Estimate lifetime survival probability from sabotage pressure.
        pressure = min(0.85, self.id_sabotage_pressure[factory_id])
        survival = max(0.2, (1.0 - pressure) ** max(1, prod_rounds // 2))

        # Diversification penalty to avoid all-in concentration.
        concentration_penalty = 1.0 / (1.0 + 0.015 * own_count)

        expected_prod = production * prod_rounds * survival * concentration_penalty
        return expected_prod + self.sell_price - self.buy_price

    def _pick_buys(
        self,
        your_lemons: float,
        your_factories: List[int],
        rounds_left: int,
    ) -> List[int]:
        budget_count = int(your_lemons // self.buy_price)
        if budget_count <= 0:
            return []

        # Score every ID by expected ROI + slight player/round jitter to break symmetry.
        scored: List[Tuple[float, int]] = []
        for factory_id in range(1, self.max_factory_id + 1):
            own_count = your_factories[factory_id - 1]
            roi = self._effective_roi(factory_id, rounds_left, own_count)
            # Tiny deterministic noise for tie-breaking.
            roi += 0.02 * self.rng.random()
            # Mild per-player specialty to reduce mirror outcomes.
            if factory_id in (16, 15, 14, 13):
                roi += 0.03 * (((factory_id + self.player_id) % 4) == 0)
            scored.append((roi, factory_id))

        scored.sort(reverse=True)

        buys: List[int] = []
        # Buy from top EV IDs with cyclical diversification.
        top_ids = [fid for roi, fid in scored if roi > 0.0][:6]
        if not top_ids:
            return []

        for idx in range(budget_count):
            # Re-evaluate lightly every pick to avoid overstacking one ID.
            if idx % 3 == 0:
                top_ids.sort(
                    key=lambda fid: self._effective_roi(fid, rounds_left, your_factories[fid - 1] + buys.count(fid)),
                    reverse=True,
                )
            pick = top_ids[idx % len(top_ids)]
            buys.append(pick)

        return buys

    def _pick_sabotages(
        self,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        rounds_left: int,
    ) -> List[int]:
        if your_lemons < self.sabotage_cost or rounds_left <= 1:
            return []

        richest_other = max(all_lemons[i] for i in range(self.num_players) if i != self.player_id)
        close_race = richest_other >= self.goal_lemons * 0.82
        late_game = rounds_left <= 22

        candidates: List[Tuple[float, int]] = []
        for factory_id in range(1, self.max_factory_id + 1):
            prod = self.production_rates[factory_id - 1]
            est_total = self.id_popularity[factory_id]
            own = your_factories[factory_id - 1]
            est_opp = max(0.0, est_total - own)

            # Estimated denial value over remaining turns.
            denial = est_opp * prod * max(1, rounds_left - 2)
            self_harm = own * prod * max(1, rounds_left - 2)
            score = denial - self_harm - self.sabotage_cost

            # Prefer sabotaging high-pressure IDs only if we are not overexposed.
            score += 8.0 * self.id_sabotage_pressure[factory_id]
            candidates.append((score, factory_id))

        candidates.sort(reverse=True)
        best_score, best_id = candidates[0]

        sabotages: List[int] = []
        # Aggressive sabotage near finish or when rival is close.
        if (close_race or late_game) and best_score > 0:
            sabotages.append(best_id)
            if your_lemons >= 2 * self.sabotage_cost:
                second_score, second_id = candidates[1]
                if second_id != best_id and second_score > self.sabotage_cost * 0.25:
                    sabotages.append(second_id)
            return sabotages

        # Mid-game opportunistic sabotage if very efficient.
        if your_lemons >= self.sabotage_cost + 2 * self.buy_price and best_score > self.sabotage_cost * 0.7:
            sabotages.append(best_id)

        return sabotages

    def _pick_sells(self, your_factories: List[int], rounds_left: int) -> List[int]:
        # Usually selling is bad, but low IDs are weak and can be recycled early.
        if rounds_left <= 5:
            return []

        sells: List[int] = []
        for factory_id in range(1, min(6, self.max_factory_id + 1)):
            count = your_factories[factory_id - 1]
            if count <= 0:
                continue
            if factory_id <= 2:
                sells.extend([factory_id] * count)
            elif factory_id <= 5 and rounds_left > 10:
                # Keep one as a hedge, recycle extras.
                extra = max(0, count - 1)
                sells.extend([factory_id] * extra)
        return sells

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

        sells = self._pick_sells(your_factories, rounds_left)
        projected_lemons = your_lemons + len(sells) * self.sell_price

        sabotages = self._pick_sabotages(projected_lemons, your_factories, all_lemons, rounds_left)
        projected_lemons -= len(sabotages) * self.sabotage_cost

        # Small reserve in endgame for tactical sabotage.
        reserve = self.sabotage_cost if rounds_left <= 14 else 0.0
        buys = self._pick_buys(max(0.0, projected_lemons - reserve), your_factories, rounds_left)

        return buys, sells, sabotages


# Local runner imports SubmissionPlayer; keep compatibility.
SubmissionPlayer = SubmissionStrategy
