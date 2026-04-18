from players.player import Player
from typing import Dict, List, Tuple
import math
import random


class SubmissionStrategy(Player):
    """Risk-adjusted investment strategy for Lemon Tycoon."""

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

        # EMA estimate of sabotage hazard in [0,1] for each ID.
        self.hazard = [0.0] * (self.max_factory_id + 1)

        # Estimated market popularity by ID from observed sabotage outcomes.
        self.popularity = [0.0] * (self.max_factory_id + 1)

        # Slight per-player asymmetry to avoid mirror lock in local self-play.
        self.rng = random.Random(3571 * (player_id + 1) + 11)

    def _update_models(
        self,
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> None:
        # Decay old information.
        for fid in range(1, self.max_factory_id + 1):
            self.hazard[fid] *= 0.80
            self.popularity[fid] *= 0.90

        # Fresh sabotage pressure from all players last round.
        hit_count = [0] * (self.max_factory_id + 1)
        for ids in sabotages_by_player:
            for fid in set(ids):
                if 1 <= fid <= self.max_factory_id:
                    hit_count[fid] += 1

        denom = max(1, self.num_players)
        for fid in range(1, self.max_factory_id + 1):
            frac = hit_count[fid] / denom
            # Blend toward recent observed rate.
            self.hazard[fid] = 0.65 * self.hazard[fid] + 0.35 * frac

        # Destroyed counts provide hard evidence of concentration.
        for fid, destroyed in destroyed_factory_counts.items():
            if 1 <= fid <= self.max_factory_id:
                self.popularity[fid] = max(self.popularity[fid], float(destroyed))

    def _short_horizon_value(self, fid: int, rounds_left: int, own_count: int) -> float:
        """Risk-adjusted buy value over a short horizon.

        Short horizon responds faster to sabotage-heavy environments.
        """
        if rounds_left <= 1:
            return self.sell_price - self.buy_price

        prod = self.production_rates[fid - 1]
        # New factories start next round.
        horizon = max(1, min(10, rounds_left - 1))

        # Convert hazard to per-round survival; clamp to avoid extremes.
        p = min(0.95, max(0.0, self.hazard[fid]))
        survival = max(0.05, 1.0 - p)

        expected_prod = 0.0
        alive_prob = 1.0
        for _ in range(horizon):
            alive_prob *= survival
            expected_prod += prod * alive_prob

        # Penalize concentration to reduce catastrophic sabotage losses.
        concentration_penalty = 1.0 / (1.0 + 0.03 * own_count)

        # Small per-player id preference for tie breaking.
        tie_break = 0.02 if ((fid + self.player_id) % 5 == 0) else 0.0
        tie_break += 0.01 * self.rng.random()

        return (expected_prod * concentration_penalty) + self.sell_price - self.buy_price + tie_break

    def _select_buys(self, lemons_after_sabotage: float, your_factories: List[int], rounds_left: int) -> List[int]:
        buy_count = int(lemons_after_sabotage // self.buy_price)
        if buy_count <= 0:
            return []

        # Score all IDs by risk-adjusted value.
        scored: List[Tuple[float, int]] = []
        for fid in range(1, self.max_factory_id + 1):
            scored.append((self._short_horizon_value(fid, rounds_left, your_factories[fid - 1]), fid))
        scored.sort(reverse=True)

        viable_ids = [fid for score, fid in scored if score > 0.0][:8]
        if not viable_ids:
            return []

        buys: List[int] = []
        # Greedy-diversified fill.
        local_added = [0] * (self.max_factory_id + 1)
        for _ in range(buy_count):
            best_fid = viable_ids[0]
            best_score = -10**18
            for fid in viable_ids:
                own = your_factories[fid - 1] + local_added[fid]
                score = self._short_horizon_value(fid, rounds_left, own)
                if score > best_score:
                    best_score = score
                    best_fid = fid
            buys.append(best_fid)
            local_added[best_fid] += 1

        return buys

    def _select_sabotage(
        self,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        rounds_left: int,
    ) -> List[int]:
        if your_lemons < self.sabotage_cost:
            return []

        top_other = max(all_lemons[i] for i in range(self.num_players) if i != self.player_id)
        emergency = top_other >= (self.goal_lemons - 150) or rounds_left <= 18

        # Evaluate sabotage targets by estimated denial minus self-harm.
        best_score = -10**18
        best_id = 1
        horizon = max(1, min(8, rounds_left - 1))
        for fid in range(1, self.max_factory_id + 1):
            prod = self.production_rates[fid - 1]
            est_total = self.popularity[fid]
            own = your_factories[fid - 1]
            opp_est = max(0.0, est_total - own)

            denial = opp_est * prod * horizon
            self_harm = own * prod * horizon
            # Prioritize IDs opponents keep buying but that are not constantly sabotaged already.
            score = denial - self_harm - self.sabotage_cost - 8.0 * self.hazard[fid]
            if score > best_score:
                best_score = score
                best_id = fid

        if emergency and best_score > 0.0:
            # In endgame, spend for denial.
            actions = [best_id]
            # Optional second sabotage if very rich and second is also attractive.
            if your_lemons >= 2 * self.sabotage_cost + self.buy_price:
                second_best_score = -10**18
                second_best_id = best_id
                for fid in range(1, self.max_factory_id + 1):
                    if fid == best_id:
                        continue
                    prod = self.production_rates[fid - 1]
                    est_total = self.popularity[fid]
                    own = your_factories[fid - 1]
                    opp_est = max(0.0, est_total - own)
                    score = opp_est * prod * horizon - own * prod * horizon - self.sabotage_cost
                    if score > second_best_score:
                        second_best_score = score
                        second_best_id = fid
                if second_best_score > 0.0:
                    actions.append(second_best_id)
            return actions

        # Midgame: sabotage rarely, only when clearly worth it.
        if your_lemons >= (3 * self.buy_price + self.sabotage_cost) and best_score > 30.0:
            return [best_id]

        return []

    def _select_sells(self, your_factories: List[int], rounds_left: int) -> List[int]:
        # Only recycle ultra-low IDs with poor payoff and enough time to re-invest.
        if rounds_left <= 6:
            return []

        sells: List[int] = []
        for fid in range(1, min(4, self.max_factory_id + 1)):
            c = your_factories[fid - 1]
            if c <= 0:
                continue
            if fid == 1:
                sells.extend([fid] * c)
            elif fid <= 3 and rounds_left > 12:
                extra = max(0, c - 1)
                sells.extend([fid] * extra)
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

        self._update_models(destroyed_factory_counts, sabotages_by_player)

        sells = self._select_sells(your_factories, rounds_left)
        projected_lemons = your_lemons + len(sells) * self.sell_price

        sabotages = self._select_sabotage(projected_lemons, your_factories, all_lemons, rounds_left)
        projected_lemons -= len(sabotages) * self.sabotage_cost

        # Keep a tiny reserve late in the game to allow tactical sabotage next round.
        reserve = self.sabotage_cost if rounds_left <= 10 else 0.0
        buys = self._select_buys(max(0.0, projected_lemons - reserve), your_factories, rounds_left)

        return buys, sells, sabotages


# Compatibility with local runner import.
SubmissionPlayer = SubmissionStrategy
