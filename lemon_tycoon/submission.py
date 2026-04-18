# from players.player import Player
# from typing import Dict, List, Tuple
# import math


# class SubmissionStrategy(Player):
#     """Rush the strongest factory ID, but avoid wasting lemons late."""

#     def __init__(
#         self,
#         player_id: int,
#         num_players: int,
#         factory_bit_width: int,
#         sell_price: float,
#         buy_price: float,
#         sabotage_cost: float,
#         initial_lemons: float,
#         goal_lemons: float,
#         max_rounds: int,
#     ):
#         self.player_id = player_id
#         self.num_players = num_players
#         self.sell_price = sell_price
#         self.buy_price = int(buy_price)
#         self.sabotage_cost = sabotage_cost
#         self.goal_lemons = goal_lemons
#         self.max_rounds = max_rounds

#         self.max_factory_id = 2 ** factory_bit_width
#         self.best_id = self.max_factory_id
#         self.best_prod = 2 * factory_bit_width

#         self.production_rates = [
#             2.0 * math.log2(factory_id)
#             for factory_id in range(1, self.max_factory_id + 1)
#         ]

#         self.id_popularity = [0.0] * (self.max_factory_id + 1)
#         self.id_pressure = [0.0] * (self.max_factory_id + 1)

#         for factory_id in range(1, self.max_factory_id + 1):
#             # Strong prior that high IDs are popular in competitive play.
#             self.id_popularity[factory_id] = 0.12 * factory_id

#     def _update_beliefs(
#         self,
#         destroyed_factory_counts: Dict[int, int],
#         sabotages_by_player: List[List[int]],
#     ) -> None:
#         for factory_id in range(1, self.max_factory_id + 1):
#             self.id_popularity[factory_id] *= 0.92
#             self.id_pressure[factory_id] *= 0.82

#         for factory_id, destroyed in destroyed_factory_counts.items():
#             if 1 <= factory_id <= self.max_factory_id:
#                 self.id_popularity[factory_id] = max(
#                     self.id_popularity[factory_id],
#                     float(destroyed),
#                 )

#         hit_count = [0] * (self.max_factory_id + 1)
#         for pid, ids in enumerate(sabotages_by_player):
#             if pid == self.player_id:
#                 continue
#             for factory_id in set(ids):
#                 if 1 <= factory_id <= self.max_factory_id:
#                     hit_count[factory_id] += 1

#         denom = max(1, self.num_players - 1)
#         for factory_id in range(1, self.max_factory_id + 1):
#             if hit_count[factory_id] > 0:
#                 self.id_pressure[factory_id] += 0.6 * \
#                     (hit_count[factory_id] / denom)

#     def _choose_buy_count(self, your_lemons: float, your_factories: List[int]) -> int:
#         lemons = int(round(your_lemons))
#         factories = your_factories[self.best_id - 1]

#         if factories <= 0:
#             return lemons // self.buy_price

#         # If the current stack can finish within two production steps, stop buying
#         # and preserve lemons for the winner tie-break.
#         no_buy_rounds = math.ceil(
#             max(0.0, self.goal_lemons - lemons) / (self.best_prod * factories)
#         )
#         if no_buy_rounds <= 2:
#             return 0

#         return lemons // self.buy_price

#     def _pick_sabotages(
#         self,
#         your_lemons: float,
#         your_factories: List[int],
#         all_lemons: List[float],
#         rounds_left: int,
#     ) -> List[int]:
#         current_production = self.best_prod * your_factories[self.best_id - 1]
#         if your_lemons + current_production >= self.goal_lemons:
#             return []
#         if your_lemons < self.sabotage_cost:
#             return []

#         richest_other = max(
#             all_lemons[player_id]
#             for player_id in range(self.num_players)
#             if player_id != self.player_id
#         )
#         if richest_other < max(900.0, your_lemons + 180.0):
#             return []

#         horizon = 3 if rounds_left <= 6 else 2
#         best_score = float("-inf")
#         best_factory_id = self.best_id

#         for factory_id in range(max(1, self.max_factory_id - 3), self.max_factory_id + 1):
#             production = self.production_rates[factory_id - 1]
#             own_count = your_factories[factory_id - 1]
#             estimated_opp_count = max(
#                 0.0, self.id_popularity[factory_id] - own_count)
#             score = (
#                 estimated_opp_count * production * horizon
#                 - own_count * production * horizon
#                 - self.sabotage_cost
#                 - 8.0 * self.id_pressure[factory_id]
#             )
#             if score > best_score:
#                 best_score = score
#                 best_factory_id = factory_id

#         threshold = 70.0
#         if rounds_left <= 6:
#             threshold = 35.0
#         if richest_other >= 1500.0:
#             threshold -= 10.0

#         if best_score > threshold:
#             return [best_factory_id]

#         return []

#     def play(
#         self,
#         round_number: int,
#         your_lemons: float,
#         your_factories: List[int],
#         all_lemons: List[float],
#         destroyed_factory_counts: Dict[int, int],
#         sabotages_by_player: List[List[int]],
#     ) -> Tuple[List[int], List[int], List[int]]:
#         rounds_left = self.max_rounds - round_number
#         self._update_beliefs(destroyed_factory_counts, sabotages_by_player)

#         sabotages = self._pick_sabotages(
#             your_lemons,
#             your_factories,
#             all_lemons,
#             rounds_left,
#         )

#         remaining_lemons = your_lemons - len(sabotages) * self.sabotage_cost
#         if remaining_lemons < 0:
#             sabotages = []
#             remaining_lemons = your_lemons

#         buy_count = self._choose_buy_count(remaining_lemons, your_factories)
#         buys = [self.best_id] * buy_count

#         return buys, [], sabotages


# SubmissionPlayer = SubmissionStrategy
from players.player import Player
from typing import Dict, List, Tuple
import math


class SubmissionStrategy(Player):
    """
    Diversified high-ID investor with opportunistic sell->sabotage plays.

    Main ideas:
    - Buy mostly among the top few IDs instead of only 16.
    - Stop buying when the game is likely to end too soon.
    - If an ID becomes crowded / repeatedly sabotaged, consider:
        sell all of our copies of that ID,
        sabotage that ID,
        rebuy into nearby IDs the same round.
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
        self.sell_price = float(sell_price)
        self.buy_price = int(buy_price)
        self.sabotage_cost = float(sabotage_cost)
        self.goal_lemons = float(goal_lemons)
        self.max_rounds = int(max_rounds)

        self.max_factory_id = 2 ** factory_bit_width

        # 1-indexed production table for convenience.
        self.prod = [0.0] * (self.max_factory_id + 1)
        for fid in range(1, self.max_factory_id + 1):
            self.prod[fid] = 2.0 * math.log2(fid)

        # Focus on the best few IDs; diversification matters because sabotage is by ID.
        self.top_ids = sorted(
            range(1, self.max_factory_id + 1),
            key=lambda x: self.prod[x],
            reverse=True,
        )[:5]

        # Belief state: rough estimate of how globally crowded each ID is.
        self.est_total_counts = [0.0] * (self.max_factory_id + 1)

        # Threat = how likely an ID is to get hit soon.
        self.threat = [0.0] * (self.max_factory_id + 1)

        # Cooldown discourages rebuilding recently sabotaged IDs too quickly.
        self.cooldown = [0] * (self.max_factory_id + 1)

        # Track our last known holdings so we can tell when a sabotage hit us.
        self.prev_own_factories = [0] * self.max_factory_id

    def _own_production(self, your_factories: List[int]) -> float:
        total = 0.0
        for fid in range(1, self.max_factory_id + 1):
            total += your_factories[fid - 1] * self.prod[fid]
        return total

    def _update_beliefs(
        self,
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> None:
        # Decay old information.
        for fid in range(1, self.max_factory_id + 1):
            self.est_total_counts[fid] *= 0.88
            self.threat[fid] *= 0.80
            if self.cooldown[fid] > 0:
                self.cooldown[fid] -= 1

        # If an ID was sabotaged, destroyed_factory_counts gives exact global count
        # of that ID that got wiped last round.
        for fid, destroyed in destroyed_factory_counts.items():
            if 1 <= fid <= self.max_factory_id:
                d = float(destroyed)
                # Blend exact observation into the estimate.
                self.est_total_counts[fid] = max(
                    d,
                    0.55 * self.est_total_counts[fid] + 0.45 * d,
                )
                self.threat[fid] += 1.25
                self.cooldown[fid] = max(self.cooldown[fid], 2)

                # If we had copies there last turn, treat it as a strong warning.
                if self.prev_own_factories[fid - 1] > 0:
                    self.threat[fid] += 1.50
                    self.cooldown[fid] = max(self.cooldown[fid], 3)

        # If opponents chose an ID for sabotage, that is a sign the ID is relevant.
        for pid, ids in enumerate(sabotages_by_player):
            if pid == self.player_id:
                continue
            for fid in set(ids):
                if 1 <= fid <= self.max_factory_id:
                    self.threat[fid] += 0.45
                    self.est_total_counts[fid] += 0.20
                    self.cooldown[fid] = max(self.cooldown[fid], 1)

    def _expected_buy_horizon(self, round_number: int, all_lemons: List[float]) -> int:
        """
        Rough estimate of how many future production rounds a new purchase
        is likely to enjoy before the game ends.

        A buy this round does NOT produce this round, so the true hard cap is:
            max_rounds - round_number - 1
        """
        future_rounds = self.max_rounds - round_number - 1
        if future_rounds <= 0:
            return 0

        richest = max(all_lemons)
        dist = self.goal_lemons - richest

        # Crude but fast heuristic for likely remaining lifetime.
        if dist <= 80:
            h = 1
        elif dist <= 250:
            h = 2
        elif dist <= 600:
            h = 3
        else:
            h = 4

        return min(future_rounds, h)

    def _should_stop_buying(
        self,
        your_lemons: float,
        your_factories: List[int],
        round_number: int,
        all_lemons: List[float],
    ) -> bool:
        own_prod = self._own_production(your_factories)
        if own_prod <= 0:
            return False

        # If we can likely finish soon with current production, keep cash.
        remaining = max(0.0, self.goal_lemons - your_lemons)
        eta = math.ceil(remaining / max(1.0, own_prod))
        if eta <= 2:
            return True

        # If the game itself is likely ending very soon, stop buying.
        horizon = self._expected_buy_horizon(round_number, all_lemons)
        if horizon < 2:
            return True

        return False

    def _choose_sabotage_plan(
        self,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        round_number: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Returns (sells, sabotages).
        If we sabotage an ID we own, we first sell all our copies because sells
        happen before sabotage.
        """
        richest_other = max(
            all_lemons[pid]
            for pid in range(self.num_players)
            if pid != self.player_id
        )

        horizon = max(1, self._expected_buy_horizon(round_number, all_lemons))
        aggressive = (
            richest_other >= self.goal_lemons * 0.70
            or richest_other > your_lemons + 140.0
        )

        best_score = float("-inf")
        best_id = None

        for fid in self.top_ids:
            own = your_factories[fid - 1]
            est_total = max(self.est_total_counts[fid], float(own))
            est_opp = max(0.0, est_total - own)

            if est_opp < 1.5:
                continue

            # Value denied to others: future production + final liquidation.
            deny_value = est_opp * (self.prod[fid] * horizon + self.sell_price)

            # Penalize uncertain / already-hot IDs a bit.
            score = deny_value - self.sabotage_cost - 6.0 * self.threat[fid]

            # Small friction penalty if we have to rotate out of our own stack.
            if own > 0:
                score -= 1.25 * own

            if score > best_score:
                best_score = score
                best_id = fid

        if best_id is None:
            return [], []

        threshold = 26.0 if aggressive else 42.0
        own = your_factories[best_id - 1]

        available_cash = your_lemons + own * self.sell_price
        if best_score > threshold and available_cash >= self.sabotage_cost:
            sells = [best_id] * own
            sabotages = [best_id]
            return sells, sabotages

        return [], []

    def _choose_buys(
        self,
        cash: float,
        your_factories: List[int],
        round_number: int,
        all_lemons: List[float],
        avoid_id: int = -1,
    ) -> List[int]:
        if cash < self.buy_price:
            return []

        if self._should_stop_buying(cash, your_factories, round_number, all_lemons):
            return []

        horizon = self._expected_buy_horizon(round_number, all_lemons)
        if horizon < 2:
            return []

        budget = int(cash) // self.buy_price
        if budget <= 0:
            return []

        temp = your_factories[:]
        buys: List[int] = []

        for _ in range(budget):
            best_id = None
            best_score = float("-inf")

            for fid in self.top_ids:
                if fid == avoid_id:
                    continue

                # Factory value over expected remaining useful life.
                gross_value = self.prod[fid] * horizon + self.sell_price
                base_score = gross_value - self.buy_price

                # Diversify away from our own huge same-ID stack.
                own_stack_penalty = 0.30 * temp[fid - 1]

                # Penalize crowded / threatened IDs.
                crowd_penalty = 0.16 * self.est_total_counts[fid]
                threat_penalty = 1.70 * self.threat[fid]
                cooldown_penalty = 2.25 * self.cooldown[fid]

                score = (
                    base_score
                    - own_stack_penalty
                    - crowd_penalty
                    - threat_penalty
                    - cooldown_penalty
                )

                # When an ID is not under pressure, production speed matters a lot
                # because winning early avoids forced liquidation for us.
                safe_bonus = 0.10 * self.prod[fid] if self.threat[fid] < 0.75 else 0.0
                score += safe_bonus

                if score > best_score:
                    best_score = score
                    best_id = fid

            if best_id is None or best_score <= 0.0:
                break

            buys.append(best_id)
            temp[best_id - 1] += 1

        return buys

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> Tuple[List[int], List[int], List[int]]:
        self._update_beliefs(destroyed_factory_counts, sabotages_by_player)

        sells, sabotages = self._choose_sabotage_plan(
            your_lemons,
            your_factories,
            all_lemons,
            round_number,
        )

        # Apply planned sells to local state before deciding buys.
        post_sell_factories = your_factories[:]
        for fid in sells:
            idx = fid - 1
            if post_sell_factories[idx] > 0:
                post_sell_factories[idx] -= 1

        cash = your_lemons + len(sells) * self.sell_price - \
            len(sabotages) * self.sabotage_cost
        if cash < 0:
            sells = []
            sabotages = []
            post_sell_factories = your_factories[:]
            cash = your_lemons

        avoid_id = sabotages[0] if sabotages else -1
        buys = self._choose_buys(
            cash,
            post_sell_factories,
            round_number,
            all_lemons,
            avoid_id=avoid_id,
        )

        # Safety: never submit an over-budget buy list.
        max_affordable = int(cash) // self.buy_price
        if len(buys) > max_affordable:
            buys = buys[:max_affordable]

        self.prev_own_factories = your_factories[:]

        return buys, sells, sabotages


SubmissionPlayer = SubmissionStrategy
