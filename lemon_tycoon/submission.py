from players.player import Player
from typing import Dict, List, Tuple
import math


class SubmissionPlayer(Player):
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

        # Heuristic popularity estimate: how many total factories (across everyone)
        # likely exist per ID. Updated from observed sabotage outcomes.
        self.id_popularity = [0.0] * (self.max_factory_id + 1)
        for i in range(1, self.max_factory_id + 1):
            # Bias toward high IDs, which rational strategies prefer.
            self.id_popularity[i] = max(0.0, self.production_rates[i - 1] / 2.0)

        self.last_round = -1

    def _update_popularity(
        self,
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> None:
        # Decay old information.
        for i in range(1, self.max_factory_id + 1):
            self.id_popularity[i] *= 0.93

        # Fresh hard evidence from last round sabotages.
        for factory_id, destroyed in destroyed_factory_counts.items():
            if 1 <= factory_id <= self.max_factory_id:
                # If an ID was sabotaged and X were destroyed, then at least that many
                # existed before sabotage.
                self.id_popularity[factory_id] = max(self.id_popularity[factory_id], float(destroyed))

        # If many players sabotaged an ID, it's probably important.
        sabotage_pressure = [0] * (self.max_factory_id + 1)
        for player_sabotages in sabotages_by_player:
            for factory_id in player_sabotages:
                if 1 <= factory_id <= self.max_factory_id:
                    sabotage_pressure[factory_id] += 1

        for factory_id in range(1, self.max_factory_id + 1):
            if sabotage_pressure[factory_id] > 0:
                self.id_popularity[factory_id] += 0.75 * sabotage_pressure[factory_id]

    def _select_sabotages(
        self,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
    ) -> List[int]:
        if your_lemons < self.sabotage_cost:
            return []

        my_top_rival_lemons = max(all_lemons[pid] for pid in range(self.num_players) if pid != self.player_id)
        rounds_left = self.max_rounds - self.last_round

        # Threat score: expected value opponents might be getting from each ID.
        # Penalize sabotaging IDs where we have many factories.
        threat = []
        total_own_factories = max(1, sum(your_factories))
        for factory_id in range(1, self.max_factory_id + 1):
            opp_estimate = max(0.0, self.id_popularity[factory_id] - your_factories[factory_id - 1])
            own_share = your_factories[factory_id - 1] / total_own_factories
            score = opp_estimate * self.production_rates[factory_id - 1] - 20.0 * own_share
            threat.append((score, factory_id))

        threat.sort(reverse=True)
        best_score, best_id = threat[0]

        sabotages: List[int] = []

        # Emergency denial near endgame or if a rival is close to goal.
        emergency = my_top_rival_lemons >= self.goal_lemons - 120 or rounds_left <= 18
        if emergency and best_score > self.sabotage_cost:
            sabotages.append(best_id)
            if your_lemons >= 2 * self.sabotage_cost:
                second_score, second_id = threat[1]
                if second_score > self.sabotage_cost * 0.85 and second_id != best_id:
                    sabotages.append(second_id)
            return sabotages

        # Normal play: occasional single sabotage when it looks efficient.
        if your_lemons >= 4 * self.buy_price and best_score > self.sabotage_cost * 1.35:
            sabotages.append(best_id)

        return sabotages

    def _choose_buy_mix(
        self,
        your_factories: List[int],
        sabotages_by_player: List[List[int]],
    ) -> List[int]:
        # Core set of strong IDs.
        candidates = [16, 15, 14, 13]
        candidates = [i for i in candidates if i <= self.max_factory_id]
        if not candidates:
            return [self.max_factory_id]

        # Penalize IDs that were heavily targeted last round.
        sabotaged_count = {i: 0 for i in candidates}
        for row in sabotages_by_player:
            for factory_id in row:
                if factory_id in sabotaged_count:
                    sabotaged_count[factory_id] += 1

        # Diversification pressure based on our current concentration.
        own_total = max(1, sum(your_factories))
        weighted_candidates = []
        for idx, factory_id in enumerate(candidates):
            base_weight = [0.48, 0.27, 0.16, 0.09][idx]
            concentration = your_factories[factory_id - 1] / own_total
            anti_concentration = max(0.5, 1.0 - 0.9 * concentration)
            anti_sabotage = 1.0 / (1.0 + 0.55 * sabotaged_count[factory_id])
            weighted_candidates.append((base_weight * anti_concentration * anti_sabotage, factory_id))

        weighted_candidates.sort(reverse=True)
        # Return IDs in descending preference order; purchase cycles through these.
        return [factory_id for _, factory_id in weighted_candidates]

    def play(
        self,
        round_number: int,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> Tuple[List[int], List[int], List[int]]:
        self.last_round = round_number
        self._update_popularity(destroyed_factory_counts, sabotages_by_player)

        rounds_left = self.max_rounds - round_number

        # Sell very low-value factories in early/mid game to recycle into high IDs.
        sells: List[int] = []
        if rounds_left > 4:
            for factory_id in range(1, min(5, self.max_factory_id + 1)):
                count = your_factories[factory_id - 1]
                if count <= 0:
                    continue
                # Sell more aggressively for lower IDs.
                if factory_id <= 2:
                    sells.extend([factory_id] * count)
                elif factory_id <= 4 and rounds_left > 7:
                    # Keep at most one to avoid over-churn.
                    extra = max(0, count - 1)
                    if extra > 0:
                        sells.extend([factory_id] * extra)

        projected_lemons = your_lemons + len(sells) * self.sell_price

        sabotages = self._select_sabotages(projected_lemons, your_factories, all_lemons)
        projected_lemons -= len(sabotages) * self.sabotage_cost

        # Keep a small reserve in the final stretch; otherwise spend aggressively.
        reserve = 0.0
        if rounds_left <= 10:
            reserve = self.buy_price

        budget = max(0.0, projected_lemons - reserve)
        buy_count = int(budget // self.buy_price)

        buys: List[int] = []
        if buy_count > 0:
            preference_order = self._choose_buy_mix(your_factories, sabotages_by_player)
            plen = len(preference_order)
            for i in range(buy_count):
                buys.append(preference_order[i % plen])

        return buys, sells, sabotages


# Alias for some judges that expect this exact class name.
SubmissionStrategy = SubmissionPlayer
