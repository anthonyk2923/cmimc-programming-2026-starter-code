from players.player import Player
from typing import Dict, List, Tuple
import math


class SubmissionStrategy(Player):
    """
    High-tempo lemon racer with adaptive diversification and late anti-leader sabotage.

    The strategy is built around three ideas:
    - Rush high-production factories, especially ID 16, while sabotage pressure is low.
    - Diversify only when the lobby shows evidence that an ID is crowded or under attack.
    - Spend lemons on sabotage much more aggressively when another player is close
      to ending the game before we can win ourselves.
    """

    CONFIG = {
        "focus_width": 5,
        "count_decay": 0.90,
        "threat_decay": 0.82,
        "count_blend": 0.48,
        "threat_from_destroy": 1.35,
        "threat_from_enemy_call": 0.55,
        "threat_if_hit_us": 1.75,
        "cooldown_from_destroy": 2,
        "cooldown_if_hit_us": 3,
        "initial_soft_cap": 5,
        "war_soft_cap": 3,
        "crowd_penalty": 0.18,
        "threat_penalty": 1.45,
        "cooldown_penalty": 2.2,
        "own_stack_penalty": 0.28,
        "safe_bonus": 0.30,
        "emergency_top_bonus": 0.35,
        "high_id_target_bonus": 0.60,
        "leader_target_bonus_scale": 6.0,
        "repeat_sabotage_penalty": 12.0,
        "failed_repeat_penalty": 18.0,
        "repeat_window": 2,
        "same_round_cycle_bonus": 0.95,
        "sabotage_rebuild_top_k": 3,
        "low_id_sabotage_evidence": 5.0,
        "low_id_penalty": 0.55,
        "late_low_id_penalty": 0.95,
        "deep_war_threat": 1.80,
        "sabotage_self_cost_rounds": 0.90,
        "sabotage_threshold_calm": 28.0,
        "sabotage_threshold_danger": 8.0,
        "sabotage_threshold_emergency": 4.0,
        "min_sabotage_round": 9,
        "early_sabotage_gap": 260.0,
        "active_leader_floor": 320.0,
        "leader_advantage_floor": 140.0,
        "mid_sabotage_gap": 180.0,
        "small_bankroll_skip_mult": 2.5,
        "small_bankroll_gap": 140.0,
        "double_sabotage_gap": 80.0,
        "double_sabotage_lead": 140.0,
        "double_sabotage_second_margin": -4.0,
    }

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
        self.buy_price = float(buy_price)
        self.sabotage_cost = float(sabotage_cost)
        self.goal_lemons = float(goal_lemons)
        self.max_rounds = int(max_rounds)
        self.cfg = dict(self.CONFIG)

        self.max_factory_id = 2 ** factory_bit_width
        self.prod = [0.0] * (self.max_factory_id + 1)
        for fid in range(1, self.max_factory_id + 1):
            self.prod[fid] = 2.0 * math.log2(fid)

        self.focus_ids = sorted(
            range(1, self.max_factory_id + 1),
            key=lambda fid: self.prod[fid],
            reverse=True,
        )[: self.cfg["focus_width"]]

        # Structural prior: smart players usually prefer the top IDs.
        self.prior_counts = [0.0] * (self.max_factory_id + 1)
        for fid in self.focus_ids:
            self.prior_counts[fid] = max(
                0.0, 0.65 * (fid - (self.max_factory_id - 4)))

        self.est_total_counts = self.prior_counts[:]
        self.threat = [0.0] * (self.max_factory_id + 1)
        self.cooldown = [0] * (self.max_factory_id + 1)

        self.prev_own_factories = [0] * self.max_factory_id
        self.prev_all_lemons: List[float] | None = None
        self.enemy_sabotage_events = 0
        self.player_growth_baseline = [0.0] * self.num_players
        self.player_id_exposure = [
            [0.0] * (self.max_factory_id + 1)
            for _ in range(self.num_players)
        ]
        self.last_sabotage_round = [-10**9] * (self.max_factory_id + 1)
        self.last_sabotage_damage = [0.0] * (self.max_factory_id + 1)
        self.last_round_leader = -1
        self.last_round_leader_growth = 0.0

    def _own_production(self, your_factories: List[int]) -> float:
        total = 0.0
        for fid in range(1, self.max_factory_id + 1):
            total += your_factories[fid - 1] * self.prod[fid]
        return total

    def _richest_other(self, all_lemons: List[float]) -> float:
        best = 0.0
        for pid in range(self.num_players):
            if pid != self.player_id and all_lemons[pid] > best:
                best = all_lemons[pid]
        return best

    def _future_buy_rounds(self, round_number: int, all_lemons: List[float]) -> int:
        remaining = self.max_rounds - round_number - 1
        if remaining <= 0:
            return 0

        gap = self.goal_lemons - max(all_lemons)
        if gap <= 60.0:
            horizon = 1
        elif gap <= 180.0:
            horizon = 2
        elif gap <= 420.0:
            horizon = 3
        elif gap <= 800.0:
            horizon = 4
        else:
            horizon = 5

        return min(remaining, horizon)

    def _existing_factory_rounds(self, round_number: int, all_lemons: List[float]) -> int:
        remaining = self.max_rounds - round_number
        if remaining <= 0:
            return 0
        return min(remaining, self._future_buy_rounds(round_number, all_lemons) + 1)

    def _can_win_without_new_buys(
        self,
        your_lemons: float,
        your_factories: List[int],
    ) -> bool:
        return your_lemons + self._own_production(your_factories) >= self.goal_lemons

    def _update_beliefs(
        self,
        all_lemons: List[float],
        destroyed_factory_counts: Dict[int, int],
        sabotages_by_player: List[List[int]],
    ) -> None:
        for fid in range(1, self.max_factory_id + 1):
            self.est_total_counts[fid] = max(
                self.prior_counts[fid],
                self.est_total_counts[fid] * self.cfg["count_decay"],
            )
            self.threat[fid] *= self.cfg["threat_decay"]
            if self.cooldown[fid] > 0:
                self.cooldown[fid] -= 1

        for fid, destroyed in destroyed_factory_counts.items():
            if 1 <= fid <= self.max_factory_id:
                observed = float(destroyed)
                self.last_sabotage_damage[fid] = observed
                self.est_total_counts[fid] = max(
                    observed,
                    (1.0 - self.cfg["count_blend"]) *
                    self.est_total_counts[fid]
                    + self.cfg["count_blend"] * observed,
                )
                self.threat[fid] += self.cfg["threat_from_destroy"]
                self.cooldown[fid] = max(
                    self.cooldown[fid],
                    int(self.cfg["cooldown_from_destroy"]),
                )

                if self.prev_own_factories[fid - 1] > 0:
                    self.threat[fid] += self.cfg["threat_if_hit_us"]
                    self.cooldown[fid] = max(
                        self.cooldown[fid],
                        int(self.cfg["cooldown_if_hit_us"]),
                    )

        for pid, ids in enumerate(sabotages_by_player):
            if pid == self.player_id:
                continue
            for fid in set(ids):
                if 1 <= fid <= self.max_factory_id:
                    self.enemy_sabotage_events += 1
                    self.threat[fid] += self.cfg["threat_from_enemy_call"]
                    self.est_total_counts[fid] += 0.25

        if self.prev_all_lemons is None:
            return

        adjusted_growth = [0.0] * self.num_players
        for pid in range(self.num_players):
            adjusted_growth[pid] = (
                all_lemons[pid]
                - self.prev_all_lemons[pid]
                + self.sabotage_cost * len(sabotages_by_player[pid])
            )

        old_baselines = self.player_growth_baseline[:]

        for fid, destroyed in destroyed_factory_counts.items():
            if not (1 <= fid <= self.max_factory_id) or destroyed <= 0:
                continue

            shocks: List[float] = [0.0] * self.num_players
            total_shock = 0.0
            for pid in range(self.num_players):
                baseline = old_baselines[pid]
                shock = max(0.0, baseline - adjusted_growth[pid])
                shocks[pid] = shock
                total_shock += shock

            # Distribute the observed hit using who actually slowed down.
            if total_shock > 1e-9:
                for pid in range(self.num_players):
                    share = shocks[pid] / total_shock
                    self.player_id_exposure[pid][fid] += share * destroyed

        for pid in range(self.num_players):
            for fid in range(1, self.max_factory_id + 1):
                self.player_id_exposure[pid][fid] *= 0.90

        for pid in range(self.num_players):
            baseline = old_baselines[pid]
            if baseline <= 0.0:
                self.player_growth_baseline[pid] = adjusted_growth[pid]
            else:
                self.player_growth_baseline[pid] = 0.72 * \
                    baseline + 0.28 * adjusted_growth[pid]

    def _leader_target_bonus(
        self,
        fid: int,
        all_lemons: List[float],
    ) -> float:
        bonus = 0.0
        for pid in range(self.num_players):
            if pid == self.player_id:
                continue
            leader_weight = max(
                0.0, all_lemons[pid] - 0.55 * self.goal_lemons) / 180.0
            if leader_weight > 0.0:
                bonus += leader_weight * self.player_id_exposure[pid][fid]
        return bonus

    def _current_leader(self, all_lemons: List[float]) -> int:
        best_pid = -1
        best_lemons = float("-inf")
        for pid in range(self.num_players):
            if pid == self.player_id:
                continue
            if all_lemons[pid] > best_lemons:
                best_lemons = all_lemons[pid]
                best_pid = pid
        return best_pid

    def _repeat_penalty(
        self,
        fid: int,
        round_number: int,
        all_lemons: List[float],
    ) -> float:
        age = round_number - self.last_sabotage_round[fid]
        if age > self.cfg["repeat_window"]:
            return 0.0

        penalty = self.cfg["repeat_sabotage_penalty"] / max(1, age)
        leader_pid = self._current_leader(all_lemons)
        if leader_pid != -1:
            leader_exposure = self.player_id_exposure[leader_pid][fid]
            if leader_exposure < 1.25:
                penalty += self.cfg["failed_repeat_penalty"] / max(1, age)
        if self.last_sabotage_damage[fid] < 2.0:
            penalty += 0.5 * self.cfg["failed_repeat_penalty"] / max(1, age)
        return penalty

    def _should_stop_buying(
        self,
        your_lemons: float,
        your_factories: List[int],
        round_number: int,
        all_lemons: List[float],
    ) -> bool:
        own_prod = self._own_production(your_factories)
        if own_prod <= 0.0:
            return False

        if your_lemons + own_prod >= self.goal_lemons:
            return True

        future_buy_rounds = self._future_buy_rounds(round_number, all_lemons)
        if future_buy_rounds < 2:
            return True

        remaining = max(0.0, self.goal_lemons - your_lemons)
        if math.ceil(remaining / own_prod) <= 2 and your_lemons >= self._richest_other(all_lemons) - 40.0:
            return True

        return False

    def _sabotage_threshold(self, richest_other: float) -> float:
        gap = self.goal_lemons - richest_other
        if gap <= 90.0:
            return self.cfg["sabotage_threshold_emergency"]
        if gap <= 220.0:
            return self.cfg["sabotage_threshold_danger"]
        return self.cfg["sabotage_threshold_calm"]

    def _estimate_total_for_id(self, fid: int, own_count: int, richest_other: float) -> float:
        estimate = max(
            self.prior_counts[fid], self.est_total_counts[fid], float(own_count))

        # When the leader is close, assume the top IDs are even more concentrated.
        gap = self.goal_lemons - richest_other
        if gap <= 220.0 and fid >= self.max_factory_id - 2:
            estimate = max(estimate, own_count + 2.2 -
                           0.35 * (self.max_factory_id - fid))
        if gap <= 120.0 and fid >= self.max_factory_id - 1:
            estimate = max(estimate, own_count + 3.0 -
                           0.60 * (self.max_factory_id - fid))

        return estimate

    def _choose_sabotage_plan(
        self,
        your_lemons: float,
        your_factories: List[int],
        all_lemons: List[float],
        round_number: int,
    ) -> Tuple[List[int], List[int]]:
        if self._can_win_without_new_buys(your_lemons, your_factories):
            return [], []

        richest_other = self._richest_other(all_lemons)
        if richest_other <= 0.0:
            return [], []

        # Early sabotage is usually just tempo suicide. Wait for either a real
        # leader, repeated sabotage pressure, or a clear endgame emergency.
        gap = self.goal_lemons - richest_other
        strong_signal = (
            self.enemy_sabotage_events >= 2
            or max(self.threat[fid] for fid in self.focus_ids[:3]) >= 1.2
            or max(self.est_total_counts[fid] for fid in self.focus_ids[:2]) >= 4.0
        )
        if round_number < self.cfg["min_sabotage_round"] and gap > self.cfg["early_sabotage_gap"] and not strong_signal:
            return [], []
        if (
            richest_other < max(
                self.cfg["active_leader_floor"], your_lemons + self.cfg["leader_advantage_floor"])
            and gap > self.cfg["mid_sabotage_gap"]
            and not strong_signal
        ):
            return [], []
        if your_lemons < self.cfg["small_bankroll_skip_mult"] * self.buy_price and gap > self.cfg["small_bankroll_gap"]:
            return [], []

        threshold = self._sabotage_threshold(richest_other)
        existing_rounds = self._existing_factory_rounds(
            round_number, all_lemons)
        if existing_rounds <= 0:
            return [], []

        candidates: List[Tuple[float, int]] = []
        for fid in self.focus_ids:
            if fid <= self.max_factory_id - 4:
                if (
                    self.est_total_counts[fid] < self.cfg["low_id_sabotage_evidence"]
                    and self.threat[fid] < 1.6
                ):
                    continue
            own = your_factories[fid - 1]
            est_total = self._estimate_total_for_id(fid, own, richest_other)
            est_opp = max(0.0, est_total - own)
            if est_opp < 0.9:
                continue

            deny_value = est_opp * \
                (self.prod[fid] * existing_rounds + self.sell_price)
            self_cost = own * self.prod[fid] * \
                self.cfg["sabotage_self_cost_rounds"]
            pressure_penalty = 5.0 * self.threat[fid]
            score = deny_value - self.sabotage_cost - self_cost - pressure_penalty

            # In emergencies, strongly prefer sabotaging the very top IDs.
            if self.goal_lemons - richest_other <= 120.0:
                score += self.cfg["emergency_top_bonus"] * \
                    max(0, fid - (self.max_factory_id - 3))
            score += self.cfg["high_id_target_bonus"] * \
                max(0, fid - (self.max_factory_id - 3))
            score += self.cfg["leader_target_bonus_scale"] * \
                self._leader_target_bonus(fid, all_lemons)
            score -= self._repeat_penalty(fid, round_number, all_lemons)

            candidates.append((score, fid))

        candidates.sort(reverse=True)
        if not candidates or candidates[0][0] <= threshold:
            return [], []

        chosen_ids: List[int] = [candidates[0][1]]
        if len(candidates) >= 2:
            second_score, second_id = candidates[1]
            gap = self.goal_lemons - richest_other
            if (
                gap <= self.cfg["double_sabotage_gap"]
                or richest_other > your_lemons + self.cfg["double_sabotage_lead"]
            ) and second_score > threshold + self.cfg["double_sabotage_second_margin"]:
                chosen_ids.append(second_id)

        sells: List[int] = []
        sell_gain = 0.0
        for fid in chosen_ids:
            own = your_factories[fid - 1]
            if own > 0:
                sells.extend([fid] * own)
                sell_gain += own * self.sell_price

        if your_lemons + sell_gain < len(chosen_ids) * self.sabotage_cost:
            return [], []

        return sells, chosen_ids

    def _buy_candidates(self, round_number: int) -> List[int]:
        # Be greedier early if the lobby has shown little sabotage pressure.
        if round_number <= 4 and self.enemy_sabotage_events == 0:
            return self.focus_ids[:2]
        if self.enemy_sabotage_events <= 1 and max(self.threat[fid] for fid in self.focus_ids[:3]) < 0.9:
            return self.focus_ids[:3]
        if max(self.threat[fid] for fid in self.focus_ids[:4]) < self.cfg["deep_war_threat"]:
            return self.focus_ids[:4]
        return self.focus_ids[:5]

    def _buy_candidates_after_sabotage(self, sabotaged_ids: List[int]) -> List[int]:
        candidates: List[int] = []
        top_k = min(self.cfg["sabotage_rebuild_top_k"], len(self.focus_ids))
        for fid in self.focus_ids[:top_k]:
            if fid not in candidates:
                candidates.append(fid)
        for fid in sabotaged_ids:
            if fid in self.focus_ids[:top_k] and fid not in candidates:
                candidates.insert(0, fid)
        return candidates

    def _choose_buys(
        self,
        cash: float,
        your_factories: List[int],
        round_number: int,
        all_lemons: List[float],
        sabotaged_ids: List[int],
    ) -> List[int]:
        if cash < self.buy_price:
            return []

        if self._should_stop_buying(cash, your_factories, round_number, all_lemons):
            return []

        future_rounds = self._future_buy_rounds(round_number, all_lemons)
        if future_rounds < 2:
            return []

        budget = int(cash // self.buy_price)
        if budget <= 0:
            return []

        temp = your_factories[:]
        buys: List[int] = []
        candidate_ids = (
            self._buy_candidates_after_sabotage(sabotaged_ids)
            if sabotaged_ids
            else self._buy_candidates(round_number)
        )

        war_mode = bool(sabotaged_ids) or self.enemy_sabotage_events > 1
        soft_cap = self.cfg["war_soft_cap"] if war_mode else self.cfg["initial_soft_cap"]

        for _ in range(budget):
            best_id = -1
            best_score = float("-inf")

            for fid in candidate_ids:
                own = temp[fid - 1]
                gross = self.prod[fid] * future_rounds + \
                    self.sell_price - self.buy_price
                stack_penalty = self.cfg["own_stack_penalty"] * \
                    max(0, own - soft_cap)
                crowd_penalty = self.cfg["crowd_penalty"] * \
                    max(0.0, self.est_total_counts[fid] - own)
                risk_penalty = (
                    self.cfg["threat_penalty"] * self.threat[fid]
                    + self.cfg["cooldown_penalty"] * self.cooldown[fid]
                )
                safe_bonus = self.cfg["safe_bonus"] * \
                    self.prod[fid] if self.threat[fid] < 0.9 else 0.0
                low_id_penalty = 0.0
                if fid <= self.max_factory_id - 4:
                    low_id_penalty += self.cfg["low_id_penalty"] * \
                        (self.max_factory_id - 3 - fid)
                    if future_rounds <= 3:
                        low_id_penalty += self.cfg["late_low_id_penalty"] * (
                            self.max_factory_id - 3 - fid)
                same_round_cycle_bonus = 0.0
                if fid in sabotaged_ids:
                    # New purchases cannot be sabotaged this round, so cycling back
                    # into a freshly-cleared top ID is sometimes exactly right.
                    same_round_cycle_bonus = self.cfg["same_round_cycle_bonus"] * \
                        self.prod[fid]

                score = (
                    gross
                    - stack_penalty
                    - crowd_penalty
                    - risk_penalty
                    - low_id_penalty
                    + safe_bonus
                    + same_round_cycle_bonus
                )

                if score > best_score:
                    best_score = score
                    best_id = fid

            if best_id == -1 or best_score <= 0.0:
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
        self._update_beliefs(
            all_lemons, destroyed_factory_counts, sabotages_by_player)

        if self.prev_all_lemons is not None:
            leader_pid = self._current_leader(all_lemons)
            self.last_round_leader = leader_pid
            if leader_pid != -1:
                self.last_round_leader_growth = all_lemons[leader_pid] - \
                    self.prev_all_lemons[leader_pid]

        sells, sabotages = self._choose_sabotage_plan(
            your_lemons,
            your_factories,
            all_lemons,
            round_number,
        )

        post_sell_factories = your_factories[:]
        for fid in sells:
            idx = fid - 1
            if post_sell_factories[idx] > 0:
                post_sell_factories[idx] -= 1

        cash = your_lemons + len(sells) * self.sell_price - \
            len(sabotages) * self.sabotage_cost
        if cash < 0.0:
            sells = []
            sabotages = []
            post_sell_factories = your_factories[:]
            cash = your_lemons

        buys = self._choose_buys(
            cash,
            post_sell_factories,
            round_number,
            all_lemons,
            sabotages,
        )

        max_affordable = int(cash // self.buy_price)
        if len(buys) > max_affordable:
            buys = buys[:max_affordable]

        for fid in sabotages:
            self.last_sabotage_round[fid] = round_number

        self.prev_own_factories = your_factories[:]
        self.prev_all_lemons = all_lemons[:]
        return buys, sells, sabotages


SubmissionPlayer = SubmissionStrategy
