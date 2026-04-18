from typing import Any, List, Tuple


def _bit_get(blob, idx: int) -> int:
    return (blob[idx >> 3] >> (idx & 7)) & 1


def _bit_set(blob: bytearray, idx: int) -> None:
    blob[idx >> 3] |= 1 << (idx & 7)


def _plan(root_degree: int) -> Tuple[int, int]:
    if root_degree >= 5:
        return 8, 5
    if root_degree >= 3:
        return 16, 8
    return 28, 12


BASE_THRESHOLD = 50
MIN_THRESHOLD = 10
BOT_FALLBACK_STEP = 220
BOT_FALLBACK_DEPTH = 13

GHOST_GOOD_SLOT = 25
GHOST_LATE_STEP = 40
GHOST_DEEP_EXTRA = 2


def SubmissionBot(
    step: int,
    total_steps: int,
    pos: int,
    last_pos: int,
    neighbors: List[int],
    has_slot: bool,
    slot_coins: int,
    data: Any,
) -> Tuple[int, Any]:
    # [0-12] visited
    # [13] depth
    # [14] mode (0 wait, 1 search, 2 settled)
    # [15] wait
    state = bytearray(16) if data is None else bytearray(data)

    if data is None:
        wait, _ = _plan(len(neighbors))
        state[15] = wait

    _bit_set(state, pos)

    depth = state[13]
    mode = state[14]

    if mode == 2:
        return -1, bytes(state)

    if mode == 0 and state[15] > 0:
        state[15] -= 1
        return -1, bytes(state)

    state[14] = 1

    threshold = BASE_THRESHOLD - step // 25
    if threshold < MIN_THRESHOLD:
        threshold = MIN_THRESHOLD

    if has_slot and (
        slot_coins >= threshold
        or (depth >= BOT_FALLBACK_DEPTH and slot_coins >= 8)
        or (step >= BOT_FALLBACK_STEP and slot_coins >= 3)
    ):
        state[14] = 2
        return -1, bytes(state)

    # 1. Prefer unvisited
    for n in neighbors:
        if not _bit_get(state, n):
            state[13] = depth + 1 if depth < 255 else 255
            return n, bytes(state)

    # 2. Avoid immediate ping-pong
    for n in neighbors:
        if n != last_pos:
            return n, bytes(state)

    # 3. Backtrack only if forced
    if last_pos != -1 and last_pos in neighbors:
        state[13] = depth - 1 if depth else 0
        return last_pos, bytes(state)

    return -1, bytes(state)


def SubmissionGhost(
    step: int,
    total_steps: int,
    pos: int,
    last_pos: int,
    neighbors: List[int],
    has_slot: bool,
    slot_coins: int,
    data: Any,
) -> Tuple[int, Any]:
    # [0-15] visited
    # [16] depth
    # [17] mode (0 explore, 1 chosen)
    # [18] target_depth
    # [19] best slot seen so far, capped to 255
    state = bytearray(20) if data is None else bytearray(data)

    if data is None:
        _, target_depth = _plan(len(neighbors))
        state[18] = target_depth

    _bit_set(state, pos)

    depth = state[16]
    mode = state[17]
    target_depth = state[18]

    if has_slot:
        capped = 255 if slot_coins > 255 else slot_coins
        if capped > state[19]:
            state[19] = capped

    if mode == 1:
        return -1, bytes(state)

    # Good slot at target depth
    if has_slot and depth >= target_depth and slot_coins >= GHOST_GOOD_SLOT:
        state[17] = 1
        return -1, bytes(state)

    # If we've gone deeper than planned, accept weaker slots
    if has_slot and depth >= target_depth + GHOST_DEEP_EXTRA and slot_coins >= 5:
        state[17] = 1
        return -1, bytes(state)

    # Late-game force commit
    if has_slot and step >= GHOST_LATE_STEP:
        state[17] = 1
        return -1, bytes(state)

    # 1. Prefer unvisited
    for n in neighbors:
        if not _bit_get(state, n):
            state[16] = depth + 1 if depth < 255 else 255
            return n, bytes(state)

    # 2. Avoid ping-pong
    for n in neighbors:
        if n != last_pos:
            return n, bytes(state)

    # 3. Backtrack
    if last_pos != -1 and last_pos in neighbors:
        state[16] = depth - 1 if depth else 0
        return last_pos, bytes(state)

    return -1, bytes(state)
