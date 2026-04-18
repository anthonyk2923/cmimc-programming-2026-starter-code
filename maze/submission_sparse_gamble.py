from typing import Any, List, Tuple


def _bit_get(blob, idx: int) -> int:
    return (blob[idx >> 3] >> (idx & 7)) & 1


def _bit_set(blob: bytearray, idx: int) -> None:
    blob[idx >> 3] |= 1 << (idx & 7)


def _style(root_degree: int) -> int:
    if root_degree <= 2:
        return 0
    if root_degree >= 8:
        return 1
    if root_degree >= 5:
        return 2
    return 3


def _bot_plan(style: int) -> Tuple[int, int, int]:
    if style == 0:
        return 90, 18, 45
    if style == 3:
        return 30, 10, 28
    if style == 2:
        return 10, 6, 22
    return 0, 2, 18


def _ghost_plan(style: int) -> Tuple[int, int, int]:
    if style == 0:
        return 18, 120, 1
    if style == 3:
        return 9, 80, 4
    if style == 2:
        return 6, 60, 6
    return 3, 35, 10


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
    if step == 1:
        last_pos = -1

    # [0-12] visited
    # [13] depth
    # [14] mode (0 wait, 1 search, 2 settled)
    # [15] wait
    # [16] style
    state = bytearray(17) if data is None else bytearray(data)

    if data is None:
        style = _style(len(neighbors))
        wait, _, _ = _bot_plan(style)
        state[15] = wait
        state[16] = style

    style = state[16]
    wait, target_depth, threshold = _bot_plan(style)

    _bit_set(state, pos)
    depth = state[13]
    mode = state[14]

    if mode == 2:
        return -1, bytes(state)

    if mode == 0 and state[15] > 0:
        state[15] -= 1
        return -1, bytes(state)

    state[14] = 1

    late_threshold = 4 if style == 0 else 3
    late_step = 280 if style == 0 else 180

    if has_slot and (
        slot_coins >= threshold
        or (depth >= target_depth and slot_coins >= 8)
        or (step >= late_step and slot_coins >= late_threshold)
    ):
        state[14] = 2
        return -1, bytes(state)

    for n in neighbors:
        if not _bit_get(state, n):
            state[13] = depth + 1 if depth < 255 else 255
            return n, bytes(state)

    for n in neighbors:
        if n != last_pos:
            return n, bytes(state)

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
    if step == 1:
        last_pos = -1

    # [0-15] visited
    # [16] depth
    # [17] mode (0 explore, 1 chosen)
    # [18] target_depth
    # [19] late_step
    # [20] accept_small
    state = bytearray(21) if data is None else bytearray(data)

    if data is None:
        style = _style(len(neighbors))
        target_depth, late_step, accept_small = _ghost_plan(style)
        state[18] = target_depth
        state[19] = late_step
        state[20] = accept_small

    _bit_set(state, pos)
    depth = state[16]
    mode = state[17]
    target_depth = state[18]
    late_step = state[19]
    accept_small = state[20]

    if mode == 1:
        return -1, bytes(state)

    if has_slot and depth >= target_depth and slot_coins >= 20:
        state[17] = 1
        return -1, bytes(state)

    if has_slot and depth >= target_depth + 2 and slot_coins >= accept_small:
        state[17] = 1
        return -1, bytes(state)

    if has_slot and step >= late_step:
        state[17] = 1
        return -1, bytes(state)

    for n in neighbors:
        if not _bit_get(state, n):
            state[16] = depth + 1 if depth < 255 else 255
            return n, bytes(state)

    for n in neighbors:
        if n != last_pos:
            return n, bytes(state)

    if last_pos != -1 and last_pos in neighbors:
        state[16] = depth - 1 if depth else 0
        return last_pos, bytes(state)

    return -1, bytes(state)
