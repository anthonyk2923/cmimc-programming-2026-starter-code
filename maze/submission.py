from typing import Any, List, Tuple


STYLE_SPARSE = 0
STYLE_DENSE = 1
STYLE_CLUSTER = 2
STYLE_HALIN = 3

BOT_WAIT = (4, 8, 8, 8)
BOT_TARGET_DEPTH = (12, 5, 5, 12)
BOT_STOP_THRESHOLD = (50, 50, 50, 50)
BOT_LATE_STEP = (220, 220, 220, 220)
BOT_LATE_THRESHOLD = (3, 3, 3, 3)

GHOST_TARGET_DEPTH = (12, 5, 5, 8)
GHOST_GOOD_SLOT = (25, 25, 25, 25)
GHOST_DEEP_EXTRA = (2, 2, 2, 3)
GHOST_DEEP_STOP = (5, 5, 5, 5)
GHOST_LATE_STEP = (100, 70, 30, 60)
GHOST_SAMPLE_LATE_ONLY = True


def _bit_get(blob, idx: int) -> int:
    return (blob[idx >> 3] >> (idx & 7)) & 1


def _bit_set(blob: bytearray, idx: int) -> None:
    blob[idx >> 3] |= 1 << (idx & 7)


def _style(root_degree: int) -> int:
    if root_degree <= 2:
        return STYLE_SPARSE
    if root_degree >= 8:
        return STYLE_DENSE
    if root_degree >= 5:
        return STYLE_CLUSTER
    return STYLE_HALIN


def _bot_plan(style: int) -> Tuple[int, int]:
    return BOT_WAIT[style], BOT_TARGET_DEPTH[style]


def _ghost_late_step(style: int) -> int:
    return GHOST_LATE_STEP[style]


def _ghost_deep_extra(style: int) -> int:
    return GHOST_DEEP_EXTRA[style]


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
        wait, _ = _bot_plan(style)
        state[15] = wait
        state[16] = style

    style = state[16]
    _bit_set(state, pos)

    depth = state[13]
    mode = state[14]

    if mode == 2:
        return -1, bytes(state)

    if mode == 0 and state[15] > 0:
        state[15] -= 1
        return -1, bytes(state)

    state[14] = 1

    if has_slot and (
        slot_coins >= BOT_STOP_THRESHOLD[style]
        or (depth >= BOT_TARGET_DEPTH[style] and slot_coins >= 8)
        or (step >= BOT_LATE_STEP[style] and slot_coins >= BOT_LATE_THRESHOLD[style])
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
    # [17] mode (0 explore, 1 chosen, 2 sample)
    # [18] target_depth
    # [19] style
    state = bytearray(20) if data is None else bytearray(data)

    if data is None:
        style = _style(len(neighbors))
        _, target_depth = _bot_plan(style)
        state[18] = target_depth
        state[19] = style

    _bit_set(state, pos)

    depth = state[16]
    mode = state[17]
    target_depth = state[18]
    style = state[19]

    late_step = _ghost_late_step(style)

    if mode == 1:
        return -1, bytes(state)

    if mode == 2:
        if slot_coins > 0 or step >= late_step + 120:
            state[17] = 1
            return -1, bytes(state)
        state[17] = 0

    if has_slot and depth >= target_depth and slot_coins >= GHOST_GOOD_SLOT[style]:
        state[17] = 1
        return -1, bytes(state)

    if has_slot and depth >= target_depth + _ghost_deep_extra(style) and slot_coins >= GHOST_DEEP_STOP[style]:
        state[17] = 1
        return -1, bytes(state)

    if has_slot and not GHOST_SAMPLE_LATE_ONLY and depth >= target_depth and slot_coins == 0:
        state[17] = 2
        return -1, bytes(state)

    if has_slot and step >= late_step:
        if slot_coins > 0:
            state[17] = 1
        else:
            state[17] = 2
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
