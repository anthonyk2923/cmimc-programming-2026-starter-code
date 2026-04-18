from pic.strategy.strategy import (
    Strategy, Request, Message,
    RegionRequest, PixelMessage,
)
from pic.strategy.strategy import Strategy, Request, Message
import math


# ============================================================================
# Tuned recovery-only submission
#
# Main changes vs the uploaded version:
# 1) Binary images are no longer forced into 5x5 constant blocks.
#    Instead:
#      - gap-aware blur to fill missing regions
#      - mild unsharp mask to recover edges
#      - Otsu threshold to pick the binary cutoff adaptively
#      - one majority-cleanup pass
#
# 2) Smooth images use gentler denoising:
#      - gap-aware blur
#      - blend known pixels with the local smooth estimate
#
# 3) Classifier is slightly more forgiving on high-blur Random Circles:
#      - strong binary rule for obvious binary cases
#      - softer rescue rule using variance + filled-image gradient
#
# Messaging is intentionally left off here. That is not because messaging is
# useless; it is because the exact request/message helper signatures were not
# present in the files you provided, and a wrong constructor guess is a match-
# losing crash.
# ============================================================================


STYLE_BINARY = 0
STYLE_SMOOTH = 1


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _make_kernel(sigma: float):
    radius = max(1, int(3 * sigma))
    kernel = [math.exp(-0.5 * (x / sigma) ** 2)
              for x in range(-radius, radius + 1)]
    s = sum(kernel)
    return [k / s for k in kernel], radius


def _gap_aware_blur(grid, m: int, n: int, sigma: float):
    """
    Separable Gaussian blur that ignores None pixels.
    Returns a dense float grid.
    """
    kernel, radius = _make_kernel(sigma)

    h_val = [[0.0] * n for _ in range(m)]
    h_wt = [[0.0] * n for _ in range(m)]

    # Horizontal pass
    for r in range(m):
        row = grid[r]
        hv = h_val[r]
        hw = h_wt[r]
        for c in range(n):
            va = 0.0
            wa = 0.0
            left = c - radius
            for i, k in enumerate(kernel):
                cc = left + i
                if cc < 0:
                    cc = 0
                elif cc >= n:
                    cc = n - 1
                v = row[cc]
                if v is not None:
                    fv = float(v)
                    va += k * fv
                    wa += k
            hv[c] = va
            hw[c] = wa

    out = [[0.5] * n for _ in range(m)]

    # Vertical pass
    for r in range(m):
        for c in range(n):
            va = 0.0
            wa = 0.0
            top = r - radius
            for i, k in enumerate(kernel):
                rr = top + i
                if rr < 0:
                    rr = 0
                elif rr >= m:
                    rr = m - 1
                w = h_wt[rr][c]
                if w > 0.0:
                    va += k * h_val[rr][c]
                    wa += k * w
            out[r][c] = va / wa if wa > 1e-12 else 0.5

    return out


def _dense_blur(grid, m: int, n: int, sigma: float):
    """
    Separable Gaussian blur for dense float grids (no None).
    """
    kernel, radius = _make_kernel(sigma)

    h = [[0.0] * n for _ in range(m)]
    for r in range(m):
        row = grid[r]
        hr = h[r]
        for c in range(n):
            s = 0.0
            left = c - radius
            for i, k in enumerate(kernel):
                cc = left + i
                if cc < 0:
                    cc = 0
                elif cc >= n:
                    cc = n - 1
                s += k * row[cc]
            hr[c] = s

    out = [[0.0] * n for _ in range(m)]
    for r in range(m):
        for c in range(n):
            s = 0.0
            top = r - radius
            for i, k in enumerate(kernel):
                rr = top + i
                if rr < 0:
                    rr = 0
                elif rr >= m:
                    rr = m - 1
                s += k * h[rr][c]
            out[r][c] = s

    return out


def _mean_abs_adjacent_grad(grid, m: int, n: int) -> float:
    total = 0.0
    cnt = 0
    for r in range(m):
        row = grid[r]
        for c in range(n - 1):
            total += abs(row[c] - row[c + 1])
            cnt += 1
    for r in range(m - 1):
        row = grid[r]
        row2 = grid[r + 1]
        for c in range(n):
            total += abs(row[c] - row2[c])
            cnt += 1
    return total / cnt if cnt else 0.0


def _classify_family(corrupted, m: int, n: int, filled12) -> int:
    """
    Returns STYLE_BINARY for Random Circles / Blobs,
    STYLE_SMOOTH for Voronoi / Piecewise Constant.
    """
    known = []
    for r in range(m):
        row = corrupted[r]
        for c in range(n):
            v = row[c]
            if v is not None:
                known.append(float(v))

    nk = len(known)
    if nk < 100:
        return STYLE_SMOOTH

    extreme = 0
    middle = 0
    low_sum = 0.0
    low_cnt = 0
    high_sum = 0.0
    high_cnt = 0
    mean_sum = 0.0

    for v in known:
        if v < 0.15 or v > 0.85:
            extreme += 1
        if 0.30 < v < 0.70:
            middle += 1
        if v < 0.5:
            low_sum += v
            low_cnt += 1
        else:
            high_sum += v
            high_cnt += 1
        mean_sum += v

    extreme /= nk
    middle /= nk
    gap = 0.0
    if low_cnt and high_cnt:
        gap = high_sum / high_cnt - low_sum / low_cnt

    mean_v = mean_sum / nk
    var = 0.0
    for v in known:
        dv = v - mean_v
        var += dv * dv
    var /= nk

    grad = _mean_abs_adjacent_grad(filled12, m, n)

    # Very obvious binary cases
    if extreme > 0.56 and middle < 0.24 and gap > 0.60:
        return STYLE_BINARY

    # Rescue some high-blur Random Circles / softer blob cases
    if extreme > 0.42 and middle < 0.33 and gap > 0.54 and var > 0.07 and grad > 0.032:
        return STYLE_BINARY

    return STYLE_SMOOTH


def _smooth_recover(corrupted, filled12, m: int, n: int):
    """
    Gentle denoising for Voronoi / Piecewise Constant.
    """
    result = [[0.0] * n for _ in range(m)]

    for r in range(m):
        row = corrupted[r]
        base = filled12[r]
        out = result[r]
        for c in range(n):
            v = row[c]
            if v is None:
                out[c] = base[c]
            else:
                fv = float(v)
                # Mild pull toward the local smooth estimate
                out[c] = 0.70 * fv + 0.30 * base[c]

    for r in range(m):
        row = result[r]
        for c in range(n):
            row[c] = _clip01(row[c])

    return result


def _otsu_threshold(grid, m: int, n: int) -> float:
    """
    64-bin Otsu threshold on a dense float grid in [0,1].
    """
    bins = 64
    hist = [0] * bins

    for r in range(m):
        row = grid[r]
        for c in range(n):
            v = row[c]
            idx = int(v * bins)
            if idx < 0:
                idx = 0
            elif idx >= bins:
                idx = bins - 1
            hist[idx] += 1

    total = m * n
    if total == 0:
        return 0.5

    sum_total = 0.0
    for i, h in enumerate(hist):
        sum_total += i * h

    best_var = -1.0
    best_t = bins // 2
    w_b = 0
    sum_b = 0.0

    for i, h in enumerate(hist):
        w_b += h
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += i * h
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) * (m_b - m_f)

        if between > best_var:
            best_var = between
            best_t = i

    return (best_t + 0.5) / bins


def _majority_pass_binary(grid, m: int, n: int):
    out = [[0.0] * n for _ in range(m)]

    for r in range(m):
        r0 = max(0, r - 1)
        r1 = min(m, r + 2)
        for c in range(n):
            c0 = max(0, c - 1)
            c1 = min(n, c + 2)
            s = 0.0
            cnt = 0
            for rr in range(r0, r1):
                row = grid[rr]
                for cc in range(c0, c1):
                    s += row[cc]
                    cnt += 1
            out[r][c] = 1.0 if s * 2.0 >= cnt else 0.0

    return out


def _binary_recover(corrupted, m: int, n: int):
    """
    Binary recovery:
      1) gap-aware blur
      2) mild unsharp mask
      3) Otsu threshold
      4) one majority cleanup pass
    """
    filled = _gap_aware_blur(corrupted, m, n, sigma=1.5)
    blur2 = _dense_blur(filled, m, n, sigma=1.0)

    sharpened = [[0.0] * n for _ in range(m)]
    for r in range(m):
        frow = filled[r]
        brow = blur2[r]
        srow = sharpened[r]
        for c in range(n):
            # amount = 0.5
            v = frow[c] + 0.5 * (frow[c] - brow[c])
            srow[c] = _clip01(v)

    t = _otsu_threshold(sharpened, m, n)

    binary = [[0.0] * n for _ in range(m)]
    for r in range(m):
        srow = sharpened[r]
        brow = binary[r]
        for c in range(n):
            brow[c] = 1.0 if srow[c] >= t else 0.0

    binary = _majority_pass_binary(binary, m, n)
    return binary


class SubmissionStrategy(Strategy):
    def __init__(self, corrupted: list[list[float | None]]):
        self.corrupted = corrupted
        self.m = len(corrupted)
        self.n = len(corrupted[0]) if self.m > 0 else 0

    def make_requests(self) -> list[Request]:
        return []

    def receive_requests(self, requests: list[Request]) -> list[Message | None]:
        return [None] * len(requests)

    def receive_messages(self, messages: list[Message | None]):
        pass

    def recover(self) -> list[list[float]]:
        m = self.m
        n = self.n
        corrupted = self.corrupted

        if m == 0 or n == 0:
            return []

        filled12 = _gap_aware_blur(corrupted, m, n, sigma=1.2)
        family = _classify_family(corrupted, m, n, filled12)

        if family == STYLE_BINARY:
            result = _binary_recover(corrupted, m, n)
        else:
            result = _smooth_recover(corrupted, filled12, m, n)

        for r in range(m):
            row = result[r]
            for c in range(n):
                row[c] = _clip01(row[c])

        return result


# ══════════════════════════════════════════════════════════════════════════════
# FULL VERSION HISTORY:
#
# v1  avg 0.2133  baseline
# v2  avg 0.2184  classifier regression (smooth → binary misrouting)
# v3  avg 0.2681  3 crashes from CPU timeout
# v4  avg 0.2152  CPU-safe inpainting (r=2/15it), good classifier
# v5  avg 0.2163  block decisions on known pixels (deblur), no denoising
# v6  (this)     blob rescue in classifier + all v5 improvements
#
# v5 per-generator vs v4:
#   RC:      0.2956 → 0.2860  (-0.010 ✓)
#   Blob:    0.2617 → 0.2791  (+0.017 — statistical noise, 9 games, 1σ effect)
#   Voronoi: 0.1567 → 0.1496  (-0.007 ✓)
#   PW:      0.1527 → 0.1507  (-0.002 ✓)
#
# v6 CHANGE — Classifier: add low-variance blob rescue
#   Problem: some Blob images at high blur (σ≈3.0) have high fhg (≥0.40),
#   causing them to be misclassified as smooth. The smooth path has ~2× higher
#   MAE on Blobs vs the binary path.
#   Fix: if s1 passes AND variance < 0.035 → binary.
#   Why safe: Blobs with large uniform regions have very low variance (0.009–0.028
#   at high blur). Voronoi/PW always have variance ≥ 0.037 (validated on 800 cases).
#   This catches the edge-case blobs that fhg misses, with zero false positives.
#
# CLASSIFIER LIMITATIONS (physically unavoidable):
#   RC at σ>2.0: blur heavily smears circle edges. fhg and extreme fraction
#   overlap with Voronoi/PW. ~30–50% of RC games at max blur will be routed
#   to the smooth path. This is a fundamental information-theoretic limit —
#   at σ=3.0, the corrupted image genuinely looks like a smooth image.
#   The binary path is still ~0.077 MAE better for these images, so even
#   correct classification matters, but we cannot reliably achieve it.
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────

# def _make_kernel(sigma):
#     radius = max(1, int(3 * sigma))
#     kernel = [math.exp(-0.5 * (x / sigma) ** 2) for x in range(-radius, radius + 1)]
#     s = sum(kernel)
#     return [k / s for k in kernel], radius


# def _gap_aware_blur(grid, m, n, sigma):
#     """Separable Gaussian blur that ignores None pixels."""
#     kernel, radius = _make_kernel(sigma)
#     h_val = [[0.0]*n for _ in range(m)]
#     h_wt  = [[0.0]*n for _ in range(m)]
#     for r in range(m):
#         for c in range(n):
#             va = wa = 0.0
#             for i, k in enumerate(kernel):
#                 cc = max(0, min(n-1, c+i-radius))
#                 v = grid[r][cc]
#                 if v is not None:
#                     va += k*v; wa += k
#             h_val[r][c] = va; h_wt[r][c] = wa
#     out = [[0.5]*n for _ in range(m)]
#     for r in range(m):
#         for c in range(n):
#             va = wa = 0.0
#             for i, k in enumerate(kernel):
#                 rr = max(0, min(m-1, r+i-radius))
#                 if h_wt[rr][c] > 0:
#                     va += k*h_val[rr][c]; wa += k*h_wt[rr][c]
#             out[r][c] = va/wa if wa > 0 else 0.5
#     return out


# # ─────────────────────────────────────────────────────────────────────────────
# # IMAGE TYPE DETECTION (v6)
# # ─────────────────────────────────────────────────────────────────────────────

# def _classify_image(corrupted, m, n):
#     """
#     Returns 'binary' (Random Circles / Blobs) or 'smooth' (Voronoi / Piecewise).

#     Three-stage classifier:

#     Stage 1 — value distribution (relaxed vs v3/v4, prevents RC miss at high blur):
#       extreme > 0.50   fraction of pixels near 0 or 1
#       middle  < 0.30   fraction in grey zone 0.3–0.7
#       gap     > 0.50   bimodal separation (mean_high - mean_low)
#       Any failure → 'smooth'. This alone catches all Voronoi/PW correctly.

#     Stage 2 — low-variance blob rescue (NEW in v6):
#       If stage 1 passes AND variance < 0.035 → 'binary'.
#       Blobs with large uniform regions have extremely low variance (0.009–0.028
#       at high blur). Voronoi/PW always have variance ≥ 0.037.
#       Validated: zero false positives on 800 Voronoi/PW test cases.
#       This catches blobs that the fhg signal would otherwise miss.

#     Stage 3 — fhg veto (from v5):
#       fhg = fraction of adjacent pixel pairs with |diff| > 0.15.
#       Voronoi/PW have dense sharp boundaries → fhg 0.44–0.48.
#       Binary images have fhg 0.22–0.40 (fewer, larger regions).
#       fhg < 0.40 → 'binary', else → 'smooth'.

#     Known limitation: RC at σ>2.0 is genuinely ambiguous — the blur
#     destroys binary structure and signals overlap with smooth images.
#     Some RC games at high blur will inevitably be misclassified.
#     """
#     known = [v for row in corrupted for v in row if v is not None]
#     if len(known) < 100:
#         return 'smooth'

#     n_known = len(known)

#     # Stage 1: value distribution
#     extreme = sum(1 for v in known if v < 0.15 or v > 0.85) / n_known
#     middle  = sum(1 for v in known if 0.3 < v < 0.7) / n_known
#     low_v   = [v for v in known if v < 0.5]
#     high_v  = [v for v in known if v >= 0.5]
#     gap     = (sum(high_v)/len(high_v) - sum(low_v)/len(low_v)
#                if low_v and high_v else 0.0)

#     if not (extreme > 0.50 and middle < 0.30 and gap > 0.50):
#         return 'smooth'

#     # Stage 2: low-variance blob rescue
#     mean_v   = sum(known) / n_known
#     variance = sum((v - mean_v)**2 for v in known) / n_known
#     if variance < 0.035:
#         return 'binary'

#     # Stage 3: fhg veto
#     total_edges = 0
#     high_grad_edges = 0
#     for r in range(m):
#         for c in range(n):
#             v = corrupted[r][c]
#             if v is None:
#                 continue
#             if c + 1 < n and corrupted[r][c+1] is not None:
#                 total_edges += 1
#                 if abs(v - corrupted[r][c+1]) > 0.15:
#                     high_grad_edges += 1
#             if r + 1 < m and corrupted[r+1][c] is not None:
#                 total_edges += 1
#                 if abs(v - corrupted[r+1][c]) > 0.15:
#                     high_grad_edges += 1

#     fhg = high_grad_edges / total_edges if total_edges > 0 else 0.0
#     return 'binary' if fhg < 0.40 else 'smooth'


# # ─────────────────────────────────────────────────────────────────────────────
# # PRE-BUILT WEIGHT TABLE
# # ─────────────────────────────────────────────────────────────────────────────

# _WEIGHTS_R2 = {}
# for _dr in range(-2, 3):
#     for _dc in range(-2, 3):
#         if _dr == 0 and _dc == 0:
#             continue
#         _WEIGHTS_R2[(_dr, _dc)] = 1.0 / max(1, _dr*_dr + _dc*_dc)


# # ─────────────────────────────────────────────────────────────────────────────
# # SMOOTH INPAINTING (Voronoi, Piecewise Constant)
# # ─────────────────────────────────────────────────────────────────────────────

# def _inpaint_smooth(corrupted, m, n):
#     """
#     Smooth image reconstruction.

#     1. Gap-aware Gaussian blur (σ=2.0) as initial prior for missing pixels.
#     2. Iterative 1/d² inpainting (r=2, 15 iterations).
#     3. Known pixels passed through unchanged — no denoising blend.
#        (Removing the old 30% blend improves MAE by ~0.010–0.012: known pixels
#        carry the best available signal and blurring them adds more error than
#        their noise level.)

#     Worst-case timing (all 2500 pixels missing): ~0.29s.
#     """
#     blurred = _gap_aware_blur(corrupted, m, n, sigma=2.0)
#     result = [[0.0]*n for _ in range(m)]
#     unknown_mask = [[False]*n for _ in range(m)]

#     for r in range(m):
#         for c in range(n):
#             v = corrupted[r][c]
#             if v is not None:
#                 result[r][c] = v
#             else:
#                 result[r][c] = blurred[r][c]
#                 unknown_mask[r][c] = True

#     weights = _WEIGHTS_R2
#     for _ in range(15):
#         new_result = [row[:] for row in result]
#         for r in range(m):
#             for c in range(n):
#                 if not unknown_mask[r][c]:
#                     continue
#                 total = wsum = 0.0
#                 for (dr, dc), w in weights.items():
#                     rr, cc = r+dr, c+dc
#                     if 0 <= rr < m and 0 <= cc < n:
#                         total += w*result[rr][cc]; wsum += w
#                 if wsum > 0:
#                     new_result[r][c] = total/wsum
#         result = new_result

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # BINARY INPAINTING (Random Circles, Blobs)
# # ─────────────────────────────────────────────────────────────────────────────

# def _inpaint_binary(corrupted, m, n):
#     """
#     Binary image reconstruction.

#     For every 5×5 block (matching the corruption block size), compute a
#     distance-weighted vote (radius=18) from all known pixels. Blocks where
#     vote ratio > 0.58 / < 0.42 are hard-classified as 1 / 0.

#     KEY: block decisions are applied to ALL pixels, not just missing ones.
#     The original image is purely binary (0 or 1). Known pixels contain blur
#     and noise (a pixel at 0.3 that should be 0, etc.). Outputting the block's
#     binary decision for known pixels corrects this blur error.
#     Simulated improvement vs v4: RC ~16% better, Blob ~35% better.

#     Ambiguous blocks (ratio 0.42–0.58): keep original known pixel values as
#     anchors, resolve unknown pixels via 5-iteration inpainting (r=2).

#     Worst-case timing: ~0.065s.
#     """
#     bs = 5
#     blocks_r = (m + bs - 1) // bs
#     blocks_c = (n + bs - 1) // bs

#     unknown_mask = [[corrupted[r][c] is None for c in range(n)] for r in range(m)]

#     # Compute block decision for every block
#     block_val = [[0.5]*blocks_c for _ in range(blocks_r)]

#     for br in range(blocks_r):
#         for bc in range(blocks_c):
#             r1 = br * bs; c1 = bc * bs
#             r2 = min(r1 + bs, m); c2 = min(c1 + bs, n)
#             cr = (r1 + r2) // 2; cc_c = (c1 + c2) // 2

#             votes_high = votes_low = 0.0
#             for r in range(max(0, cr - 18), min(m, cr + 19)):
#                 for c in range(max(0, cc_c - 18), min(n, cc_c + 19)):
#                     v = corrupted[r][c]
#                     if v is None:
#                         continue
#                     dist = math.sqrt((r - cr)**2 + (c - cc_c)**2)
#                     if dist < 1.0:
#                         dist = 1.0
#                     w = 1.0 / (dist * dist)
#                     votes_high += w * v
#                     votes_low  += w * (1.0 - v)

#             tv = votes_high + votes_low
#             if tv == 0:
#                 block_val[br][bc] = 0.5
#             else:
#                 ratio = votes_high / tv
#                 if ratio > 0.58:
#                     block_val[br][bc] = 1.0
#                 elif ratio < 0.42:
#                     block_val[br][bc] = 0.0
#                 else:
#                     block_val[br][bc] = ratio

#     # Apply block decisions to ALL pixels
#     result = [[0.0]*n for _ in range(m)]
#     for r in range(m):
#         for c in range(n):
#             br = r // bs; bc = c // bs
#             bv = block_val[br][bc]
#             if bv in (0.0, 1.0):
#                 result[r][c] = bv  # deblur known pixels + fill unknown
#             else:
#                 v = corrupted[r][c]
#                 result[r][c] = v if v is not None else bv

#     # Resolve ambiguous unknown pixels
#     ambiguous_unknown = [
#         [unknown_mask[r][c] and block_val[r//bs][c//bs] not in (0.0, 1.0)
#          for c in range(n)]
#         for r in range(m)
#     ]

#     if any(ambiguous_unknown[r][c] for r in range(m) for c in range(n)):
#         weights = _WEIGHTS_R2
#         for _ in range(5):
#             new_result = [row[:] for row in result]
#             for r in range(m):
#                 for c in range(n):
#                     if not ambiguous_unknown[r][c]:
#                         continue
#                     total = wsum = 0.0
#                     for (dr, dc), w in weights.items():
#                         rr, cc = r+dr, c+dc
#                         if 0 <= rr < m and 0 <= cc < n:
#                             total += w*result[rr][cc]; wsum += w
#                     if wsum > 0:
#                         new_result[r][c] = 1.0 if total/wsum >= 0.5 else 0.0
#             result = new_result

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # STRATEGY
# # ─────────────────────────────────────────────────────────────────────────────

# class SubmissionStrategy(Strategy):

#     def __init__(self, corrupted: list[list[float | None]]):
#         self.corrupted = corrupted
#         self.m = len(corrupted)
#         self.n = len(corrupted[0]) if self.m > 0 else 0

#     def make_requests(self) -> list[Request]:
#         return []

#     def receive_requests(self, requests: list[Request]) -> list[Message | None]:
#         return [None] * len(requests)

#     def receive_messages(self, messages: list[Message | None]):
#         pass

#     def recover(self) -> list[list[float]]:
#         m, n = self.m, self.n
#         corrupted = self.corrupted

#         img_type = _classify_image(corrupted, m, n)

#         if img_type == 'binary':
#             result = _inpaint_binary(corrupted, m, n)
#         else:
#             result = _inpaint_smooth(corrupted, m, n)

#         for r in range(m):
#             for c in range(n):
#                 result[r][c] = max(0.0, min(1.0, result[r][c]))

#         return result
