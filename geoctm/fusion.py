"""
Geometry-constrained character fusion — the core contribution.

The original CTM (Che et al., MAPR 2022) matches characters between frames as
*independent points*: a Hungarian assignment on Euclidean centre distance with a
fixed 35 px gate, and unmatched tracks are coasted by the mean translation dC.

A license plate is a rigid planar object. Its character centres therefore move
between frames under a single planar motion (translation + rotation + scale; a
homography in the general case), not under 35 px of independent slack. For a
single-row plate the centres are nearly collinear, which makes homography
estimation degenerate, so we fit the richest model the data can always support:
a SIMILARITY transform T(x) = sRx + t, 4 degrees of freedom, 2 correspondences.
The original paper's adaptive rotation (AR) and mean-translation (dC) steps are
special cases of this model.

Three components:

  G1  Similarity-transform (RANSAC) constrained inter-frame matching, with
      height-scaled gates, transform-based coasting, and union-find duplicate
      removal guarded by a co-occurrence veto.
  G2  Format-constrained decoding: the plate grammar is applied as a
      position-constrained argmax over character CLASS SCORES — the
      probabilistic generalisation of the original code's unconditional
      "0 -> O, 1 -> I" string substitution.
  G3  Frame-quality weighted voting: each frame's vote is weighted by that
      frame's resolution (original crop height) and sharpness (Laplacian
      variance).

This module is standalone: it depends only on numpy, scipy and OpenCV, and it
never imports the upstream CTM code. See docs/METHOD.md.
"""
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
# A "tracked character" is a dict:
#   box     : [x1, y1, x2, y2]   last observed (or predicted) position
#   labels  : [str, ...]         class votes, one per contributing detection
#   confs   : [float, ...]       detector confidence of each vote
#   weights : [float, ...]       frame-quality weight of each vote (G3)
#   frames  : set[int]           frame indices that voted (used by dedupe)
#
# A detection is a triple (labels, confs, [x1, y1, x2, y2]). Votes are lists
# because the upstream merge_box step can fuse several boxes into one; a plain
# single detection is simply a one-element list.


def _new_char(labels, confs, box, frame_weight, frame_id):
    return {'box': list(map(float, box)),
            'labels': list(labels),
            'confs': [float(c) for c in confs],
            'weights': [frame_weight] * len(labels),
            'frames': {frame_id}}


def _add_votes(char, labels, confs, frame_weight, frame_id):
    char['labels'].extend(labels)
    char['confs'].extend(float(c) for c in confs)
    char['weights'].extend([frame_weight] * len(labels))
    char['frames'].add(frame_id)


def _centers(boxes):
    b = np.asarray(boxes, dtype=np.float32)
    return np.stack(((b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0), axis=1)


def _iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA + 1) * max(0.0, yB - yA + 1)
    areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(areaA + areaB - inter)


def _median_char_h(boxes):
    b = np.asarray(boxes, dtype=np.float32)
    return float(np.median(b[:, 3] - b[:, 1])) if len(b) else 0.0


def _hungarian(cost):
    r, c = linear_sum_assignment(cost)
    return np.stack((r, c), axis=1)


def _apply_transform(M, pts):
    """Apply a 2x3 affine matrix to Nx2 points."""
    pts = np.asarray(pts, dtype=np.float32)
    return pts @ M[:, :2].T + M[:, 2]


# ---------------------------------------------------------------------------
# G1 — geometry-constrained matching
# ---------------------------------------------------------------------------

def estimate_motion(old_pts, new_pts, char_h):
    """Estimate the inter-frame similarity transform from matched centres.

    Returns (M, kind) with M a 2x3 matrix and kind in
    {'similarity', 'translation', 'identity'} — a model hierarchy that falls
    back to whatever the data actually supports.
    """
    old_pts = np.asarray(old_pts, dtype=np.float32)
    new_pts = np.asarray(new_pts, dtype=np.float32)

    if len(old_pts) >= 2:
        thr = max(3.0, 0.30 * char_h)
        M, inliers = cv2.estimateAffinePartial2D(
            old_pts.reshape(-1, 1, 2), new_pts.reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=thr,
            maxIters=500, confidence=0.99)
        if M is not None and inliers is not None and inliers.sum() >= 2:
            # Guard against wild scale/rotation estimates from noisy 2-point
            # samples: a plate does not change scale by >40% between frames.
            s = math.hypot(M[0, 0], M[0, 1])
            if 0.6 < s < 1.4:
                return M.astype(np.float32), 'similarity'

    if len(old_pts) >= 1:
        d = np.median(new_pts - old_pts, axis=0)
        M = np.array([[1, 0, d[0]], [0, 1, d[1]]], dtype=np.float32)
        return M, 'translation'

    return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 'identity'


def match_characters(storage, detections, frame_weight=1.0, frame_id=0):
    """One frame of geometry-constrained matching. Replaces CTM's matching_char.

    storage      : list[dict]  updated in place and returned
    detections   : list[(labels, confs, [x1, y1, x2, y2])]
    frame_weight : this frame's G3 quality weight
    frame_id     : frame index, recorded for the co-occurrence test in dedupe()

    Two-stage design. Stage 1 uses a loose gate purely to get correspondences
    good enough to fit the motion model; stage 2 predicts every stored character
    through that model and re-matches with a tight, character-height-scaled gate.
    A fixed pixel gate cannot do this: 35 px is generous for a distant plate and
    restrictive for a near one.
    """
    if not detections:
        return storage
    if not storage:
        return [_new_char(l, c, b, frame_weight, frame_id) for l, c, b in detections]

    det_boxes = [d[2] for d in detections]
    old_c = _centers([s['box'] for s in storage])
    new_c = _centers(det_boxes)
    char_h = _median_char_h(det_boxes) or _median_char_h([s['box'] for s in storage])

    # --- stage 1: tentative matching (loose gate) -> motion estimate
    dist = np.linalg.norm(old_c[:, None, :] - new_c[None, :, :], axis=2)
    gate1 = max(35.0, 1.2 * char_h)
    tentative = [(i, j) for i, j in _hungarian(dist) if dist[i, j] <= gate1]

    if tentative:
        M, kind = estimate_motion(old_c[[i for i, _ in tentative]],
                                  new_c[[j for _, j in tentative]], char_h)
    else:
        M, kind = estimate_motion([], [], char_h)

    # --- stage 2: predict all stored characters, match with a tight gate
    pred_c = _apply_transform(M, old_c)
    dist2 = np.linalg.norm(pred_c[:, None, :] - new_c[None, :, :], axis=2)
    gate2 = max(10.0, 0.60 * char_h)
    matches = [(i, j) for i, j in _hungarian(dist2) if dist2[i, j] <= gate2]
    matched_old = {i for i, _ in matches}
    matched_new = {j for _, j in matches}

    for i, j in matches:
        labels, confs, box = detections[j]
        storage[i]['box'] = list(map(float, box))
        _add_votes(storage[i], labels, confs, frame_weight, frame_id)

    # --- unmatched stored characters: coast through the transform, not dC
    scale = math.hypot(M[0, 0], M[0, 1])
    for i, s in enumerate(storage):
        if i in matched_old:
            continue
        x1, y1, x2, y2 = s['box']
        cx, cy = _apply_transform(M, [[(x1 + x2) / 2, (y1 + y2) / 2]])[0]
        w, h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
        s['box'] = [float(cx - w), float(cy - h), float(cx + w), float(cy + h)]

    # --- unmatched detections: duplicate-character rejection
    # If a detection overlaps an existing track strongly (IoU), its vote joins
    # that track instead of opening a new one. This is what stops readings like
    # "993P2626" (a character duplicated into two tracks).
    # IoU rather than centre distance: for narrow glyphs such as '1' a
    # NEIGHBOURING character's centre can also be close, and the vote would be
    # attributed to the wrong track.
    for j, (labels, confs, box) in enumerate(detections):
        if j in matched_new:
            continue
        best_i, best_iou = -1, 0.0
        for i, s in enumerate(storage):
            v = _iou(box, s['box'])
            if v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0 and best_iou >= 0.25 and frame_id not in storage[best_i]['frames']:
            _add_votes(storage[best_i], labels, confs, frame_weight, frame_id)
        else:
            storage.append(_new_char(labels, confs, box, frame_weight, frame_id))

    return storage


# ---------------------------------------------------------------------------
# G1 (final step) — duplicate track removal
# ---------------------------------------------------------------------------

def dedupe(storage, iou_thr=0.25, dx_ratio=0.70, dy_ratio=0.40, cooc_thr=0.60):
    """Collapse several tracks that describe the same physical character.

    Two constraints are combined:

    (a) CO-OCCURRENCE (decisive). Two tracks that both received a vote in the
        SAME frame correspond to TWO SEPARATE detections in that frame, so they
        are overwhelmingly likely to be DIFFERENT characters. Duplicate tracks,
        by contrast, are born when one dies and another opens, so their frame
        sets are largely disjoint. On UFPR the separation is sharp: genuine
        neighbouring pairs co-occur in 93-100% of frames, duplicate pairs in
        23-43%. If the overlap ratio exceeds cooc_thr the merge is vetoed —
        even when the geometry says "these are in the same place".

    (b) GEOMETRY. Genuine neighbouring characters are separated by at least one
        character width (measured: dx >= 11 px, IoU <= 0.14); duplicates overlap
        (IoU 0.42-0.83, dx <= 0.46 character widths). The vertical condition is
        required: on a two-row (motorcycle) plate the upper and lower row
        characters overlap horizontally without being the same character.

    Merging is done with UNION-FIND over groups. The upstream greedy chained
    merge (merge_box_arr_track) grew its bounding box as it merged and therefore
    swallowed genuine neighbouring characters as well.
    """
    n = len(storage)
    if n < 2:
        return storage

    boxes = [s['box'] for s in storage]
    char_w = float(np.median([b[2] - b[0] for b in boxes]))
    char_h = float(np.median([b[3] - b[1] for b in boxes]))

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Frame sets per group: co-occurrence must stay valid after a merge.
    fr = [set(s.get('frames', set())) for s in storage]

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs((boxes[i][0] + boxes[i][2]) / 2 - (boxes[j][0] + boxes[j][2]) / 2)
            dy = abs((boxes[i][1] + boxes[i][3]) / 2 - (boxes[j][1] + boxes[j][3]) / 2)
            near = dx < dx_ratio * char_w and dy < dy_ratio * char_h
            if near or _iou(boxes[i], boxes[j]) > iou_thr:
                pairs.append((dx / max(char_w, 1e-6), i, j))

    for _, i, j in sorted(pairs):            # closest (most certain) pairs first
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        inter = len(fr[ri] & fr[rj])
        if inter > cooc_thr * min(len(fr[ri]), len(fr[rj])):
            continue                         # (a) seen together: distinct characters
        parent[ri] = rj
        fr[rj] |= fr[ri]

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out = []
    for members in groups.values():
        bs = [boxes[i] for i in members]
        out.append({
            'box': [min(b[0] for b in bs), min(b[1] for b in bs),
                    max(b[2] for b in bs), max(b[3] for b in bs)],
            'labels': [l for i in members for l in storage[i]['labels']],
            'confs': [c for i in members for c in storage[i]['confs']],
            'weights': [w for i in members for w in storage[i]['weights']],
            'frames': set().union(*[storage[i].get('frames', set()) for i in members]),
        })
    return out


# ---------------------------------------------------------------------------
# G3 — quality-weighted voting, and reading order
# ---------------------------------------------------------------------------

def frame_quality(orig_h, max_h, sharpness, max_sharpness):
    """G3 frame weight: resolution x sharpness (geometric mean).

    NOTE: the functional form must be chosen on a VALIDATION split; it is not
    tuned on test. See docs/METHOD.md.
    """
    h = orig_h / max_h if max_h > 0 else 1.0
    s = sharpness / max_sharpness if max_sharpness > 0 else 1.0
    return max(0.05, math.sqrt(max(h, 1e-6) * max(s, 1e-6)))


def class_scores(char, weighted=True):
    """Class -> total confidence score, optionally scaled by the G3 weight."""
    sc = {}
    for l, c, w in zip(char['labels'], char['confs'], char['weights']):
        sc[l] = sc.get(l, 0.0) + float(c) * (float(w) if weighted else 1.0)
    return sc


def read_track(storage, n_frames, mean_h, mean_w,
               min_support=0.5, weighted=True, hp=None):
    """Reduce a track store to class-score dicts in reading order.

    Returns list[dict] — one {class: score} per character position.

    Duplicate tracks are removed first, then the same support filter as the
    original protocol is applied: a character must have been seen in at least
    min_support * n_frames frames. On a two-row plate (mean_h * 2 > mean_w) the
    upper row is read first.
    """
    rows = dedupe(storage, **(hp or {}))
    rows = [s for s in rows if len(s['labels']) >= min_support * n_frames]
    if not rows:
        return []

    c = _centers([r['box'] for r in rows])
    two_row = mean_h * 2 > mean_w
    if two_row and len(rows) >= 2:
        x, y = c[:, 0], c[:, 1]
        n = len(x)
        ssxx = float(np.sum(x * x) - n * x.mean() ** 2)
        a = (float(np.sum(y * x) - n * y.mean() * x.mean()) / ssxx) if abs(ssxx) > 1e-6 else 0.0
        b = y.mean() - a * x.mean()
        upper = [i for i in range(n) if a * x[i] + b - y[i] >= 0]
        lower = [i for i in range(n) if i not in upper]
        order = sorted(upper, key=lambda i: x[i]) + sorted(lower, key=lambda i: x[i])
    else:
        order = sorted(range(len(rows)), key=lambda i: c[i, 0])

    return [class_scores(rows[i], weighted=weighted) for i in order]


# ---------------------------------------------------------------------------
# Convenience: the whole pipeline for one track
# ---------------------------------------------------------------------------

def fuse_track(frames, mean_h=None, mean_w=None, n_frames=None,
               grammar='brazil', hp=None, confusion=None):
    """Run the full geometry-constrained fusion over one track.

    frames : list of per-frame dicts
             {'dets': [(labels, confs, box), ...], 'h': int, 'w': int,
              'sharp': float}
             Frames with no detections may be included; they still count
             towards n_frames, exactly as in the original protocol.

    Returns the decoded plate string.
    """
    from .decode import decode_with_confusion, decode_with_grammar

    live = [f for f in frames if f['dets']]
    if not live:
        return ''

    max_h = max(f['h'] for f in live)
    max_s = max(f.get('sharp', 1.0) for f in live)

    storage = []
    for k, f in enumerate(frames):
        if not f['dets']:
            continue
        w = frame_quality(f['h'], max_h, f.get('sharp', 1.0), max_s)
        storage = match_characters(storage, f['dets'], frame_weight=w, frame_id=k)

    n_frames = n_frames if n_frames is not None else len(frames)
    mean_h = mean_h if mean_h is not None else float(np.mean([f['h'] for f in frames]))
    mean_w = mean_w if mean_w is not None else float(np.mean([f['w'] for f in frames]))

    scores = read_track(storage, n_frames, mean_h, mean_w, weighted=True, hp=hp)
    if confusion:
        return decode_with_confusion(scores, confusion, grammar)
    return decode_with_grammar(scores, grammar)
