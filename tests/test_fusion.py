"""
Unit tests for the fusion core. These need NO weights and NO dataset — they run
on synthetic detections, so a fresh clone can verify the contribution itself
before downloading anything.

    python -m pytest tests/ -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoctm import fusion                                    # noqa: E402
from geoctm.decode import decode_plain, decode_with_grammar  # noqa: E402


def make_plate(text, x0=10.0, y0=10.0, w=12.0, h=24.0, gap=4.0):
    """Lay out `text` as a row of character boxes."""
    boxes = []
    for i, _ in enumerate(text):
        x = x0 + i * (w + gap)
        boxes.append([x, y0, x + w, y0 + h])
    return boxes


def as_detections(text, boxes, conf=0.9):
    return [([c], [conf], b) for c, b in zip(text, boxes)]


def transform_boxes(boxes, scale=1.0, angle_deg=0.0, tx=0.0, ty=0.0,
                    cx=100.0, cy=20.0):
    """Apply a similarity transform to a list of corner boxes."""
    import math
    a = math.radians(angle_deg)
    ca, sa = scale * math.cos(a), scale * math.sin(a)

    def pt(x, y):
        dx, dy = x - cx, y - cy
        return (cx + ca * dx - sa * dy + tx, cy + sa * dx + ca * dy + ty)

    out = []
    for x1, y1, x2, y2 in boxes:
        (nx1, ny1), (nx2, ny2) = pt(x1, y1), pt(x2, y2)
        out.append([min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)])
    return out


# ---------------------------------------------------------------------------
# Motion estimation
# ---------------------------------------------------------------------------

def test_estimate_motion_recovers_translation():
    src = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)
    dst = src + np.array([7.0, -3.0], dtype=np.float32)
    M, kind = fusion.estimate_motion(src, dst, char_h=24.0)
    assert kind in ('similarity', 'translation')
    assert np.allclose(fusion._apply_transform(M, src), dst, atol=0.5)


def test_estimate_motion_recovers_scale():
    src = np.array([[0, 0], [10, 5], [20, 0], [30, 5]], dtype=np.float32)
    dst = src * 1.2
    M, kind = fusion.estimate_motion(src, dst, char_h=24.0)
    assert kind == 'similarity'
    assert np.allclose(fusion._apply_transform(M, src), dst, atol=0.5)


def test_estimate_motion_degrades_gracefully():
    """No correspondences at all must not raise — it must fall back to identity."""
    M, kind = fusion.estimate_motion([], [], char_h=24.0)
    assert kind == 'identity'
    assert np.allclose(M, [[1, 0, 0], [0, 1, 0]])


def test_estimate_motion_rejects_absurd_scale():
    """A 5x scale jump between frames is not physical; reject the similarity fit."""
    src = np.array([[0, 0], [10, 0]], dtype=np.float32)
    dst = src * 5.0
    _, kind = fusion.estimate_motion(src, dst, char_h=24.0)
    assert kind == 'translation'


# ---------------------------------------------------------------------------
# Matching across frames
# ---------------------------------------------------------------------------

def test_stable_plate_accumulates_one_track_per_character():
    text = 'ABC1234'
    boxes = make_plate(text)
    storage = []
    for f in range(10):
        storage = fusion.match_characters(storage, as_detections(text, boxes),
                                          frame_weight=1.0, frame_id=f)
    assert len(storage) == len(text)
    assert all(len(s['labels']) == 10 for s in storage)


def test_zooming_plate_does_not_fragment_tracks():
    """The regression this whole method exists for.

    A plate growing in the frame (a car approaching) moves each character by far
    more than a fixed gate allows near the edges of the plate. Under the
    similarity model the tracks must survive intact.
    """
    text = 'ABC1234'
    boxes = make_plate(text)
    storage = []
    for f in range(12):
        cur = transform_boxes(boxes, scale=1.0 + 0.06 * f, tx=3.0 * f)
        storage = fusion.match_characters(storage, as_detections(text, cur),
                                          frame_weight=1.0, frame_id=f)
    assert len(storage) == len(text), (
        f'plate fragmented into {len(storage)} tracks for {len(text)} characters')


def test_rotating_plate_does_not_fragment_tracks():
    text = 'ABC1234'
    boxes = make_plate(text)
    storage = []
    for f in range(12):
        cur = transform_boxes(boxes, angle_deg=1.8 * f)
        storage = fusion.match_characters(storage, as_detections(text, cur),
                                          frame_weight=1.0, frame_id=f)
    assert len(storage) == len(text)


def test_reading_survives_a_dropped_frame():
    """A frame where the detector misses a character must not open a new track."""
    text = 'ABC1234'
    boxes = make_plate(text)
    storage = []
    for f in range(10):
        keep = slice(1, None) if f == 5 else slice(None)
        storage = fusion.match_characters(
            storage, as_detections(text[keep], boxes[keep]),
            frame_weight=1.0, frame_id=f)
    assert len(storage) == len(text)


# ---------------------------------------------------------------------------
# Dedupe: the co-occurrence veto
# ---------------------------------------------------------------------------

def test_dedupe_merges_a_duplicate_track():
    """Two overlapping tracks seen in DISJOINT frames are one character."""
    box = [10.0, 10.0, 22.0, 34.0]
    storage = [
        {'box': box, 'labels': ['5'] * 4, 'confs': [0.9] * 4,
         'weights': [1.0] * 4, 'frames': {0, 1, 2, 3}},
        {'box': [11.0, 10.0, 23.0, 34.0], 'labels': ['5'] * 4, 'confs': [0.9] * 4,
         'weights': [1.0] * 4, 'frames': {4, 5, 6, 7}},
    ]
    assert len(fusion.dedupe(storage)) == 1


def test_dedupe_refuses_to_merge_co_occurring_tracks():
    """Two tracks that vote in the SAME frames are two real characters.

    Geometry alone would merge them here (the boxes overlap heavily). The
    co-occurrence veto must override that.
    """
    frames = set(range(8))
    storage = [
        {'box': [10.0, 10.0, 22.0, 34.0], 'labels': ['A'] * 8, 'confs': [0.9] * 8,
         'weights': [1.0] * 8, 'frames': set(frames)},
        {'box': [11.0, 10.0, 23.0, 34.0], 'labels': ['K'] * 8, 'confs': [0.9] * 8,
         'weights': [1.0] * 8, 'frames': set(frames)},
    ]
    assert len(fusion.dedupe(storage)) == 2


def test_dedupe_keeps_vertically_separated_characters():
    """Two-row (motorcycle) plates: same x, different y is NOT a duplicate."""
    storage = [
        {'box': [10.0, 10.0, 22.0, 34.0], 'labels': ['A'] * 4, 'confs': [0.9] * 4,
         'weights': [1.0] * 4, 'frames': {0, 1, 2, 3}},
        {'box': [10.0, 40.0, 22.0, 64.0], 'labels': ['7'] * 4, 'confs': [0.9] * 4,
         'weights': [1.0] * 4, 'frames': {4, 5, 6, 7}},
    ]
    assert len(fusion.dedupe(storage)) == 2


# ---------------------------------------------------------------------------
# Voting and decoding
# ---------------------------------------------------------------------------

def test_quality_weighting_lets_good_frames_outvote_bad_ones():
    """Three sharp frames should beat five blurred ones."""
    char = {'box': [0, 0, 10, 20],
            'labels': ['8'] * 5 + ['B'] * 3,
            'confs': [0.6] * 5 + [0.9] * 3,
            'weights': [0.1] * 5 + [1.0] * 3,
            'frames': set(range(8))}
    unweighted = fusion.class_scores(char, weighted=False)
    weighted = fusion.class_scores(char, weighted=True)
    assert max(unweighted, key=unweighted.get) == '8'
    assert max(weighted, key=weighted.get) == 'B'


def test_grammar_folds_lookalikes_onto_allowed_classes():
    """A '0' in a letter position becomes 'O' — Brazilian format is AAA-NNNN."""
    scores = ([{'0': 1.0}, {'B': 1.0}, {'C': 1.0}] +
              [{str(d): 1.0} for d in (1, 2, 3, 4)])
    assert decode_with_grammar(scores, 'brazil') == 'OBC1234'
    assert decode_plain(scores) == '0BC1234'


def test_grammar_does_not_override_strong_evidence():
    """The point of G2 over the original string edit.

    In a DIGIT position a confident '0' must stay '0'. The old unconditional
    substitution only touched the first three characters, but the principle is
    the same: a look-alike swap has to win on evidence, not by fiat.
    """
    scores = ([{'A': 1.0}, {'B': 1.0}, {'C': 1.0}] +
              [{'0': 9.0, 'O': 0.1}] + [{str(d): 1.0} for d in (2, 3, 4)])
    assert decode_with_grammar(scores, 'brazil') == 'ABC0234'


def test_unknown_grammar_falls_back_to_plain():
    scores = [{'A': 1.0}, {'B': 1.0}]
    assert decode_with_grammar(scores, 'no-such-grammar') == 'AB'


# ---------------------------------------------------------------------------
# End to end, on synthetic data
# ---------------------------------------------------------------------------

def test_fuse_track_reads_a_synthetic_approaching_plate():
    text = 'ABC1234'
    boxes = make_plate(text)
    frames = []
    for f in range(12):
        cur = transform_boxes(boxes, scale=1.0 + 0.05 * f, tx=2.0 * f)
        frames.append({'dets': as_detections(text, cur),
                       'h': 40, 'w': 160, 'sharp': 100.0})
    assert fusion.fuse_track(frames) == text


def test_fuse_track_handles_empty_input():
    assert fusion.fuse_track([]) == ''
    assert fusion.fuse_track([{'dets': [], 'h': 40, 'w': 160, 'sharp': 1.0}]) == ''


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
