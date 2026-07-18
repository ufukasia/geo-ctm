"""
Baseline rows — faithful reproductions of the published method.

  B1   single best frame                         (lower bound)
  B2   CTM association + plain majority vote     (fusion ablation)
  B3   the original CTM, verbatim                (the published method)

B3 is a line-by-line reproduction of upstream evaluate.py's second half. It is
NOT refactored, NOT tidied, and NOT "improved": on the 60-track UFPR test split
it must reproduce the published 58/60 = 96.67%, failing on exactly track0107 and
track0136. tests/test_reproduction.py asserts this. If that number ever moves,
the comparison is invalid and the benchmark should be treated as broken.
"""
import numpy as np

from .decode import crude_norm
from .detect import build_track_box


def majority_char(track_box_):
    """B2: plain majority vote, ties broken by summed confidence."""
    labels = [l[0] for l in track_box_[5]]
    confs = [c[0] for c in track_box_[4]]
    tally = {}
    for l, c in zip(labels, confs):
        n, s = tally.get(l, (0, 0.0))
        tally[l] = (n + 1, s + float(c))
    return max(tally, key=lambda k: tally[k])


def fuse_ctm(cache, process_plate, char_fn=None):
    """B3 / B2 — the original CTM fusion.

    char_fn = process_plate.get_maximum_conf_char  -> B3 (published)
    char_fn = majority_char                        -> B2 (ablation)
    """
    char_fn = char_fn or process_plate.get_maximum_conf_char

    track_boxs = [build_track_box(fr['dets']) for fr in cache['frames'] if fr['dets']]
    if not track_boxs:
        return ''

    n_pngs = cache['n_pngs']
    heights, widths = cache['Hs'], cache['Ws']

    old_char = np.zeros((0, 0))
    for track_box in track_boxs:
        arr_track = process_plate.matching_char(old_char, track_box)
        old_char = arr_track

    mean_h = np.mean(np.array(heights)) if len(heights) > 0 else 0
    mean_w = np.mean(np.array(widths)) if len(widths) > 0 else 0

    if arr_track.shape[0] > 7:
        arr_track = process_plate.merge_box_arr_track(arr_track)
    arr_track = sorted(arr_track, key=lambda x: float(x[0]))

    arr_track = np.array([a for a in arr_track if len(a[5]) >= 1 / 2 * n_pngs],
                         dtype=object)
    re = ''.join(char_fn(a) for a in arr_track)

    if mean_h * 2 > mean_w and arr_track.shape[0] > 0:
        center_x = (arr_track[:, 0] + arr_track[:, 2]) / 2
        center_y = (arr_track[:, 1] + arr_track[:, 3]) / 2
        chars = ['{}'.format(char_fn(a)) for a in arr_track]
        _, re = process_plate.find_chars_plate(center_x, center_y, chars)

    return crude_norm(re)


def fuse_single_frame(cache, process_plate):
    """B1: read the single frame with the highest summed confidence."""
    best_i, best_score = -1, -1.0
    for i, fr in enumerate(cache['frames']):
        if not fr['dets']:
            continue
        score = sum(max(float(c) for c in conf.split('-'))
                    for _, conf, _ in fr['dets'])
        if score > best_score:
            best_i, best_score = i, score
    if best_i < 0:
        return ''

    fr = cache['frames'][best_i]
    arr_track = build_track_box(fr['dets'])
    if arr_track.shape[0] > 7:
        arr_track = process_plate.merge_box_arr_track(arr_track)
    arr_track = np.array(sorted(arr_track, key=lambda x: float(x[0])), dtype=object)

    re = ''.join(process_plate.get_maximum_conf_char(a) for a in arr_track)
    if fr['h'] * 2 > fr['w'] and arr_track.shape[0] > 0:
        center_x = (arr_track[:, 0] + arr_track[:, 2]) / 2
        center_y = (arr_track[:, 1] + arr_track[:, 3]) / 2
        chars = ['{}'.format(process_plate.get_maximum_conf_char(a)) for a in arr_track]
        _, re = process_plate.find_chars_plate(center_x, center_y, chars)

    return crude_norm(re)
