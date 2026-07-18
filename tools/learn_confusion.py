"""
Learn the character confusion matrix from the TRAINING split (G2).

    python tools/learn_confusion.py --split training

Output: confusion_matrix.json  ->  {"P": {detected_class: {true_class: prob}}}

The original code forces the plate format with an unconditional string edit
("in the first 3 characters, 0 -> O and 1 -> I"). That never learns which
character is actually confused with which, and it ignores confidence entirely.
Our first attempt replaced it with a hand-written look-alike table, which gave
no measurable gain — reported honestly in the paper, and still available as the
B5 row. This script does the principled version: the detector's real confusion
matrix P(true | detected) is estimated on TRAINING and used as an emission model
during fusion (the B5c row).

Split discipline: the matrix is learned on TRAINING, thresholds are selected on
VALIDATION, results are reported on TEST. The three are disjoint.

Requires the UFPR training split, which is not in the upstream repo — build it
from the raw UFPR-ALPR dataset first.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoctm import HP_FROZEN, fusion, upstream               # noqa: E402
from geoctm.detect import to_geo_detections                  # noqa: E402
from tools.eval_ufpr import load_caches                      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='training',
                   choices=['training', 'validation', 'test'])
    p.add_argument('--out', default='./confusion_matrix.json')
    p.add_argument('--alpha', type=float, default=0.5,
                   help='Laplace smoothing, for confusions never observed')
    p.add_argument('--ctm-root', default=None)
    p.add_argument('--cache-dir', default='./cache')
    args = p.parse_args()

    if args.split == 'test':
        print('WARNING: learning the confusion prior on TEST invalidates every '
              'number this repository reports. Doing it anyway because you '
              'asked.\n')

    process_plate = upstream.load_process_plate(args.ctm_root)
    detector = upstream.load_detector(args.ctm_root)
    tracks_dir, labels_path = upstream.default_split_paths(args.ctm_root, args.split)
    gts, caches = load_caches(detector, process_plate, tracks_dir, labels_path,
                              args.cache_dir, args.split)

    counts = defaultdict(lambda: defaultdict(float))
    used, skipped = 0, 0

    for fd, (cache_r, _) in caches.items():
        gt = gts[fd]
        frames = cache_r['frames']
        live = [f for f in frames if f['dets']]
        if not live:
            continue

        max_h = max(f['h'] for f in live)
        max_s = max(f['sharp'] for f in live)
        storage = []
        for k, fr in enumerate(frames):
            if not fr['dets']:
                continue
            w = fusion.frame_quality(fr['h'], max_h, fr['sharp'], max_s)
            storage = fusion.match_characters(storage, to_geo_detections(fr['dets']),
                                              frame_weight=w, frame_id=k)

        scores = fusion.read_track(storage, cache_r['n_pngs'],
                                   float(np.mean(cache_r['Hs'])),
                                   float(np.mean(cache_r['Ws'])),
                                   hp=HP_FROZEN)

        # Position-wise alignment is only safe when the lengths agree. A track
        # read at the wrong length would attribute every character's evidence to
        # the wrong ground-truth class, poisoning the matrix — so skip it.
        if len(scores) != len(gt):
            skipped += 1
            continue

        used += 1
        for sc, true_ch in zip(scores, gt):
            for det_ch, s in sc.items():
                counts[det_ch][true_ch] += float(s)

    classes = sorted({c for d in counts for c in counts[d]} | set(counts))
    P = {}
    for d in classes:
        row = {c: counts[d].get(c, 0.0) + args.alpha for c in classes}
        z = sum(row.values())
        P[d] = {c: v / z for c, v in row.items()}

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'split': args.split, 'tracks_used': used,
                   'tracks_skipped': skipped, 'P': P}, f, indent=1)

    print(f'{used} tracks used, {skipped} skipped (length mismatch)')
    print(f'{len(classes)} classes. Strongest confusions, P(true | detected), '
          f'off-diagonal:')
    off = [(P[d][c], d, c) for d in P for c in P[d] if c != d]
    for pr, d, c in sorted(off, reverse=True)[:12]:
        print(f'  detected "{d}" -> true "{c}" : {pr:.3f}')
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
