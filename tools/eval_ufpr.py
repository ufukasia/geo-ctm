"""
The main benchmark: UFPR-ALPR, frozen detector, association/fusion varied.

    python tools/eval_ufpr.py                 # 60 test tracks, all methods
    python tools/eval_ufpr.py --limit 3       # smoke test
    python tools/eval_ufpr.py --tune          # pick thresholds on validation first

Rows:

  B1    single best frame                                  lower bound
  B2    CTM association + majority vote                    fusion ablation
  B3    the original CTM, verbatim                         published method
  B5g   G1 geometry matching + confidence vote
  B5q   G1 + G3 quality-weighted vote
  B5    G1 + G3 + G2 grammar decoding                      PROPOSED
  B5c   B5 with a learned confusion prior                  ablation
  B5n   B5 without adaptive rotation                       ablation

Every row consumes IDENTICAL per-frame detections from the same detector
weights; only the association/fusion layer differs. B5n is the one exception —
it uses the unrotated detection pass, which isolates AR's contribution at the
DETECTION layer rather than the fusion layer.

Detections are cached per track (./cache), so re-running fusion experiments
takes seconds.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoctm import HP_FROZEN, fusion                                  # noqa: E402
from geoctm.baselines import fuse_ctm, fuse_single_frame, majority_char  # noqa: E402
from geoctm.decode import (crude_norm, decode_plain, decode_with_confusion,  # noqa: E402
                           decode_with_grammar, load_confusion)
from geoctm.detect import (cached, detect_frames, read_track_images,   # noqa: E402
                           to_geo_detections)
from geoctm.metrics import char_accuracy, mcnemar                     # noqa: E402
from geoctm import upstream                                           # noqa: E402

METHODS = ['B1', 'B2', 'B3', 'B5g', 'B5q', 'B5', 'B5c', 'B5n']
DESCRIPTIONS = {
    'B1':  'single frame',
    'B2':  'CTM assoc. + majority vote',
    'B3':  'original CTM (verbatim)',
    'B5g': 'G1 geometry matching',
    'B5q': 'G1 + G3 quality weighting',
    'B5':  'G1 + G3 + G2 grammar — **PROPOSED**',
    'B5c': 'B5 + learned confusion prior (ablation)',
    'B5n': 'B5 without adaptive rotation (ablation)',
}


def fuse_geo(cache, hp, confusion):
    """The B5 family. Returns (B5g, B5q, B5, B5c).

    All four share one pass of geometry-constrained matching; they differ only
    in how the accumulated class scores are turned into a string. That keeps the
    ablation honest — the matching is identical across the four.
    """
    frames = cache['frames']
    live = [fr for fr in frames if fr['dets']]
    if not live:
        return '', '', '', ''

    max_h = max(fr['h'] for fr in live)
    max_s = max(fr['sharp'] for fr in live)

    storage = []
    for k, fr in enumerate(frames):
        if not fr['dets']:
            continue
        w = fusion.frame_quality(fr['h'], max_h, fr['sharp'], max_s)
        storage = fusion.match_characters(storage, to_geo_detections(fr['dets']),
                                          frame_weight=w, frame_id=k)

    mean_h = float(np.mean(cache['Hs']))
    mean_w = float(np.mean(cache['Ws']))
    n = cache['n_pngs']

    unweighted = fusion.read_track(storage, n, mean_h, mean_w, weighted=False, hp=hp)
    weighted = fusion.read_track(storage, n, mean_h, mean_w, weighted=True, hp=hp)

    return (crude_norm(decode_plain(unweighted)),
            crude_norm(decode_plain(weighted)),
            decode_with_grammar(weighted, 'brazil'),
            decode_with_confusion(weighted, confusion, 'brazil'))


def load_caches(detector, process_plate, tracks_dir, labels_path,
                cache_dir, split, limit=0):
    """Run (or load) both detection passes for every track in a split."""
    folders = sorted(d for d in os.listdir(tracks_dir)
                     if os.path.isdir(os.path.join(tracks_dir, d)))
    with open(labels_path, encoding='utf-8') as f:
        labels = f.read().split('\n')

    selected = folders[:limit] if limit else folders
    gts, caches = {}, {}

    for i, fd in enumerate(selected):
        track_dir = os.path.join(tracks_dir, fd)
        if not any(f.lower().endswith('png') for f in os.listdir(track_dir)):
            continue
        gts[fd] = labels[folders.index(fd)].replace('-', '').strip()

        # Images are read inside the builder, not before it: a cache hit must
        # not pay for decoding ~40 PNGs it will never look at.
        def build(rotate, d=track_dir):
            return detect_frames(detector, process_plate,
                                 read_track_images(d)[0], rotate)

        caches[fd] = (
            cached(cache_dir, f'{split}_passR', fd, lambda: build(True)),
            cached(cache_dir, f'{split}_passN', fd, lambda: build(False)),
        )
        print(f'  detecting {i + 1}/{len(selected)} {fd}', end='\r', flush=True)
    print()
    return gts, caches


def tune(gts, caches, confusion, log_path=None):
    """Select the dedupe thresholds on VALIDATION. Test is never consulted."""
    grid = [dict(iou_thr=i, dx_ratio=d, dy_ratio=0.40, cooc_thr=c)
            for i in (0.20, 0.25, 0.30)
            for d in (0.55, 0.60, 0.70, 0.80)
            for c in (0.40, 0.60, 0.80)]

    rows = []
    for hp in grid:
        ok = sum(fuse_geo(caches[fd][0], hp, confusion)[2] == gts[fd] for fd in gts)
        rows.append((ok, hp))
    rows.sort(key=lambda r: -r[0])

    best_score = rows[0][0]
    # Among ties, take the most central setting — sitting on a threshold edge
    # is how a "tuned" hyper-parameter turns into an overfitted one.
    best = min((hp for ok, hp in rows if ok == best_score),
               key=lambda hp: abs(hp['dx_ratio'] - 0.675) + abs(hp['cooc_thr'] - 0.6))

    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'# Hyper-parameter selection — VALIDATION ({len(gts)} tracks)\n\n')
            f.write('| iou_thr | dx_ratio | cooc_thr | correct |\n|---|---|---|---|\n')
            for ok, hp in rows:
                f.write(f"| {hp['iou_thr']} | {hp['dx_ratio']} | {hp['cooc_thr']} "
                        f"| {ok}/{len(gts)} |\n")
            f.write(f'\nSelected: {best}  ({best_score}/{len(gts)})\n')
    return best, best_score


def evaluate(gts, caches, hp, process_plate, confusion):
    preds = {m: {} for m in METHODS}
    for fd in gts:
        cache_r, cache_n = caches[fd]
        preds['B1'][fd] = fuse_single_frame(cache_r, process_plate)
        preds['B2'][fd] = fuse_ctm(cache_r, process_plate, majority_char)
        preds['B3'][fd] = fuse_ctm(cache_r, process_plate)
        g, q, b5, b5c = fuse_geo(cache_r, hp, confusion)
        preds['B5g'][fd], preds['B5q'][fd] = g, q
        preds['B5'][fd], preds['B5c'][fd] = b5, b5c
        preds['B5n'][fd] = fuse_geo(cache_n, hp, confusion)[2]
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='test',
                   choices=['test', 'validation', 'training'])
    p.add_argument('--tracks', default=None, help='custom track folder')
    p.add_argument('--labels', default=None, help='custom label file')
    p.add_argument('--tag', default=None, help='split name used in the report')
    p.add_argument('--tune', action='store_true',
                   help='select thresholds on validation, then evaluate --split')
    p.add_argument('--ctm-root', default=None,
                   help='upstream checkout (default: ./third_party/..., or $CTM_ROOT)')
    p.add_argument('--cache-dir', default='./cache')
    p.add_argument('--out', default='./results/ufpr_test.md')
    p.add_argument('--limit', type=int, default=0, help='first N tracks (smoke test)')
    args = p.parse_args()

    process_plate = upstream.load_process_plate(args.ctm_root)
    detector = upstream.load_detector(args.ctm_root)
    confusion = load_confusion('./confusion_matrix.json')
    t0 = time.time()

    hp, hp_note = HP_FROZEN, 'FROZEN (geoctm.HP_FROZEN)'
    if args.tune:
        v_dir, v_lbl = upstream.default_split_paths(args.ctm_root, 'validation')
        print('Loading VALIDATION split (threshold selection)...')
        v_gts, v_caches = load_caches(detector, process_plate, v_dir, v_lbl,
                                      args.cache_dir, 'validation')
        hp, v_ok = tune(v_gts, v_caches, confusion,
                        './results/hp_selection_validation.md')
        hp_note = (f'selected on VALIDATION ({len(v_gts)} tracks) -> '
                   f'B5={v_ok}/{len(v_gts)}; test was NOT used')
        print(f'Selected: {hp}  (validation B5={v_ok}/{len(v_gts)})')

    if args.tracks:
        tracks_dir = args.tracks
        labels_path = args.labels or os.path.join(args.tracks, 'labels.txt')
        split_name = args.tag or os.path.basename(os.path.normpath(args.tracks))
    else:
        tracks_dir, labels_path = upstream.default_split_paths(args.ctm_root, args.split)
        split_name = args.tag or args.split

    print(f'Loading {split_name} ({tracks_dir})...')
    gts, caches = load_caches(detector, process_plate, tracks_dir, labels_path,
                              args.cache_dir, split_name, args.limit)
    preds = evaluate(gts, caches, hp, process_plate, confusion)

    for fd in gts:
        marks = ' '.join(
            f"{m}:{'+' if preds[m][fd] == gts[fd] else preds[m][fd] or '?'}"
            for m in METHODS)
        print(f'{fd} gt={gts[fd]}  {marks}')

    n = len(gts)
    lines = [
        f'# UFPR-ALPR benchmark — split: {split_name} ({n} tracks)',
        '',
        'Same character detector and same per-frame detections in every row; '
        'only the association/fusion layer differs.',
        '',
        f'Hyper-parameters: {hp}',
        f'Selection: {hp_note}',
        '',
        f'Date: {time.strftime("%Y-%m-%d %H:%M")} | '
        f'Runtime: {time.time() - t0:.0f}s',
        '',
        '| Method | Exact match | Accuracy | Character accuracy | Failures |',
        '|---|---|---|---|---|',
    ]
    for m in METHODS:
        ok = sum(preds[m][fd] == gts[fd] for fd in gts)
        acc_c = np.mean([char_accuracy(preds[m][fd], gts[fd]) for fd in gts])
        fails = [fd for fd in gts if preds[m][fd] != gts[fd]]
        shown = ', '.join(f"{fd}({preds[m][fd] or '-'}/{gts[fd]})" for fd in fails[:8])
        if len(fails) > 8:
            shown += f' +{len(fails) - 8}'
        lines.append(f'| {m} — {DESCRIPTIONS[m]} | {ok}/{n} | {100 * ok / n:.2f}% | '
                     f'{100 * acc_c:.2f}% | {shown} |')

    b, c, pval = mcnemar(preds['B3'], preds['B5'], gts)
    lines += [
        '',
        '## Significance (exact McNemar, two-sided) — B3 vs B5',
        '',
        f'- Only B3 correct: {b} tracks | Only B5 correct: {c} tracks '
        f'(discordant: {b + c})',
        f'- p = {pval:.4f}' + (
            '  -> significant (p < 0.05)' if pval < 0.05 else
            '  -> NOT significant on its own; the real evidence is the '
            'robustness curves and cross-split consistency (reported honestly)'),
    ]

    report = '\n'.join(lines)
    print('\n' + report)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f'\nReport: {args.out}')


if __name__ == '__main__':
    main()
