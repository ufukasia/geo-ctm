"""
Controlled degradation sweep — the experiment that tests the central claim.

    python tools/robustness.py --axis dynamic          # the three ramp axes
    python tools/robustness.py --axis yaw_ramp --limit 20
    python tools/robustness.py --axis all              # everything (slow)

For each axis and level, every track is degraded, re-detected and re-fused, and
B1/B3/B5 accuracies plus a paired McNemar test are reported.

This is slow by construction: detections cannot be cached, because the whole
point is that the images change. Expect roughly (levels x tracks) detector
passes per axis.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoctm import HP_FROZEN, upstream                                # noqa: E402
from geoctm.baselines import fuse_ctm, fuse_single_frame              # noqa: E402
from geoctm.decode import load_confusion                              # noqa: E402
from geoctm.degrade import AXES, DYNAMIC_AXES, degrade_track          # noqa: E402
from geoctm.detect import detect_frames, read_track_images            # noqa: E402
from geoctm.metrics import mcnemar                                    # noqa: E402
from tools.eval_ufpr import fuse_geo                                  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='test',
                   choices=['test', 'validation', 'training'])
    p.add_argument('--axis', default='dynamic',
                   choices=['all', 'dynamic'] + list(AXES))
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--ctm-root', default=None)
    p.add_argument('--out', default='./results/robustness.md')
    args = p.parse_args()

    process_plate = upstream.load_process_plate(args.ctm_root)
    detector = upstream.load_detector(args.ctm_root)
    confusion = load_confusion('./confusion_matrix.json')

    tracks_dir, labels_path = upstream.default_split_paths(args.ctm_root, args.split)
    folders = sorted(d for d in os.listdir(tracks_dir)
                     if os.path.isdir(os.path.join(tracks_dir, d)))
    with open(labels_path, encoding='utf-8') as f:
        labels = f.read().split('\n')

    selected = folders[:args.limit] if args.limit else folders
    gts = {fd: labels[folders.index(fd)].replace('-', '').strip() for fd in selected}
    originals = {fd: read_track_images(os.path.join(tracks_dir, fd))[0]
                 for fd in selected}

    if args.axis == 'all':
        axes = list(AXES)
    elif args.axis == 'dynamic':
        axes = list(DYNAMIC_AXES)
    else:
        axes = [args.axis]

    hp = HP_FROZEN
    t0 = time.time()
    report = [
        f'# Robustness sweep ({args.split}, {len(selected)} tracks)',
        '',
        'Same detector weights throughout; NO retraining on degraded data.',
        'What is measured is the robustness of the fusion layer given identical '
        'detections.',
        f'Hyper-parameters (selected on validation): {hp}',
        '',
    ]

    for axis in axes:
        title, levels, _ = AXES[axis]
        rows = {'B1': [], 'B3': [], 'B5': []}
        sig = []

        for lv in levels:
            preds = {'B1': {}, 'B3': {}, 'B5': {}}
            for fd in selected:
                imgs = degrade_track(originals[fd], axis, lv)
                cache = detect_frames(detector, process_plate, imgs, True)
                preds['B1'][fd] = fuse_single_frame(cache, process_plate)
                preds['B3'][fd] = fuse_ctm(cache, process_plate)
                preds['B5'][fd] = fuse_geo(cache, hp, confusion)[3]

            accs = {}
            for m in ('B1', 'B3', 'B5'):
                accs[m] = 100.0 * sum(preds[m][fd] == gts[fd]
                                      for fd in selected) / len(selected)
                rows[m].append(accs[m])

            b, c, pval = mcnemar(preds['B3'], preds['B5'], gts)
            sig.append(pval)
            print(f'[{axis}={lv}] B1={accs["B1"]:.1f}%  B3={accs["B3"]:.1f}%  '
                  f'B5={accs["B5"]:.1f}%  (McNemar p={pval:.4f}; only B3={b}, '
                  f'only B5={c})   ({time.time() - t0:.0f}s)', flush=True)

        kind = ('DYNAMIC (inter-frame change — the axis that tests the thesis)'
                if axis in DYNAMIC_AXES else 'static (detector robustness)')
        report += [f'## Axis: {axis} — {title}  [{kind}]', '',
                   '| Method | ' + ' | '.join(str(l) for l in levels) + ' |',
                   '|---' * (len(levels) + 1) + '|']
        for m in ('B1', 'B3', 'B5'):
            report.append(f'| {m} | ' + ' | '.join(f'{v:.1f}' for v in rows[m]) + ' |')

        gap = [rows['B5'][i] - rows['B3'][i] for i in range(len(levels))]
        report += ['| **B5-B3 (delta)** | ' + ' | '.join(f'{g:+.1f}' for g in gap) + ' |',
                   '| McNemar p | ' + ' | '.join(
                       f'{p:.3f}' + ('*' if p < 0.05 else '') for p in sig) + ' |',
                   '', '(* p < 0.05)', '']

    text = '\n'.join(report)
    print('\n' + text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'\nReport: {args.out}')


if __name__ == '__main__':
    main()
