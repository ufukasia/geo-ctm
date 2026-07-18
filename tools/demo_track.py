"""
Read one track and show what each method produces. The "does it work" script.

    python tools/demo_track.py third_party/Character-Time-series-Matching/test_tracks/track0091
    python tools/demo_track.py <track_dir> --degrade yaw_ramp --level 40

A track is a folder of cropped plate images, one per frame, in filename order —
the same format the upstream repo's test_tracks uses.

With --degrade you can watch the two methods diverge: on clean frames CTM and
the proposed method usually agree, and under a ramped distortion they do not.
That divergence is the paper's point, and this script is the cheapest way to
see it for yourself.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoctm import HP_FROZEN, upstream                                # noqa: E402
from geoctm.baselines import fuse_ctm, fuse_single_frame              # noqa: E402
from geoctm.decode import load_confusion                              # noqa: E402
from geoctm.degrade import AXES, degrade_track                        # noqa: E402
from geoctm.detect import detect_frames, read_track_images            # noqa: E402
from tools.eval_ufpr import fuse_geo                                  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('track', help='folder of cropped plate frames')
    p.add_argument('--degrade', default=None, choices=list(AXES),
                   help='apply a controlled distortion first')
    p.add_argument('--level', type=float, default=0)
    p.add_argument('--ctm-root', default=None)
    p.add_argument('--gt', default=None, help='ground truth, for a verdict line')
    args = p.parse_args()

    images, pngs = read_track_images(args.track)
    if not images:
        sys.exit(f'No PNG frames in {args.track}')
    print(f'{len(pngs)} frames from {args.track}')

    if args.degrade and args.level:
        images = degrade_track(images, args.degrade, args.level)
        print(f'degraded: {args.degrade} at level {args.level} '
              f'({AXES[args.degrade][0]})')

    process_plate = upstream.load_process_plate(args.ctm_root)
    detector = upstream.load_detector(args.ctm_root)
    confusion = load_confusion('./confusion_matrix.json')

    print('running detector...')
    cache = detect_frames(detector, process_plate, images, adaptive_rotation=True)

    b1 = fuse_single_frame(cache, process_plate)
    b3 = fuse_ctm(cache, process_plate)
    _, _, b5, b5c = fuse_geo(cache, HP_FROZEN, confusion)

    print()
    for name, value in (('B1  single frame', b1),
                        ('B3  original CTM', b3),
                        ('B5  proposed', b5),
                        ('B5c proposed + confusion prior', b5c)):
        mark = ''
        if args.gt:
            mark = '  correct' if value == args.gt.upper() else '  WRONG'
        print(f'  {name:<34} {value or "(nothing read)"}{mark}')

    if args.gt:
        print(f'\n  ground truth                       {args.gt.upper()}')


if __name__ == '__main__':
    main()
