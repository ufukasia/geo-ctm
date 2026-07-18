"""
geo-ctm — geometry-constrained character fusion for video license plate
recognition.

Quick use, once you have per-frame character detections::

    from geoctm import fuse_track

    plate = fuse_track([
        {'dets': [(['5'], [0.91], [10, 4, 22, 30]), ...],
         'h': 48, 'w': 160, 'sharp': 120.0},
        ...
    ])

Everything in `fusion`, `decode` and `degrade` is self-contained (numpy, scipy,
OpenCV). `upstream`, `detect` and `baselines` additionally need a checkout of
the original CTM repository — see tools/setup_upstream.py.
"""

from .decode import (decode_plain, decode_with_confusion, decode_with_grammar,
                     load_confusion)
from .fusion import (dedupe, estimate_motion, frame_quality, fuse_track,
                     match_characters, read_track)

__version__ = '1.0.0'

#: Duplicate-track thresholds. Selected by grid search on the UFPR VALIDATION
#: split (30 tracks) and FROZEN before touching test. See docs/METHOD.md.
HP_FROZEN = dict(iou_thr=0.20, dx_ratio=0.70, dy_ratio=0.40, cooc_thr=0.60)

__all__ = [
    'HP_FROZEN', '__version__',
    'dedupe', 'estimate_motion', 'frame_quality', 'fuse_track',
    'match_characters', 'read_track',
    'decode_plain', 'decode_with_confusion', 'decode_with_grammar',
    'load_confusion',
]
