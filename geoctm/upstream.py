"""
Bridge to the upstream CTM repository.

We do NOT vendor the original Character-Time-series-Matching code, its trained
weights, or its bundled copy of YOLOv5. Two reasons:

  1. The upstream repository ships no LICENSE file, so there is no grant that
     would let us redistribute it.
  2. Its bundled YOLOv5 is GPL-3.0 upstream, which would relicense this repo.

Instead `tools/setup_upstream.py` clones it into ./third_party/ at install time
and this module wires it onto sys.path. Everything under geoctm/ that is *ours*
(fusion.py, decode.py, degrade.py) runs without it.

The upstream code assumes it is being run from its own directory: it reads
'./character_name.txt' and 'exp/weights/best.pt' as relative paths, and
Char_detection_yolo instantiates a detector at import time. We therefore chdir
into the upstream root while importing and while constructing the detector.
"""
import contextlib
import os
import sys

#: Where setup_upstream.py puts the clone, relative to the repo root.
DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'third_party', 'Character-Time-series-Matching')

_ENV_VAR = 'CTM_ROOT'


class UpstreamMissing(RuntimeError):
    pass


def upstream_root(root=None):
    """Resolve the upstream checkout: argument > $CTM_ROOT > ./third_party/."""
    root = root or os.environ.get(_ENV_VAR) or DEFAULT_ROOT
    root = os.path.abspath(root)
    if not os.path.isfile(os.path.join(root, 'process_plate.py')):
        raise UpstreamMissing(
            f"Upstream CTM not found at {root}\n"
            f"Run:  python tools/setup_upstream.py\n"
            f"or point {_ENV_VAR} at an existing checkout.")
    return root


@contextlib.contextmanager
def _cwd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def load_process_plate(root=None):
    """Import upstream process_plate (the original CTM association + voting)."""
    root = upstream_root(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import process_plate           # noqa: E402  (path set above)
    return process_plate


def load_detector(root=None, weights=None):
    """Construct the upstream character detector.

    Returns an object with .detect(image) -> (results, resized_image), where
    results is [[label, conf_str, (xc, yc, w, h)], ...].
    """
    root = upstream_root(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    yolo_dir = os.path.join(root, 'yolov5')
    if yolo_dir not in sys.path:
        sys.path.insert(0, yolo_dir)

    with _cwd(root):
        from Char_detection_yolo import CharacterDetection   # noqa: E402
        return CharacterDetection(weights_path=weights)


def default_split_paths(root=None, split='test'):
    """Paths to the upstream evaluation splits.

    Only 'test' ships with the upstream repo (60 cropped tracks, downloaded by
    setup_upstream.py). 'validation' and 'training' must be built from the raw
    UFPR-ALPR dataset with tools/prepare_ufpr.py.
    """
    root = upstream_root(root)
    table = {
        'test':       ('test_tracks', 'test_results.txt'),
        'validation': ('ufpr_validation', 'ufpr_validation/labels.txt'),
        'training':   ('ufpr_training', 'ufpr_training/labels.txt'),
    }
    tracks, labels = table[split]
    return os.path.join(root, tracks), os.path.join(root, labels)
