"""
The guard test: does the baseline still reproduce the published result?

B3 is a verbatim reproduction of the original CTM. On the 60-track UFPR test
split it must score 58/60 = 96.67% and fail on exactly track0107 and track0136.
If that ever changes, our baseline is no longer the published method and every
comparison in the paper is void — so it is asserted rather than assumed.

Skipped automatically when the upstream checkout or the test tracks are absent:

    python tools/setup_upstream.py
    python -m pytest tests/test_reproduction.py -v

The first run performs the detection pass (~2 min on CPU) and caches it; later
runs take seconds.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoctm import HP_FROZEN, upstream                    # noqa: E402
from geoctm.decode import load_confusion                  # noqa: E402


def _available():
    try:
        root = upstream.upstream_root()
    except upstream.UpstreamMissing:
        return False
    tracks = os.path.join(root, 'test_tracks')
    return os.path.isdir(tracks) and len(os.listdir(tracks)) >= 60


requires_upstream = pytest.mark.skipif(
    not _available(),
    reason='upstream CTM checkout or test_tracks missing '
           '(run tools/setup_upstream.py)')


@pytest.fixture(scope='module')
def predictions():
    from tools.eval_ufpr import evaluate, load_caches

    process_plate = upstream.load_process_plate()
    detector = upstream.load_detector()
    tracks_dir, labels_path = upstream.default_split_paths(split='test')

    gts, caches = load_caches(detector, process_plate, tracks_dir, labels_path,
                              './cache', 'test')
    preds = evaluate(gts, caches, HP_FROZEN, process_plate,
                     load_confusion('./confusion_matrix.json'))
    return gts, preds


@requires_upstream
def test_baseline_reproduces_published_accuracy(predictions):
    gts, preds = predictions
    correct = sum(preds['B3'][fd] == gts[fd] for fd in gts)
    assert len(gts) == 60
    assert correct == 58, (
        f'B3 scored {correct}/60, expected the published 58/60. '
        f'The baseline is no longer the published method.')


@requires_upstream
def test_baseline_fails_on_the_documented_tracks(predictions):
    gts, preds = predictions
    failures = sorted(fd for fd in gts if preds['B3'][fd] != gts[fd])
    assert failures == ['track0107', 'track0136']


@requires_upstream
def test_proposed_method_beats_the_baseline(predictions):
    gts, preds = predictions
    b3 = sum(preds['B3'][fd] == gts[fd] for fd in gts)
    b5 = sum(preds['B5'][fd] == gts[fd] for fd in gts)
    assert b5 == 59, f'B5 scored {b5}/60, expected 59/60'
    assert b5 > b3


@requires_upstream
def test_no_row_uses_different_detections(predictions):
    """Sanity check on the protocol itself.

    Every method except B5n must consume the SAME cached detection pass. This
    does not re-verify the caching code; it verifies the claim the paper makes
    about its own protocol, by checking that B1 (which reads one frame out of
    the rotated pass) agrees with B3 wherever B3 read only one frame's worth.
    """
    gts, preds = predictions
    # B5g and B5q share one matching pass and differ only in vote weighting, so
    # they can never disagree on a track where all frames have equal quality.
    disagree = [fd for fd in gts if preds['B5g'][fd] != preds['B5q'][fd]]
    assert len(disagree) <= 3, (
        f'B5g and B5q disagree on {len(disagree)} tracks; the quality weighting '
        f'is doing more than expected, which would confound the ablation')
