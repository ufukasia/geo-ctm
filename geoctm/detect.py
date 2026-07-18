"""
The detection pass, and its on-disk cache.

Every benchmark row consumes THE SAME per-frame detections; only the
association/fusion layer differs. That is the whole point of the protocol — a
difference in the table cannot be attributed to a better detector, because
there is only one detector and one set of detections.

Two passes are cached per track:

  R  adaptive rotation ON  — a byte-for-byte reproduction of the loop in the
                             upstream evaluate.py (the AR / alpha accumulator)
  N  adaptive rotation OFF — the unrotated pass, used for the AR ablation

Each frame additionally records its resolution and sharpness, which the G3
quality weighting needs.
"""
import json
import math
import os

import cv2
import numpy as np

CACHE_VERSION = 2


def build_track_box(dets):
    """Build the upstream `track_box` array from merged detections.

    Reproduces the construction in evaluate.py exactly, including its
    asymmetric rounding (three coordinates are rounded, the fourth is not).
    Kept faithful on purpose: the baseline rows must be the published method,
    not a tidied-up version of it.
    """
    track_box = []
    for label, confidence, box in dets:
        track_box.append([
            int(round(box[0] - box[2] / 2)),
            int(round(box[1] - box[3] / 2)),
            int(round(box[0] + box[2] / 2)),
            box[1] + box[3] / 2,
            [[float(c)] for c in confidence.split('-')],
            [[l] for l in label.split('-')],
        ])
    return np.array(track_box, dtype=object)


def detect_frames(detector, process_plate, images, adaptive_rotation=True):
    """Run the character detector over one track's frames.

    images : list of BGR frames (already cropped to the plate)

    Works on in-memory images so that the robustness sweep can feed it
    deliberately degraded copies.
    """
    alpha = 0
    frames, widths, heights = [], [], []

    for image in images:
        h, w, _ = image.shape
        widths.append(w)
        heights.append(h)

        if adaptive_rotation:
            import imutils
            image = imutils.rotate(image, math.degrees(alpha))

        sharp = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                    cv2.CV_64F).var())
        detections, _ = detector.detect(image)
        if len(detections) == 0:
            frames.append({'dets': [], 'h': h, 'w': w, 'sharp': sharp})
            continue

        dets = process_plate.merge_box(detections)
        frames.append({'dets': [[l, c, list(b)] for l, c, b in dets],
                       'h': h, 'w': w, 'sharp': sharp})

        if adaptive_rotation:
            track_box = build_track_box(dets)
            center_x = (track_box[:, 0] + track_box[:, 2]) / 2
            center_y = (track_box[:, 1] + track_box[:, 3]) / 2
            degree = process_plate.find_angle(center_x, center_y)
            if 3 < abs(math.degrees(degree)) < 25:
                alpha -= degree

    return {'frames': frames, 'Ws': widths, 'Hs': heights,
            'n_pngs': len(images)}


def imread_unicode(path):
    """cv2.imread that survives non-ASCII paths.

    On Windows cv2.imread goes through a narrow-char API and returns None for
    any path containing non-ASCII characters — which includes most non-English
    user folders (C:/Users/.../Masaüstü/...). It fails silently, so the caller
    sees a None frame rather than an error. Decoding from bytes avoids the
    filename entirely.
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_track_images(track_dir):
    """Read one track folder's PNG frames.

    ORDERING: plain lexicographic sort, matching upstream evaluate.py exactly.
    Note that this is NOT temporal order — frames are named track0091[2].png,
    track0091[10].png, ... so [10] sorts before [2]. Every published number,
    ours and the baseline's, was produced under this ordering, so the comparison
    is internally consistent. It is kept deliberately: changing it would silently
    invalidate the reproduction of the published 96.67%. See docs/METHOD.md.
    """
    pngs = sorted(f for f in os.listdir(track_dir) if f.lower().endswith('png'))
    images = [imread_unicode(os.path.join(track_dir, f)) for f in pngs]

    missing = [f for f, im in zip(pngs, images) if im is None]
    if missing:
        raise IOError(f'Could not decode {len(missing)} frame(s) in {track_dir}, '
                      f'first: {missing[0]}')
    return images, pngs


def cached(cache_dir, kind, name, builder):
    """Memoise a detection pass to <cache_dir>/<kind>/<name>.json."""
    path = os.path.join(cache_dir, kind, name + '.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if data.get('ver') == CACHE_VERSION:
            return data

    data = builder()
    data['ver'] = CACHE_VERSION
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return data


def to_geo_detections(dets):
    """Convert cached merge_box output into the fusion module's format.

    Upstream stores boxes as (xc, yc, w, h) with '-'-joined label/confidence
    strings; fusion.py wants corner boxes and parallel lists.
    """
    out = []
    for label, conf, box in dets:
        labels = label.split('-')
        confs = [float(c) for c in conf.split('-')]
        cx, cy, w, h = box
        out.append((labels, confs, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return out
