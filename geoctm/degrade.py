"""
Controlled degradation models — the sweep that actually tests the claim.

The thesis is that CTM matches characters as independent points under a
translation-only assumption, while a plate is a rigid plane. That difference
should be invisible when the camera looks straight at the plate and should grow
when it does not — the drone / high-mount regime.

We have no drone dataset. So the UFPR test tracks are degraded in physically
meaningful, controlled ways and each method's collapse curve is measured.

Two families, and the distinction is the whole experiment:

  STATIC axes  apply the SAME distortion to every frame. What breaks here is
               THE DETECTOR; the fusion layer is not the deciding factor. These
               are reported for context only.

  DYNAMIC (ramp) axes vary the distortion ALONG the track — what a moving
               camera actually does. The inter-frame image transform is then no
               longer a pure translation, which is precisely where CTM's fixed
               35 px, translation-only assumption breaks. THIS is where the
               claim is tested.

The detector weights never change: there is no retraining on degraded data.
What is measured is the robustness of the FUSION layer given identical
detections.
"""
import math

import cv2
import numpy as np


def warp_view(img, yaw_deg=0.0, pitch_deg=0.0, f_ratio=3.0):
    """Rotate the plate plane in 3D and re-project it through a pinhole camera.

    A planar point (X, Y, 0) is rotated to (X', Y', Z') and projected as
    u = f X'/Z', v = f Y'/Z', with f = f_ratio * width. A smaller f_ratio means
    a wider angle and harsher perspective. The output canvas is sized to contain
    the projected corners.
    """
    if abs(yaw_deg) < 1e-6 and abs(pitch_deg) < 1e-6:
        return img

    h, w = img.shape[:2]
    f = f_ratio * w
    cy, cp = math.radians(yaw_deg), math.radians(pitch_deg)
    Ry = np.array([[math.cos(cy), 0, math.sin(cy)],
                   [0, 1, 0],
                   [-math.sin(cy), 0, math.cos(cy)]])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(cp), -math.sin(cp)],
                   [0, math.sin(cp), math.cos(cp)]])
    R = Rx @ Ry

    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    pts3 = np.array([[x - w / 2, y - h / 2, 0.0] for x, y in src])

    proj = []
    for p in pts3:
        X, Y, Z = R @ p
        Z += f                      # camera sits f away from the plane
        proj.append([f * X / Z, f * Y / Z])
    proj = np.array(proj, dtype=np.float32)

    proj -= proj.min(axis=0)
    out_w = max(8, int(round(proj[:, 0].max())) + 1)
    out_h = max(8, int(round(proj[:, 1].max())) + 1)
    M = cv2.getPerspectiveTransform(src, proj)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def rescale_height(img, target_h):
    """Drop the plate to target_h pixels, then scale back up.

    The scale-back-up is required, not cosmetic. The detector's input pipeline
    letterboxes every crop to a fixed size anyway, so a plain downscale would
    measure nothing. Scaling back up makes the information loss permanent while
    keeping the geometry fixed — the variable under test is RESOLUTION.
    """
    h, w = img.shape[:2]
    if target_h >= h:
        return img
    s = target_h / h
    small = cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def motion_blur(img, k):
    """Horizontal motion blur with a k-pixel kernel."""
    if k <= 1:
        return img
    kern = np.zeros((k, k), dtype=np.float32)
    kern[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kern)


def inplane_rotate(img, deg):
    if abs(deg) < 1e-6:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# axis -> (title, levels, fn(img, level, t)) where t = frame / (n - 1) in [0, 1]
AXES = {
    # --- static: detector robustness, reported for context
    'yaw':   ('Static yaw (degrees)', [0, 10, 20, 30, 40, 50, 60],
              lambda im, lv, t: warp_view(im, yaw_deg=lv)),
    'pitch': ('Static pitch (degrees)', [0, 10, 20, 30, 40, 50, 60],
              lambda im, lv, t: warp_view(im, pitch_deg=lv)),
    'scale': ('Plate height (px)', [0, 24, 20, 16, 12, 10, 8],
              lambda im, lv, t: rescale_height(im, lv)),
    'blur':  ('Motion blur (px)', [0, 3, 5, 7, 9, 11, 13],
              lambda im, lv, t: motion_blur(im, lv)),

    # --- dynamic: where the thesis is tested
    'yaw_ramp':  ('Yaw swept 0 -> L across the track (degrees)',
                  [0, 10, 20, 30, 40, 50, 60],
                  lambda im, lv, t: warp_view(im, yaw_deg=lv * t)),
    'roll_ramp': ('In-plane rotation swept 0 -> L across the track (degrees)',
                  [0, 5, 10, 15, 20, 25, 30],
                  lambda im, lv, t: inplane_rotate(im, lv * t)),
    'zoom_ramp': ('Resolution falling across the track (to px height)',
                  [0, 24, 20, 16, 12, 10, 8],
                  lambda im, lv, t: rescale_height(
                      im, int(round(im.shape[0] - (im.shape[0] - lv) * t)))),
}

DYNAMIC_AXES = ('yaw_ramp', 'roll_ramp', 'zoom_ramp')


def degrade(img, axis, level, t):
    """Apply one degradation. level == 0 always means 'clean'."""
    if level == 0:
        return img
    return AXES[axis][2](img, level, t)


def degrade_track(images, axis, level):
    """Apply a degradation across a whole track, ramping where applicable."""
    n = max(1, len(images) - 1)
    return [degrade(im, axis, level, i / n) for i, im in enumerate(images)]
