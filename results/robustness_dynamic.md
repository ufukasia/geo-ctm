# Robustness sweep — dynamic axes (UFPR-ALPR test, 60 tracks)

Same detector weights throughout; NO retraining on degraded data.
What is measured is the robustness of the fusion layer given identical detections.

Hyper-parameters (selected on validation): `{'iou_thr': 0.2, 'dx_ratio': 0.7, 'dy_ratio': 0.4, 'cooc_thr': 0.6}`

Reproduce with:

```
python tools/robustness.py --axis dynamic
```

## Axis: yaw_ramp — yaw swept 0 -> L across the track (degrees)

| Method | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| B1 single frame | 88.3 | 88.3 | 86.7 | 86.7 | 93.3 | 91.7 | 83.3 |
| B3 original CTM | 96.7 | 98.3 | 95.0 | 86.7 | 86.7 | 71.7 | 33.3 |
| B5 proposed | 98.3 | 95.0 | 93.3 | 90.0 | 86.7 | 83.3 | 81.7 |
| **B5 - B3** | +1.7 | -3.3 | -1.7 | +3.3 | +0.0 | +11.7 | **+48.3** |
| McNemar p | 1.000 | 0.500 | 1.000 | 0.688 | 1.000 | 0.092 | 0.000* |

## Axis: roll_ramp — in-plane rotation swept 0 -> L across the track (degrees)

| Method | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| B1 single frame | 88.3 | 91.7 | 78.3 | 76.7 | 83.3 | 78.3 | 76.7 |
| B3 original CTM | 96.7 | 95.0 | 96.7 | 83.3 | 83.3 | 83.3 | 70.0 |
| B5 proposed | 98.3 | 93.3 | 93.3 | 85.0 | 85.0 | 85.0 | 75.0 |
| **B5 - B3** | +1.7 | -1.7 | -3.3 | +1.7 | +1.7 | +1.7 | +5.0 |
| McNemar p | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 0.508 |

## Axis: zoom_ramp — resolution falling across the track (to px height)

| Method | 0 | 24 | 20 | 16 | 12 | 10 | 8 |
|---|---|---|---|---|---|---|---|
| B1 single frame | 88.3 | 91.7 | 93.3 | 95.0 | 91.7 | 88.3 | 88.3 |
| B3 original CTM | 96.7 | 96.7 | 93.3 | 95.0 | 86.7 | 76.7 | 71.7 |
| B5 proposed | 98.3 | 96.7 | 93.3 | 93.3 | 93.3 | 91.7 | 90.0 |
| **B5 - B3** | +1.7 | +0.0 | +0.0 | -1.7 | +6.7 | +15.0 | **+18.3** |
| McNemar p | 1.000 | 1.000 | 1.000 | 1.000 | 0.289 | 0.004* | 0.003* |

`*` p < 0.05 (exact two-sided McNemar).

## Reading these tables honestly

At mild distortion levels the two methods are indistinguishable, and at a few
levels the original CTM is **ahead** (yaw_ramp 10-20 deg, roll_ramp 5-10 deg,
zoom_ramp 16 px). Those deltas are small and none are significant — but they are
real and they are reported rather than smoothed away.

The separation appears where the thesis predicts it should: at the harsh end of
the dynamic axes, where the inter-frame image transform is furthest from a pure
translation. At yaw_ramp 60 deg CTM collapses to 33.3% while the geometry-
constrained fusion holds 81.7%.

Note also the B1 row. Under heavy yaw, single-frame reading **beats** CTM
(83.3% vs 33.3%). That is the diagnosis in one line: CTM is not merely failing
to help there, its association is actively destroying information that a single
frame still carries.
