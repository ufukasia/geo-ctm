# RodoSol-ALPR — cross-dataset generalisation

Does the core finding hold on a second dataset, with a different country,
plate format and camera?

RodoSol is a **single-frame** dataset and our contribution is multi-frame, so a
controlled camera-approach sequence is synthesised from each plate patch
(perspective + scale ramp) and the same robustness question is asked. The
character detector here is a YOLO11n trained on RodoSol (36 classes) — the
UFPR-trained detector reads RodoSol at 0% and would have measured nothing.

## cars-br (120 plates)

| Distortion | B1 single frame | B3 CTM | B5 proposed |
|---|---|---|---|
| clean | 93.3% | 93.3% | 93.3% |
| yaw 30° | 85.8% | 66.7% | **86.7%** |
| yaw 50° | 4.2% | 26.7% | **65.8%** |
| scale 14 px | 92.5% | 93.3% | 93.3% |
| scale 10 px | 92.5% | 94.2% | 94.2% |

## cars-me (120 plates)

| Distortion | B1 single frame | B3 CTM | B5 proposed |
|---|---|---|---|
| clean | 87.5% | 87.5% | 87.5% |
| yaw 30° | 80.0% | 70.8% | **86.7%** |
| yaw 50° | 0.0% | 14.2% | **55.0%** |
| scale 14 px | 86.7% | 87.5% | 87.5% |
| scale 10 px | 86.7% | 87.5% | 87.5% |

## What this does and does not show

It shows the finding is not an artefact of UFPR: on a different country's plates
with a differently trained detector, the same pattern appears — clean frames are
a tie, and the gap opens under dynamic perspective.

It does **not** show performance on real RodoSol video, because no such video
exists in the dataset. The sequences are synthetic. The scale axis is a tie
throughout, which is the honest expectation: shrinking a plate uniformly leaves
the inter-frame transform close to a similarity, so CTM's assumption is not
violated and there is nothing for the geometry constraint to fix.
