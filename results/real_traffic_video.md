# Real traffic video — the regime where the gap is large

UFPR-ALPR makes tracking **free**: one vehicle per frame, tracks handed to you
as folders of crops. We measured it — neither the tracker nor the fusion layer
changes the reading much in that setting, and the benchmark table above shows
exactly that (58/60 vs 59/60).

Real traffic has multiple vehicles, occlusion, and fast approach. This is the
table from that regime, on human-verified ground truth over 87 tracks:

| Metric | CTM (original) | Proposed (GEO) |
|---|---|---|
| Full-plate accuracy (end of track) | 17.2% | **29.9%** |
| Reading changes per track (mean) | 19.52 | **13.44** |
| Confident lock rate | 69% | **73%** |
| Frames to lock (median) | 10 | 10 |

Accuracy: 1.9x, McNemar p = 0.012.

## Two things worth saying plainly

**Both numbers are low.** 29.9% is not a deployable full-plate accuracy. The
comparison is between two fusion layers on hard footage, not a claim that this
system solves real-traffic ALPR.

**The 17.2% is the number most favourable to CTM.** A second re-derivation path
in the renderer scores CTM at 10.3%, because it reads the last detection frame
rather than the end-of-track combined vote. Using the lower figure would nearly
triple our apparent advantage. We report the higher one.

Reading changes per track measures flicker: how many times the displayed reading
changes before the track ends. Fewer is better for anything that has to act on
the reading. Lock rate is how often a confident, stable reading is reached at
all.

The real-traffic pipeline (plate detection, multi-object tracking, ground-truth
scoring) is not part of this repository — it depends on a video that is not ours
to redistribute. The numbers are reported here for completeness; the
reproducible parts of the paper are the UFPR benchmark and the robustness sweep.
