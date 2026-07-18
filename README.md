# geo-ctm — Geometry-Constrained Character Fusion for Video License Plate Recognition

Reference implementation for the paper *(under review)*.

Character Time-series Matching (CTM, Che et al., MAPR 2022) fuses per-frame
character detections into one plate reading. It matches characters between
frames as **independent points**: a Hungarian assignment on centre distance with
a fixed 35 px gate, and unmatched characters coasted by the mean translation.

But a license plate is a **rigid plane**. Its character centres move between
frames under a single planar motion — translation, rotation and scale together —
not under 35 px of independent slack per character. This repository replaces the
point-wise association with a RANSAC-estimated similarity transform, adds
frame-quality weighted voting and score-level grammar decoding, and measures
what that changes.

**Headline result:** on the standard UFPR-ALPR benchmark the two methods are
nearly tied (96.67% -> 98.33%, a single track). Under *dynamic* geometric
distortion — a camera whose viewing angle changes across the track, which is
what a moving or airborne camera actually does — CTM collapses to 33.3% while
this method holds 81.7%.

The benchmark everyone reports on cannot see the difference. That is the point
of the paper.

---

## Install

```bash
git clone https://github.com/ufukasia/geo-ctm.git
cd geo-ctm
pip install -r requirements.txt
python tools/setup_upstream.py
```

`setup_upstream.py` clones the original CTM repository into `third_party/`.
We do not vendor it — see [Licensing](#licensing) for why. It also tells you
where to download the 60 cropped UFPR-ALPR test tracks, which are not in git.

Verify the fusion core with no weights and no dataset:

```bash
python -m pytest tests/test_fusion.py -v
```

## Use

Read one track and compare every method:

```bash
python tools/demo_track.py third_party/Character-Time-series-Matching/test_tracks/track0091
```

```
  B1  single frame                   MLS5511
  B3  original CTM                   MLS5511
  B5  proposed                       MLS5511
```

On clean frames every method agrees — this is the regime the standard benchmark
measures. Now break the camera geometry the way a moving camera does:

```bash
python tools/demo_track.py <...>/track0093 --degrade yaw_ramp --level 60 --gt AZZ6958
```

```
  B1  single frame                   AZZ6958  correct
  B3  original CTM                   AZZZ958  WRONG
  B5  proposed                       AZZ6958  correct
```

CTM duplicates a `Z` and drops the `6` — one fragmented track, read twice. Note
that single-frame reading also gets it right: under this distortion CTM's
association is not merely failing to help, it is destroying information a single
frame still carries.

Full benchmark (~2 min for the first detection pass, seconds thereafter):

```bash
python tools/eval_ufpr.py                    # -> results/ufpr_test.md
python tools/robustness.py --axis dynamic    # -> results/robustness.md  (slow)
```

Use the fusion directly on your own detections:

```python
from geoctm import fuse_track

plate = fuse_track([
    {'dets': [(['A'], [0.91], [10, 4, 22, 30]), ...],   # (labels, confs, xyxy)
     'h': 48, 'w': 160, 'sharp': 120.0},                 # crop size, Laplacian var
    ...
])
```

`geoctm.fusion`, `geoctm.decode` and `geoctm.degrade` depend only on numpy,
scipy and OpenCV — no PyTorch, no upstream checkout.

## Results

| | | |
|---|---|---|
| [`results/ufpr_test.md`](results/ufpr_test.md) | UFPR-ALPR, 60 tracks | reproduce: `tools/eval_ufpr.py` |
| [`results/robustness_dynamic.md`](results/robustness_dynamic.md) | controlled distortion sweep | reproduce: `tools/robustness.py` |
| [`results/rodosol_cross_dataset.md`](results/rodosol_cross_dataset.md) | RodoSol-ALPR generalisation | needs a RodoSol-trained detector |
| [`results/real_traffic_video.md`](results/real_traffic_video.md) | real traffic video, 87 tracks | video not redistributable |

### UFPR-ALPR test split

| Method | Exact match | Accuracy |
|---|---|---|
| B1 single frame | 53/60 | 88.33% |
| B2 CTM association + majority vote | 58/60 | 96.67% |
| **B3 original CTM (verbatim)** | **58/60** | **96.67%** |
| B5g G1 geometry matching | 59/60 | 98.33% |
| B5q G1 + G3 quality weighting | 59/60 | 98.33% |
| **B5 G1 + G3 + G2 — proposed** | **59/60** | **98.33%** |
| B5n proposed, no adaptive rotation | 57/60 | 95.00% |

Every row consumes the **same detector weights and the same per-frame
detections**; only the association/fusion layer differs. A difference in this
table cannot come from a better detector, because there is only one detector.

`tests/test_reproduction.py` asserts that B3 still scores exactly 58/60 and
still fails on exactly `track0107` and `track0136` — the published result. If
that ever breaks, the baseline is no longer the published method and every
comparison here is void.

### Honest reading of the numbers

- **On UFPR the improvement is one track.** McNemar p = 1.0. It is not
  statistically significant and we do not claim it is. The evidence for the
  method is the robustness curves, not this table.
- **At mild distortion the original CTM is sometimes ahead** (yaw_ramp 10-20°,
  roll_ramp 5-10°). Small, not significant, and reported rather than smoothed.
- **Real-traffic accuracy is low for both methods** (17.2% vs 29.9%). That is a
  comparison of fusion layers on hard footage, not a deployable system.
- Where CTM is reported at 17.2%, a second derivation path gives 10.3%. We use
  the higher number, which is the one less favourable to us.

## What is actually new

| | |
|---|---|
| **G1** | Similarity-transform (RANSAC) constrained matching. Two stages: a loose gate to fit the motion model, then a tight character-height-scaled gate on transform-predicted positions. Unmatched tracks coast through the transform instead of a mean translation. Duplicate tracks are merged by union-find, vetoed by a co-occurrence test. |
| **G2** | The plate grammar applied to character **class scores** rather than to the finished string. A look-alike substitution wins only if the accumulated evidence supports it. Optionally driven by a confusion matrix learned on the training split. |
| **G3** | Votes weighted by frame resolution and sharpness, so three sharp frames can outvote five blurred ones. |

CTM's adaptive rotation and mean-translation coasting are special cases of the
G1 transform model. Details and the failure analysis are in
[`docs/METHOD.md`](docs/METHOD.md), including a failure mode of **our** method
that the paper documents (an inserted phantom digit).

## Repository layout

```
geoctm/          the contribution — no upstream dependency
  fusion.py        G1 matching, G3 weighting, dedupe, reading order
  decode.py        G2 grammar and confusion decoding
  degrade.py       controlled distortion models
  detect.py        detection pass + cache (needs upstream detector)
  baselines.py     B1/B2/B3 — verbatim reproductions of the published method
  metrics.py       edit distance, exact McNemar
  upstream.py      bridge to the cloned CTM repo
tools/           command-line entry points
tests/           unit tests + the reproduction guard
results/         measured results, with reproduction commands
docs/METHOD.md   method, protocol, and known failure modes
```

## Licensing

This repository is MIT licensed. It contains **our** code only.

It deliberately does **not** vendor the upstream CTM repository, its trained
weights, or its bundled copy of YOLOv5:

- the upstream repository ships no LICENSE file, so there is no grant that would
  permit redistribution;
- its bundled YOLOv5 is GPL-3.0 upstream, which would relicense this repository.

`tools/setup_upstream.py` fetches them at install time instead. If you use the
baseline comparisons, you are using their code under whatever terms they offer
it, and you should cite their paper.

## Citation

```bibtex
@article{geoctm2026,
  title  = {Geometry-Constrained Character Fusion for Video License Plate Recognition},
  author = {Asil, Ufuk and others},
  note   = {Under review},
  year   = {2026}
}
```

Please also cite the method this work builds on and compares against:

```bibtex
@INPROCEEDINGS{9924897,
  author    = {Quang, Huy Che and Thanh, Tung Do and Van, Cuong Truong},
  booktitle = {2022 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  title     = {Character Time-series Matching For Robust License Plate Recognition},
  year      = {2022},
  pages     = {1-6},
  doi       = {10.1109/MAPR56351.2022.9924897}
}
```

## Acknowledgements

- [Character-Time-series-Matching](https://github.com/chequanghuy/Character-Time-series-Matching) — the method this work extends
- [UFPR-ALPR](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/) and [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) datasets
- [YOLOv5](https://github.com/ultralytics/yolov5)
