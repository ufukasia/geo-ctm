<div align="center">

# Geo-CTM

### Geometry-Constrained Multi-Frame Character Association for License Plate Recognition on Moving Cameras

[![Paper](https://img.shields.io/badge/Paper-Sensors%202026-0b5aa2)](https://www.mdpi.com/1424-8220/26/18/5704)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs26185704-blue)](https://doi.org/10.3390/s26185704)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-3776ab.svg)](pyproject.toml)

**Ufuk Asil** · **İlker Yoncacı**<br>
Department of Software Engineering, OSTİM Technical University, Ankara, Türkiye

*Sensors* **2026**, 26(18), 5704 — [Paper](https://www.mdpi.com/1424-8220/26/18/5704) · [DOI](https://doi.org/10.3390/s26185704) · [Citation](#citation)

</div>

---

This repository is the official implementation of the paper published in
*Sensors*. It contains the Geo-CTM association and fusion
layer, verbatim reproductions of the baselines, and the scripts that
regenerate the benchmark and robustness results.

## Overview

Multi-frame fusion turns per-frame license plate character detections into one
stable reading. **Character Time-Series Matching (CTM)** (Che et al., MAPR 2022)
is a leading approach. It associates characters across frames as *independent
points*, using Hungarian assignment with a fixed 35 px Euclidean gate and a
translation-only motion model, and reports 96.7% accuracy on UFPR-ALPR.

That result depends on the evaluation protocol. A license plate is a **rigid
planar object**: between two frames, all of its character centres move under a
single planar motion (translation, rotation and scale together). On a moving
camera, such as a drone, a helmet-mounted camera or a mobile platform, the
inter-frame geometry changes during the track. A fixed gate combined with
translation-only propagation then fragments tracks, so temporal fusion can
perform *worse* than reading a single frame.

**Geo-CTM** replaces point-wise association with a geometry-constrained
pipeline:

- **height-scaled adaptive matching gates** in place of a fixed pixel threshold,
- **inter-frame similarity-transform estimation** with RANSAC,
- **transform-guided coasting** of unmatched characters,
- **co-occurrence-constrained duplicate track elimination**,

plus frame-quality-weighted voting and score-level plate-grammar decoding.

<p align="center">
<b>Key result.</b> Same detector, same per-frame detections, only the fusion layer changes:<br>
on clean UFPR-ALPR the methods are comparable (96.67% → 98.33%, a difference of one track);<br>
under a 0 → 60° dynamic perspective ramp, CTM drops to <b>33.3%</b> and Geo-CTM holds <b>81.7%</b> (McNemar p &lt; 0.05).
</p>

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the results](#reproducing-the-results)
- [Results](#results)
- [Method](#method)
- [Repository layout](#repository-layout)
- [Licensing](#licensing)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Installation

```bash
git clone https://github.com/ufukasia/geo-ctm.git
cd geo-ctm
pip install -r requirements.txt
python tools/setup_upstream.py
```

`tools/setup_upstream.py` clones the original CTM repository (detector code and
weights) into `third_party/`. It is fetched at install time rather than
included here; [Licensing](#licensing) explains why. The script also prints
where to download the 60 cropped UFPR-ALPR test tracks, which are not stored in
git.

The fusion core (`geoctm.fusion`, `geoctm.decode`, `geoctm.degrade`) depends
only on NumPy, SciPy and OpenCV. It needs neither PyTorch nor the upstream
checkout:

```bash
pip install .                          # fusion core only
pip install ".[benchmark,dev]"         # + benchmark harness and tests
```

Check the fusion core without weights or data:

```bash
python -m pytest tests/test_fusion.py -v
```

## Quick start

### Compare all methods on a single track

```bash
python tools/demo_track.py third_party/Character-Time-series-Matching/test_tracks/track0091
```

```
  B1  single frame                   MLS5511
  B3  original CTM                   MLS5511
  B5  proposed                       MLS5511
```

On clean frames every method agrees, and this is the condition the standard
benchmark measures. Next, apply a dynamic yaw ramp to simulate a moving
camera:

```bash
python tools/demo_track.py <...>/track0093 --degrade yaw_ramp --level 60 --gt AZZ6958
```

```
  B1  single frame                   AZZ6958  correct
  B3  original CTM                   AZZZ958  WRONG
  B5  proposed                       AZZ6958  correct
```

CTM duplicates a `Z` and drops the `6`: a single plate is fragmented into
extra tracks. The single-frame reading is still correct, so under this
distortion CTM's association discards information that one frame alone still
contains.

### Use the fusion layer on your own detections

```python
from geoctm import fuse_track

plate = fuse_track([
    {'dets': [(['A'], [0.91], [10, 4, 22, 30]), ...],   # (labels, confs, xyxy)
     'h': 48, 'w': 160, 'sharp': 120.0},                 # crop size, Laplacian variance
    ...
])
```

Each list element is one frame. `dets` holds that frame's character detections,
`h`/`w` is the plate-crop size, and `sharp` is a sharpness score used for
quality weighting. The default plate grammar is Brazilian (`grammar='brazil'`,
as in UFPR-ALPR).

## Reproducing the results

| Result | Script | Output | Runtime (CPU) |
|---|---|---|---|
| UFPR-ALPR test split, 60 tracks | `python tools/eval_ufpr.py` | `results/ufpr_test.md` | ~2 min first run (detection pass), seconds afterwards (cached) |
| Dynamic robustness sweep | `python tools/robustness.py --axis dynamic` | `results/robustness.md` | slow |
| Confusion prior (B5c ablation) | `python tools/learn_confusion.py --split training` | `confusion_matrix.json` | requires the UFPR training split |
| Reproduction guard | `python -m pytest tests/test_reproduction.py` | — | requires upstream + data |

`tests/test_reproduction.py` checks that the verbatim CTM baseline (B3) still
scores exactly **58/60** and still fails on exactly `track0107` and
`track0136`, which matches the published CTM result. If this test fails, B3 no
longer reproduces the published method and the comparisons are not valid.

## Results

Every row in every table uses the **same detector weights and the same
per-frame detections**. Only the association/fusion layer differs, so detection
quality cannot explain any difference between rows. Hyper-parameters were
selected on the validation split and frozen before the test split was
evaluated (`geoctm.HP_FROZEN`).

| File | Content | Reproducible here |
|---|---|---|
| [`results/ufpr_test.md`](results/ufpr_test.md) | UFPR-ALPR test split (60 tracks), per-track failures, McNemar test | ✅ `tools/eval_ufpr.py` |
| [`results/robustness_dynamic.md`](results/robustness_dynamic.md) | Controlled yaw / roll / zoom ramps | ✅ `tools/robustness.py` |
| [`results/rodosol_cross_dataset.md`](results/rodosol_cross_dataset.md) | Cross-dataset check on RodoSol-ALPR | ⚠️ requires a RodoSol-trained detector |
| [`results/real_traffic_video.md`](results/real_traffic_video.md) | Real moving-camera traffic video | ❌ video cannot be redistributed |

### UFPR-ALPR test split

| Method | Exact match | Accuracy |
|---|:---:|:---:|
| B1 — single frame | 53/60 | 88.33% |
| B2 — CTM association + majority vote | 58/60 | 96.67% |
| **B3 — original CTM (verbatim)** | **58/60** | **96.67%** |
| B5g — G1 geometry matching | 59/60 | 98.33% |
| B5q — G1 + G3 quality weighting | 59/60 | 98.33% |
| **B5 — G1 + G3 + G2 (Geo-CTM, proposed)** | **59/60** | **98.33%** |
| B5n — proposed without adaptive rotation (ablation) | 57/60 | 95.00% |

### Dynamic perspective (yaw ramp 0 → L across the track, UFPR-ALPR test)

| Method | 0° | 10° | 20° | 30° | 40° | 50° | 60° |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B1 single frame | 88.3 | 88.3 | 86.7 | 86.7 | 93.3 | 91.7 | 83.3 |
| B3 original CTM | 96.7 | 98.3 | 95.0 | 86.7 | 86.7 | 71.7 | 33.3 |
| **B5 Geo-CTM** | 98.3 | 95.0 | 93.3 | 90.0 | 86.7 | 83.3 | **81.7** |
| B5 − B3 | +1.7 | −3.3 | −1.7 | +3.3 | +0.0 | +11.7 | **+48.3** |
| McNemar p | 1.000 | 0.500 | 1.000 | 0.688 | 1.000 | 0.092 | 0.000\* |

\* p < 0.05, exact two-sided McNemar. Roll and zoom ramps are in
[`results/robustness_dynamic.md`](results/robustness_dynamic.md).

### Real moving-camera traffic video (as reported in the paper)

The fusion layer is the only variable in each comparison; the tracker is held
fixed. The evaluation uses 86 human-verified tracks.

| Tracker | CTM | Geo-CTM | Δ | Exact McNemar p |
|---|:---:|:---:|:---:|:---:|
| IoU tracker (original CTM system) | 15.1% | **26.7%** | +11.6 pts | 0.021 |
| ByteTrack | 16.3% | **29.1%** | +12.8 pts | 0.013 |

Holding the fusion layer fixed and changing only the tracker moves accuracy by
1–2 points, which is not statistically significant. The paper is the
authoritative source for these figures.
[`results/real_traffic_video.md`](results/real_traffic_video.md) contains the
repository's own measurement on 87 tracks, kept unchanged; its figures differ
slightly from the published ones.

### Limitations and scope

- **On clean UFPR-ALPR the gain is one track** (McNemar p = 1.0), which is not
  statistically significant. The evidence for the method comes from the
  robustness experiments and the real-video results, not from this table.
- **At mild distortion CTM is sometimes slightly ahead** (yaw ramp 10–20°,
  roll ramp 5–10°). These differences are small and not significant. They are
  reported as measured.
- **Absolute real-traffic accuracy is low for both methods.** The comparison
  isolates the fusion layer on difficult footage; it is not a claim of a
  deployable end-to-end system.
- **The robustness sweep is synthetic.** It uses a physically motivated pinhole
  re-projection, but it is not real drone footage. It establishes a conditional
  result: *if* the inter-frame transform departs from pure translation, CTM's
  association degrades and Geo-CTM's does not.
- **A known failure mode of Geo-CTM** (an inserted phantom `1`) is diagnosed in
  [`docs/METHOD.md`](docs/METHOD.md#5-known-failure-modes).

## Method

| Component | Description |
|---|---|
| **G1** — geometry-constrained matching | Two-stage association. A loose gate yields correspondences for fitting an inter-frame **similarity transform** with RANSAC (falling back to translation, then identity, when data is sparse). A tight gate, scaled by character height, then re-matches characters at their transform-predicted positions. Unmatched characters are coasted through the transform rather than by a mean translation. Duplicate tracks are merged by union-find, with a co-occurrence veto. |
| **G2** — score-level grammar decoding | The plate grammar acts on character **class scores** rather than on the finished string, so a look-alike substitution (`0`/`O`, `1`/`I`) is applied only when the accumulated evidence supports it. Optionally uses a confusion matrix learned on the training split. |
| **G3** — quality-weighted voting | Each frame's vote is weighted by resolution and sharpness, so a few sharp close-up frames can outweigh many blurred distant ones. |

CTM's adaptive rotation and its mean-translation coasting are special cases of
the G1 similarity model. A similarity transform is used instead of a homography
because the character centres of a single-row plate are nearly collinear, which
makes homography estimation degenerate. The motion-model ablation in the paper
shows that the similarity transform gives the best-identified model on planar
plates. The full description, evaluation protocol and failure analysis are in
[`docs/METHOD.md`](docs/METHOD.md).

## Repository layout

```
geoctm/            the contribution; no upstream dependency
  fusion.py          G1 matching, G3 weighting, deduplication, reading order
  decode.py          G2 grammar and confusion-matrix decoding
  degrade.py         controlled distortion models (yaw / roll / zoom ramps)
  detect.py          detection pass + cache (requires upstream detector)
  baselines.py       B1 / B2 / B3: verbatim reproductions of the published method
  metrics.py         edit distance, exact McNemar test
  upstream.py        bridge to the cloned CTM repository
tools/             command-line entry points (evaluation, robustness, demo)
tests/             unit tests + reproduction guard
results/           measured results with reproduction commands
docs/METHOD.md     method details, protocol, known failure modes
```

## Licensing

The code in this repository is released under the [MIT License](LICENSE) and
contains **only our own code**.

It intentionally does **not** include the upstream CTM repository, its trained
weights, or its bundled copy of YOLOv5:

- the upstream repository has no LICENSE file, so there is no grant that
  permits redistribution;
- its bundled YOLOv5 is GPL-3.0, which would force this repository to be
  relicensed.

`tools/setup_upstream.py` fetches these components at install time instead. If
you use the baseline comparisons, you use the upstream code under its authors'
terms, and you should cite their paper (see below).

The published article is open access under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this code or build on this work, please cite:

> U. Asil and İ. Yoncacı, "Geometry-Constrained Multi-Frame Character
> Association for License Plate Recognition on Moving Cameras," *Sensors*,
> vol. 26, no. 18, p. 5704, 2026. doi: [10.3390/s26185704](https://doi.org/10.3390/s26185704)

```bibtex
@article{asil2026geoctm,
  author  = {Asil, Ufuk and Yoncac{\i}, {\.I}lker},
  title   = {Geometry-Constrained Multi-Frame Character Association for License Plate Recognition on Moving Cameras},
  journal = {Sensors},
  volume  = {26},
  number  = {18},
  pages   = {5704},
  year    = {2026},
  publisher = {MDPI},
  doi     = {10.3390/s26185704},
  url     = {https://www.mdpi.com/1424-8220/26/18/5704}
}
```

Please also cite the method this work builds on and compares against:

```bibtex
@inproceedings{9924897,
  author    = {Quang, Huy Che and Thanh, Tung Do and Van, Cuong Truong},
  booktitle = {2022 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  title     = {Character Time-series Matching For Robust License Plate Recognition},
  year      = {2022},
  pages     = {1-6},
  doi       = {10.1109/MAPR56351.2022.9924897}
}
```

GitHub's **"Cite this repository"** button (from [`CITATION.cff`](CITATION.cff))
provides the paper citation in APA and BibTeX.

## Acknowledgements

- [Character-Time-series-Matching](https://github.com/chequanghuy/Character-Time-series-Matching), the method this work extends and the source of the baseline detector
- [UFPR-ALPR](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/) and [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) datasets
- [YOLOv5](https://github.com/ultralytics/yolov5)

For questions or issues, please open a
[GitHub issue](https://github.com/ufukasia/geo-ctm/issues).
