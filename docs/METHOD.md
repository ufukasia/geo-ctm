# Method, protocol, and known failure modes

## 1. The problem with point-wise association

CTM's `matching_char` treats each character as an independent point:

```
cost[i][j] = ||center(track_i) - center(det_j)||
match if cost <= 35 px
unmatched tracks: coast by mean translation dC
```

Two assumptions are buried here.

**A fixed 35 px gate.** A gate in absolute pixels is generous for a distant
plate whose characters are 12 px tall and restrictive for a near one whose
characters are 40 px tall. The correct scale is the character height, which is
observable in every frame.

**Translation-only motion.** `dC` is a pure translation. When the camera's
viewing angle changes across the track, the inter-frame image transform is not a
translation: characters at the two ends of the plate move by different amounts
and in different directions. The gate then fails asymmetrically — it holds at
the centre of the plate and breaks at the ends, which is exactly how a single
plate fragments into extra tracks.

## 2. The geometric constraint

A plate is planar and rigid. Its character centres between two frames are
related by one planar motion. In general that is a homography, but for a
single-row plate the centres are nearly collinear, and homography estimation
from near-collinear points is degenerate.

So we fit the richest model the data can *always* support: a **similarity
transform**

```
T(x) = sRx + t        4 DoF, 2 correspondences suffice
```

estimated by RANSAC (`cv2.estimateAffinePartial2D`), with a model hierarchy that
falls back to translation and then identity when the data is too thin.

CTM's adaptive rotation (the `alpha` accumulator) and its `dC` coasting are both
special cases of this model — rotation-only and translation-only respectively.

### Matching, in two stages

1. **Loose gate** (`max(35, 1.2 * char_h)`) — Hungarian assignment, used only to
   obtain correspondences good enough to fit the motion model. It is deliberately
   permissive; wrong matches here cost little because RANSAC discards them.
2. **Tight gate** (`max(10, 0.60 * char_h)`) — every stored character is
   predicted through the fitted transform and re-matched. The gate now scales
   with character height.

Unmatched stored characters are coasted **through the transform**, including its
scale factor, rather than by a mean translation.

### Duplicate rejection

Two mechanisms, at two moments.

**During matching:** an unmatched detection that overlaps an existing track
strongly (IoU >= 0.25) contributes its vote to that track instead of opening a
new one. IoU rather than centre distance, because for a narrow glyph such as `1`
the *neighbouring* character's centre can also be close, which would attribute
the vote to the wrong character.

**After matching (`dedupe`):** union-find merging of tracks, guarded by a
**co-occurrence veto**. Two tracks that both received a vote in the same frame
correspond to two separate detections in that frame, so they are almost
certainly different characters. Duplicate tracks are born when one dies and
another opens, so their frame sets are largely disjoint. Measured on UFPR the
separation is sharp:

| pair type | frames co-occurring | IoU | dx |
|---|---|---|---|
| genuine neighbours | 93-100% | <= 0.14 | >= 11 px |
| duplicates | 23-43% | 0.42-0.83 | <= 0.46 char widths |

If the co-occurrence ratio exceeds `cooc_thr = 0.60` the merge is refused, even
when the geometry says the two boxes sit on top of each other.

The upstream `merge_box_arr_track` merges greedily in a chain, growing its
bounding box as it goes, which made it swallow genuine neighbouring characters
(observed on track0129: `A` + `K`). Union-find over groups fixes that.

## 3. Voting and decoding

**G3.** Each frame's vote is weighted by `sqrt(resolution x sharpness)`, both
normalised within the track. A sharp close-up frame counts for more than a
blurred distant one. Floor of 0.05, so no frame is fully silenced.

**G2.** The original code applies the plate grammar as a string edit on the
finished reading: in the first three characters, `0 -> O` and `1 -> I`,
unconditionally. That substitution cannot be overruled by evidence.

Here the grammar acts on class **scores**. In a letter position the score of `1`
is folded onto `I`, and the argmax is taken over the folded scores — so a
look-alike substitution wins only when the accumulated evidence supports it.
`decode_with_confusion` replaces the hand-written look-alike table with
`P(true | detected)` learned from the training split.

## 4. Experimental protocol

**Frozen detector.** Every benchmark row uses the same detector weights and the
same cached per-frame detections. Only the association/fusion layer varies. A
difference in the table cannot be attributed to detection quality.

**Split discipline.** The confusion matrix is learned on TRAINING. The dedupe
thresholds are selected by grid search on VALIDATION (30 tracks) and frozen
before test is touched — `geoctm.HP_FROZEN`. Results are reported on TEST. Among
tied grid points the most central is chosen, since a hyper-parameter sitting on
a threshold edge is one that has been overfitted.

**Verbatim baseline.** B3 is a line-by-line reproduction of upstream
`evaluate.py`, including its asymmetric coordinate rounding. It is not tidied.
`tests/test_reproduction.py` asserts it still scores 58/60 and still fails on
`track0107` and `track0136`.

**Significance.** Exact two-sided McNemar on paired predictions, counting only
discordant tracks.

### A known quirk of the protocol: frame ordering

Track frames are read in plain lexicographic filename order, matching upstream
`evaluate.py` exactly. This is **not** temporal order — frames are named
`track0091[2].png`, `track0091[10].png`, so `[10]` sorts before `[2]`.

Every number reported here, ours and the baseline's, was produced under this
ordering, so the comparison is internally consistent and fair. We keep it
because changing it would silently invalidate the reproduction of the published
96.67%.

It is worth stating that this probably *understates* both methods: a temporal
method given scrambled frame order is working harder than it needs to. Whether
natural sort changes the ranking is an open question we have not measured.

## 5. Known failure modes

### Ours: the phantom digit

The fusion sometimes inserts a spurious character, always a `1`:

```
59D303340  ->  59D3103340
51G65818   ->  51G658181
```

Diagnosis: a shadow track survives the co-occurrence veto and its votes are
counted as a separate character position. The veto is a threshold on frame-set
overlap, and a short-lived duplicate that appears in few frames can fall below
it while still being a duplicate. `1` dominates because it is the narrowest
glyph — the geometric duplicate test is weakest exactly where the box is
thinnest.

This is not fixed. It is documented because it is the honest bound on the
method, and because a fix that suppresses it by tightening the veto would
re-merge genuine neighbouring characters.

### Theirs: dropping and duplication

The original CTM's real-traffic failure modes are character **dropping**
(`93P192626 -> 9P1926`), character **duplication**
(`59V193547 -> 59V119335477`), and track fragmentation. The first two are the
two directions of the same association failure.

### Both: the scale axis is a tie

On the pure resolution axis neither method separates from the other, which is
the expected result: shrinking a plate uniformly leaves the inter-frame
transform close to a similarity, so CTM's assumption is not violated and there
is nothing for the geometry constraint to repair. Reporting it as a tie is part
of the argument, not a hole in it.

## 6. What the robustness sweep does and does not prove

The distortions are **synthetic**. They are physically motivated — a pinhole
re-projection of the plate plane, a real motion-blur kernel, genuine resolution
loss — but they are not drone footage.

What the sweep establishes is a *conditional*: **if** the inter-frame transform
departs from translation, **then** CTM's association degrades and the
geometry-constrained version does not. The mechanism is identified and measured.

What it does not establish is how often that condition holds in any particular
real deployment. The real-traffic video results are the closest evidence we have
for that, and they are limited to one camera and 87 tracks.
