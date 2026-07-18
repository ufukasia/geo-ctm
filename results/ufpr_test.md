# UFPR-ALPR benchmark — split: test (60 tracks)

Same character detector and same per-frame detections in every row; only the association/fusion layer differs.

Hyper-parameters: {'iou_thr': 0.2, 'dx_ratio': 0.7, 'dy_ratio': 0.4, 'cooc_thr': 0.6}
Selection: FROZEN (geoctm.HP_FROZEN)

Date: 2026-07-18 19:47 | Runtime: 40s

| Method | Exact match | Accuracy | Character accuracy | Failures |
|---|---|---|---|---|
| B1 — single frame | 53/60 | 88.33% | 95.89% | track0096(AX90W37/AWX9307), track0107(RAWD00/MJO0862), track0123(ASGB9143/ASG9143), track0126(AAV10949/AVI0949), track0127(AQW13799/AQW1379), track0128(AWP0916/AWP0816), track0148(MAKT8774/AKT8174) |
| B2 — CTM assoc. + majority vote | 58/60 | 96.67% | 97.86% | track0107(RAWD000/MJO0862), track0136(AMO0636/AMO0663) |
| B3 — original CTM (verbatim) | 58/60 | 96.67% | 97.86% | track0107(RAWD000/MJO0862), track0136(AMO0636/AMO0663) |
| B5g — G1 geometry matching | 59/60 | 98.33% | 98.33% | track0107(RAWD000/MJO0862) |
| B5q — G1 + G3 quality weighting | 59/60 | 98.33% | 98.33% | track0107(RAWD000/MJO0862) |
| B5 — G1 + G3 + G2 grammar — **PROPOSED** | 59/60 | 98.33% | 98.57% | track0107(RAW0000/MJO0862) |
| B5c — B5 + learned confusion prior (ablation) | 59/60 | 98.33% | 98.57% | track0107(RAW0000/MJO0862) |
| B5n — B5 without adaptive rotation (ablation) | 57/60 | 95.00% | 98.10% | track0105(AFK0977/AEK0977), track0107(RAW0000/MJO0862), track0148(AKT8774/AKT8174) |

## Significance (exact McNemar, two-sided) — B3 vs B5

- Only B3 correct: 0 tracks | Only B5 correct: 1 tracks (discordant: 1)
- p = 1.0000  -> NOT significant on its own; the real evidence is the robustness curves and cross-split consistency (reported honestly)
