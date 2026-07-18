"""Evaluation metrics: edit distance and paired significance testing."""
from math import comb


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(pred, gt):
    """1 - normalised edit distance, in [0, 1]."""
    return 1 - levenshtein(pred, gt) / max(len(gt), len(pred), 1)


def mcnemar(preds_a, preds_b, gts):
    """Exact two-sided McNemar (binomial) on paired predictions.

    Returns (only_a_correct, only_b_correct, p).

    The reviewer's question — "could 58/60 -> 59/60 be chance?" — has to be
    answered without distributional assumptions, so only the DISCORDANT tracks
    are counted. Tracks both methods get right, or both get wrong, carry no
    information about which method is better.
    """
    b = sum(1 for k in gts if preds_a[k] == gts[k] and preds_b[k] != gts[k])
    c = sum(1 for k in gts if preds_a[k] != gts[k] and preds_b[k] == gts[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return b, c, p
