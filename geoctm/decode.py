"""
G2 — format-constrained decoding.

The original CTM applies the plate grammar as an unconditional string edit on
the finished reading: the first three characters get "0" -> "O" and "1" -> "I".
That substitution cannot be undone by evidence — a confident '0' in a letter
position becomes 'O' regardless of how weak the alternative is.

Here the grammar is applied to the character CLASS SCORES instead. Scores of
disallowed classes are FOLDED onto their allowed look-alike (in a letter
position, the score of '1' is added to 'I'), and the argmax is taken over the
folded scores. A look-alike substitution therefore wins only if the accumulated
evidence supports it.

decode_with_confusion goes one step further and replaces the hand-written
look-alike table with a confusion matrix learned from the TRAINING split.
"""

_LOOKALIKE_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A',
                        '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
_LOOKALIKE_TO_DIGIT = {v: k for k, v in _LOOKALIKE_TO_LETTER.items()}

# Grammars: name -> (length, is_letter_position(pos) -> bool)
GRAMMARS = {
    'brazil': (7, lambda pos: pos < 3),          # AAA-NNNN
}


def decode_plain(scores_seq):
    """Unconstrained decoding: argmax at every position."""
    return ''.join(max(sc, key=sc.get) for sc in scores_seq if sc)


def decode_with_grammar(scores_seq, grammar='brazil'):
    """Position-constrained argmax with look-alike score folding."""
    spec = GRAMMARS.get(grammar)
    if spec is None or len(scores_seq) != spec[0]:
        return decode_plain(scores_seq)
    _, is_letter = spec

    out = []
    for pos, sc in enumerate(scores_seq):
        want_letter = is_letter(pos)
        table = _LOOKALIKE_TO_LETTER if want_letter else _LOOKALIKE_TO_DIGIT
        folded = {}
        for k, v in sc.items():
            if k.isalpha() == want_letter:
                folded[k] = folded.get(k, 0.0) + v
            elif k in table:
                folded[table[k]] = folded.get(table[k], 0.0) + v
        out.append(max(folded, key=folded.get) if folded else max(sc, key=sc.get))
    return ''.join(out)


def decode_with_confusion(scores_seq, P, grammar='brazil'):
    """Learned confusion prior + format grammar.

    P[d][c] = P(true = c | detected = d), learned on the TRAINING split by
    tools/learn_confusion.py. Evidence is redistributed at every position:

        score(c) = SUM_d  vote_score(d) * P(c | d)

    This replaces the hand-written look-alike table: which character is confused
    with which comes FROM THE DATA and is weighted by confidence. The grammar is
    applied as a position constraint AFTER the redistribution.
    """
    if not P:
        return decode_with_grammar(scores_seq, grammar)

    spec = GRAMMARS.get(grammar)
    n = len(scores_seq)
    out = []
    for pos, sc in enumerate(scores_seq):
        post = {}
        for d, s in sc.items():
            row = P.get(d)
            if row:
                for c, p in row.items():
                    post[c] = post.get(c, 0.0) + s * p
            else:
                post[d] = post.get(d, 0.0) + s

        if spec is not None and n == spec[0]:
            want_letter = spec[1](pos)
            allowed = {c: v for c, v in post.items()
                       if (c.isalpha() if want_letter else c.isdigit())}
            post = allowed or post

        out.append(max(post, key=post.get) if post else '')
    return ''.join(out)


def load_confusion(path='confusion_matrix.json'):
    """Load a learned confusion matrix, or None if the file does not exist."""
    import json
    import os
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f).get('P')


def crude_norm(text):
    """The ORIGINAL protocol's normalisation, kept verbatim for baseline rows.

    Only used to score B1/B2/B3 exactly as upstream evaluate.py scores them.
    Our own method does not use it — that is the point of G2.
    """
    text = text.replace('-', '')
    return text[0:3].replace('0', 'O').replace('1', 'I') + text[3:]
