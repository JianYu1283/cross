# -*- coding: utf-8 -*-
"""Apply reviewer text/format fixes to paper_draft.md (run once, idempotent)."""
import io

P = 'paper/paper_draft.md'
s = io.open(P, encoding='utf-8').read()

edits = [
    # (old, new, label)
    ('plain HAN', 'MANDO-style HAN re-implementation', 'scope "plain HAN"'),
    ('three node types and six edge relations',
     'three node types and five edge relations', 'five edge relations'),
    ('`balances``[``msg.sender``]``\xa0=\xa00`',
     '`balances[msg.sender] = 0`', 'balances typo'),
    ('0.043 F1**.',
     '0.043 F1** under rich features (low-level call on).', 'abstract 0.951 config'),
    ('but the collapse is architecture-specific.',
     'but the collapse is specific to this HAN re-implementation.', 'finding-3 subtitle'),
    ('best described as uncertain.',
     'best described as uncertain. Under Holm correction across the six architecture '
     'comparisons, only the one-hot/lowcall-on difference survives (adjusted p < 0.05); '
     'the two borderline differences (p = 0.017 and p = 0.020) do not survive, '
     'reinforcing the uncertain reading.', 'Holm correction'),
    ('The cross-contract representation does not further improve detection accuracy on this task',
     'Adding inter-contract call edges (the cross-contract graph representation) does not '
     'further improve detection accuracy on this task', 'cross-contract wording (a)'),
    ('attack-path reconstruction and the benchmark, not in raw accuracy.',
     'attack-path reconstruction, while the cross-contract benchmark (Section 5.2) lies '
     'in exposing the ordering gap, not in raw accuracy.', 'cross-contract wording (b)'),
    ('collapsing from +27/+35 points to +4 points as features grow richer.',
     'collapsing from +27/+35 points to +4 points as features grow richer; the rich-regime '
     '+4.1-point effect has bootstrap 95% CI [+2.1, +6.5] points.', 'fig5 CI'),
    ('because the benchmark is saturated by the unchecked low-level call signal.',
     'because the benchmark is saturated by the unchecked low-level call signal (SmartCheck '
     'is used as an order-unaware SOLIDITY_UNCHECKED_CALL proxy, not as a dedicated '
     'reentrancy detector).', 'fig4 proxy label'),
    ('and the source code of the proposed method are available from the corresponding '
     'author upon reasonable request.',
     'and the source code of the proposed method will be released publicly upon acceptance.',
     'data availability'),
    ('The pipeline, benchmark, and baselines are available from the corresponding author '
     'to support reproduction.',
     'The pipeline, benchmark, and baselines will be released publicly upon acceptance to '
     'support reproduction. Our contribution is a controlled mechanism study of which '
     'representation factors drive detection, rather than a deployable detector, which would '
     'additionally require order-sensitive reasoning.', 'conclusion'),
    ('The CodeBERT and GraphCodeBERT baselines truncate inputs to 512 tokens, discarding '
     'the tail of long contracts.',
     'The CodeBERT and GraphCodeBERT baselines truncate inputs to 512 tokens, discarding '
     'the tail of long contracts. The transformer and MANDO-HAN baselines use 3 seeds versus '
     '5 for HGT and GAT (reflecting transformer fine-tuning cost and the retention of MANDO '
     'hyperparameters), so their variance estimates are less stable than the HGT and GAT '
     'estimates.', 'seed note'),
]

ok = 0
for old, new, label in edits:
    n = s.count(old)
    if n == 0:
        print('NOT FOUND: ' + label)
    else:
        s = s.replace(old, new)
        ok += 1
        print(f'OK ({n}x): ' + label)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
print(f'done: {ok}/{len(edits)} edits applied')
