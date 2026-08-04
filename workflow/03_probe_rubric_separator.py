"""Probe whether the rubric's word-separator convention behaves as documented.

The rubric claims patterns written ``action.aware`` match hyphenated, spaced, and joined
spellings. In a regular expression ``.`` matches EXACTLY ONE character, so the joined
spelling cannot match. This script checks the concrete cases that matter, above all whether
``mis.bind`` matches "misbinding" -- the central term of the project's own framing.
"""

from __future__ import annotations

import re

CASES = [
    ("mis.bind", ["misbinding", "mis-binding", "mis binding"]),
    ("mis.attribut", ["misattribution", "mis-attribution"]),
    ("action.link", ["action link", "action-link", "actionlink"]),
    ("action.aware", ["action-aware", "action aware", "actionaware"]),
    ("per.action", ["per-action", "per action", "peraction"]),
    ("cross.source", ["cross-source", "cross source", "crosssource"]),
    ("ground.truth", ["ground truth", "ground-truth", "groundtruth"]),
    ("entity.domain", ["entity-domain", "entity domain", "entitydomain"]),
]

FIXED = "[-_\\s]?"

print("=" * 78)
print("Rubric separator probe: '.' (current) vs '[-_\\s]?' (proposed)")
print("=" * 78)
broken = 0
for pattern, samples in CASES:
    for sample in samples:
        cur = bool(re.search(pattern, sample))
        new = bool(re.search(pattern.replace(".", FIXED, 1), sample))
        status = "ok " if cur else "MISS"
        if not cur:
            broken += 1
        print(f"  [{status}] pattern={pattern:<16} sample={sample:<18} "
              f"current={cur!s:<5} proposed={new}")
print("-" * 78)
print(f"Spellings missed by the current convention: {broken}")
print("=" * 78)
