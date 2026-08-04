"""Replace the README's trailing 'Next step' section with the Step 4 handoff."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

NEW_SECTION = """## Next step

**Step 4 — AegisLink method and baselines** (`CONTRACT.md` Sections 9, 10). Implement
`AegisLink.verify(e, d, a) -> VERIFIED | PLAUSIBLE | UNVERIFIED | CONTRADICTED` over the seven
required evidence families, with the six-component minimum architecture and the seven required
ablations, plus the ten baselines. Section 9 forbids a simple weighted sum of URL reputation
signals.

Six Step 3 outputs constrain Step 4 directly:

1. **Consume evidence only through `web_rag.exposure` at `VERIFIER` tier.** `role`,
   `adversarial` and `lifecycle.prior_grants` are unavailable *by construction*, not by
   convention. Use `load_public_domain_registry()` for descriptors, the derived
   `LifecycleObservation`s for the RQ3 lifecycle family, and `public_delegation_evidence()` for
   backlink *addressing* — the actions must still be read off the page.
2. **Replay from `data/benchmark/retrieval_snapshots.json` and check the `replay_fingerprint`
   first.** Every baseline must see identical evidence (Section 10). A mismatch means an input
   drifted and the run must stop rather than mix evidence.
3. **Read pages at `phase="live"`.** Reading the indexed phase would silently disable
   `content_change_after_indexing` for the whole experiment.
4. **Implement `web_rag.trace_recorder.StagePipeline`.** RQ2 attribution then comes for free, and
   the real reader is expected to populate the `inferred_action_authorizations` and
   `presented_links` arms that the Step 3 surrogate leaves empty.
5. **Call `ExperimentMatrix.require_admitted()` before any run.** Fit the `tau` thresholds on
   `train_development` / `validation` only; rule R3 refuses a threshold-selection run that
   touches `test` or either holdout. Adaptive Stratum B is `secondary_evaluation` only.
6. **Substituting a pinned open-weight encoder requires re-freezing the snapshots.**
   `encoder_id` is inside the fingerprint, so the change is detected rather than absorbed.

One measurement from Step 3 is worth carrying into the method design: an authorized domain is
present in the top 10 for 84-95% of queries but reaches rank 1 for only 0-52% of them depending
on the action. Retrieval recall is therefore not the binding constraint — the misbinding
opportunity is created by ranking, so a verifier that reranks on authorization evidence has real
headroom, and `ATPR` is at risk mainly from over-rejection rather than from evidence being
absent.
"""


def main() -> int:
    text = README.read_text(encoding="utf-8")
    marker = "## Next step"
    idx = text.rindex(marker)
    README.write_text(text[:idx] + NEW_SECTION, encoding="utf-8")
    print(f"README next-step section rewritten ({len(NEW_SECTION)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
