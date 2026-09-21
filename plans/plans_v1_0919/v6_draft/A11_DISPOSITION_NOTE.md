# A11 Disposition Note

**Date:** 2026-09-21
**Re:** research_spec_v6.yaml claims and gates pending validation

This note documents the disposition of audit finding A11 without modifying the
historical research_spec_v6.yaml file.

## Pre-written Claims Downgraded to Hypotheses

The following claims in research_spec_v6.yaml (lines 11-12) are pre-written
assertions that have not been validated with real measurements:

> "it beats end-to-end conditional adaptation on regret and transfers zero-shot
> to new budgets, dictionary members and lead times"

**Status:** Downgraded to unproven hypotheses H1-H3 pending actual measurement:

- **H1 (Regret):** The response-route controller beats end-to-end conditional
  adaptation on selection regret.
- **H2 (Transfer):** The controller transfers zero-shot to new budgets,
  dictionary members, and lead times.
- **H3 (Interactions):** Learned off-diagonal Gram interactions improve
  selection over diagonal-only or scalar-risk baselines.

These hypotheses must be validated through the P1-P3 experiments defined in
research_spec_v6.yaml before being stated as claims in any publication.

## Fixed Gate Threshold Downgraded

The `ge_2_to_3_percent_Z500_72h` gate (line 153) specifies a fixed percentage
improvement threshold:

> `gate: paired_block_bootstrap_significant_and_ge_2_to_3_percent_Z500_72h`

**Status:** Downgraded to "historical reference only".

The actual go/no-go threshold going forward must come from a locally-derived
`threshold_certificate.json` containing a block-bootstrap-derived minimum
detectable effect (delta_MDE). This ensures the threshold reflects the actual
statistical power of the evaluation setup rather than an arbitrary historical
figure.

## Action Required

1. Do NOT use the 2-3% Z500 72h threshold as a hard gate until a
   `threshold_certificate.json` is generated from the pilot data.
2. Treat H1-H3 as hypotheses to test, not validated claims.
3. Update any downstream documentation that references these claims as
   established facts.
