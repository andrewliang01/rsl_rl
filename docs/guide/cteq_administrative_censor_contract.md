# CTEQ administrative-censor contract

## Why the original right-censor flag is insufficient

The original CTEQ label builder only accepts a complete 25-bin (0.5 s)
forecast window. Its `right_censored=true` target means that the selected
touchdown or liftoff did not occur in any of those 25 bins. The current
survival loss therefore scores survival through the complete horizon.

An episode ending after, for example, six observed bins has different
semantics. If no event was observed, the only valid statement is survival
through bins `[0, 6)`. Relabelling this as ordinary horizon censoring would
incorrectly assert survival through another 19 unobserved bins. Relabelling
the termination itself as touchdown or liftoff would also be wrong.

## Fail-closed adapter inputs

`build_cteq_administrative_censor_batch` accepts only:

- debounced foot-contact touchdown/liftoff bins already observed before the
  boundary (`-1` means no observed event);
- the exact number of complete post-anchor bins with usable contact truth;
- an exhaustive, mutually exclusive episode-boundary taxonomy:
  time limit, base contact, or another non-time-limit early termination;
- an optional runner provenance SHA-256 receipt.

A partial nonterminal rollout is rejected. Event bins at or beyond the audited
boundary are rejected. A termination-source string cannot be substituted for
the debounced foot-contact source contract.

## Outputs

Each `[batch, foot, event-type]` target contains:

- `event_bin` and `event_observed` for a real touchdown/liftoff only;
- `right_censored`;
- `censor_after_bin`, the number of fully observed event-free bins;
- `loss_eligible`, false for zero-exposure boundaries;
- a stable reason code distinguishing natural horizon censoring, time-limit
  administrative censoring, base-contact censoring, and other early
  termination censoring.

The audit receipt records per-reason target counts, per-reason episode counts,
the label tensor hash, runner provenance state, and the invariant
`termination_is_event=false`.

## Current boundary

This change is a CPU label-contract primitive only. It is not connected to the
actor, Gym, PPO, or GPU execution. The existing full-horizon CTEQ loss does not
consume `censor_after_bin`; an administrative-censor-aware survival loss is a
separate required step. Training remains disabled until the real runner proves
the exact `done`/`time_outs`/base-contact mapping and whether the terminal
contact sample is valid and included in `fully_observed_bins`.
