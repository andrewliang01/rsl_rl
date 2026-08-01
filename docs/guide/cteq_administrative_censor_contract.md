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

The NumPy `administrative_censor_survival_loss` and Torch
`CteqAdministrativeSurvivalLoss` consume the same four label tensors. For an
event in bin `j`, they score the prior survival and hazard at `j`. For a censor
boundary `m`, they score survival only through `[0,m)`. Their censored Brier
diagnostic collapses all probability at and after `m` into `S(m)`; at `m=25`
it is exactly the original full-horizon 26-class Brier score.

Every eligible foot/event target keeps equal weight. Zero-exposure targets are
zeroed and excluded from both numerator and denominator; an all-zero eligible
batch is rejected because its mean loss is undefined. Per-role counts and
NLL/Brier sums make the denominator auditable.

This remains a CPU contract primitive. It is not connected to the actor, Gym,
PPO, or GPU execution. Training remains disabled until the real runner proves
the exact `done`/`time_outs`/base-contact mapping and whether the terminal
contact sample is valid and included in `fully_observed_bins`.

## On-policy termination provenance

`build_cteq_on_policy_termination_provenance` freezes the required boundary
between an auto-reset environment and the administrative label builder. In
addition to `dones`, the environment-owned `extras` must provide:

- `time_outs`, base-contact termination, and other termination masks;
- a reset-before terminal foot-contact sample and its validity mask;
- the episode ID before the step, the episode ID attached to terminal contact,
  and the post-step episode ID.

The three termination masks must be mutually exclusive and exhaustive. Every
done row requires a valid terminal contact whose episode ID equals the
pre-reset ID, while its post-step ID must be strictly newer. Non-done rows must
remain in the same episode and carry no latent terminal contact. These checks
prevent reset-state contact from being appended to the previous episode's
future TD/LO trace.

The stock RSL on-policy interface currently guarantees only `dones` and an
optional `extras["time_outs"]`; it does not guarantee the remaining fields.
Missing terminal contact or episode IDs therefore fail closed. A validated
provenance batch can authenticate the matching termination masks in the
administrative-censor receipt, but `training_ready` remains false until a real
environment adapter emits and receipts this exact interface.
The later collector must additionally receipt the anchor-to-terminal sample
count used as `fully_observed_bins`; this step-level provenance contract does
not infer that count from reset observations.

## Opt-in IsaacLab recorder bridge

The companion Lab-Pro adapter can use IsaacLab's recorder ordering without
patching IsaacLab: `record_post_step()` runs after termination computation and
before `record_pre_reset()`/`_reset_idx()`. Its disabled-by-default recorder
captures ordered left/right foot contact, the individual `base_contact` term,
raw timeout, and monotonic episode IDs at that boundary. A simultaneous raw
timeout and MDP termination is classified by the MDP reason for CTEQ labels,
while the original `extras["time_outs"]` is left untouched for PPO.

`build_cteq_isaaclab_termination_batch` is an explicit CPU bridge. GPU tensors
are rejected unless the caller passes `allow_device_transfer=True`, and every
transferred field is listed in the receipt. The result is marked as privileged
truth and rejects actor-observation, critic-hidden-state, and reward access.
Neither the stock runner nor any registered Gym task calls this bridge.

The recorder does **not** infer `fully_observed_bins` from episode length. It
requires an environment-owned provider to prove the number of complete 20 ms
post-anchor contact samples observed before the boundary, together with a
provider source SHA and semantic receipt. No such real collector is currently
implemented. Without it the Lab-Pro recorder omits those extras and this bridge
fails closed. Consequently `training_ready=false` remains mandatory.
