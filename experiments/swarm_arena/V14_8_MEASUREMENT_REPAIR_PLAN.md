# V14.8 measurement-repaired A6000 run

The user explicitly authorized using the available 4×A6000 and extending the
time budget on 2026-09-05. Keep the $60 ceiling; permit a 30-hour TTL at
$1.68/hour ($50.40 maximum). Terminate early on rejection, terminal fault,
deadline/budget trigger, or completion. Never rent a second concurrent pod.

This is a fresh run from the same four distinct V13 update-80 adapters, not a
resume of the incomplete V14.7 evaluation. The repaired initializer evaluation
and separate SFT reference require a fresh evaluation identity. Preserve the
192-row update-zero evaluation and stage gates 10/20/30/40. Keep all training
data, reward, optimizer, scheduler, dtypes, parity thresholds and one-quarantine
per-stage semantics unchanged. The optional semantic probe is not part of this
run. No frozen held-out data is opened.

## Admission and timing

Use GPU0 for training and GPUs1,2,3 for isolated inference. Before heavy setup,
install exact-pod terminal, idle, deadline and recovery guards. A fresh 128-sample
parity probe at four concurrent requests per server must pass pooled and every
policy-local gate. Bind the clean public source, CPU bundle, configs, all four
adapters and hardware to a fresh runtime certificate and production plan.
Require working private HF/W&B authentication and compact mirror preflights.

Prior A6000 evaluation throughput was approximately 1.3 games/minute. Do not
call a training-speed estimate measured: no durable update timing exists for
this exact repaired stack. The user-authorized extended rental instead reserves
900 seconds per logical update, with a 1.25 safety factor, 600 seconds per
checkpoint and 2700 seconds for final synchronization. With one hour of remaining
setup, the 672-game, 40-update schedule reserves approximately 25.86 hours.
This is a conservative allocation, not a throughput guarantee or GPU certificate.
Recompute against the actual provider deadline before launch and monitor actual
progress against remaining time. Never extend a failing scientific gate or
resample a failed slot to meet the schedule.

The `staged-reservation-profile-v1` explicitly labels training timing as
unmeasured and records the user's extended-time authorization. It does not
masquerade as `staged-operational-profile-v1`. Any later performance report uses
operational telemetry only, not rewards or gate results. A process failure leaves
a durable terminal marker; recovery does not automatically restart from zero.

## Identity and claim boundary

The CPU bundle hashes the repaired runtime source and preserves the parent
scientific identities. Its CPU status does not authorize skipping exact-host
certification. Publish and independently fetch/hash-verify the source and bundle
before launch. Report useful message use, RL-specific improvement and ordinary
gameplay separately; a beautiful presentation is not evidence of a positive result.
