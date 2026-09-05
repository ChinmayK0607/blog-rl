# V12 live rollout-gap audit through update 99

Scope: training-distribution rollouts through durable update 99 and compact development rows through update 80. This is neither held-out evidence nor a V12 checkpoint-selection result.

## Main read

- Critical fact-supported target action: 95.8%; mean counterfactual effect +0.0984.
- Decoy private-supported target action: 67.3% overall, improving to 80.0% in updates 80–99.
- Blue-0 decoy failure persists: 60.0% misleading-target obedience in updates 80–99.
- Ordinary focused nonzero advantage: 14.4% overall; blue-0 is 0% and blue-3 is 11.1% in updates 80–99.
- Critical/decoy action direction is clean: active target effect +0.0674 versus alternate-target effect -0.0910.
- Update-80 development: communication +0.0315, specificity +0.0468, RL-specific lift +0.0480; confidence intervals include zero.

`compact_rollout_gap_audit.json` is the reproducible aggregate. `v13_interim_repair_case_selection.json` selects only training cases and must be regenerated after V12 completes.
