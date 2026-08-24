# V11 diverse receiver scale-up

V11 scales the successful part of v10—receiver behavior changing with teammate
message content—without merely repeating the same twelve small maps for more
updates.

## What changes

- 96 training handoff bundles instead of 12, with two latent worlds each.
- Every ordered sender/receiver role appears eight times and covers graph sizes
  12, 14, 16, 18, and 20 plus all seven source horizons.
- 24 independent development bundles and 36 independent frozen-final bundles.
- One ordinary preservation group in every update; matched decoys remain
  evaluation-only.
- Receiver-only target swapping, terminal team return, four separate policy
  adapters, and the frozen SFT initializer remain unchanged.

The 180-update curriculum contains 540 critical groups and 180 ordinary groups.
With four counterfactual replicas per group, it provides 2,160 focused receiver
samples and 720 ordinary preservation samples. Receiver and sender slots each
receive exactly 135 critical groups; left/right worlds each appear 270 times.

The opponent rotation contains the base model, the SFT initializer, the
development-selected v10 update-40 blue-0 policy, and the current blue-0
snapshot. The historical checkpoint is used as an opponent only; v11 still
starts from the SFT initializer. Because the public v10 export has stale LoRA
rank metadata, prepare the weight-identical corrected view documented in
`V10_CLEAN_HOLDOUT.md` before loading `historical-opponent`.

## Stages

1. Updates 0–31: one remaining turn, 36 bundles. This teaches the direct
   fact-to-action mapping under a strictly certified terminal contrast.
2. Updates 32–91: two remaining turns, 72 bundles. This introduces short
   coordination sequences while the causal signal remains close to the
   receiver action.
3. Updates 92–179: four remaining turns, all 96 bundles. This tests transfer to
   representative multi-turn play.

Every update contains three critical receiver groups and one procedural
ordinary group. There is no message bonus, action bonus, oracle reward, or
learned judge.

## CPU admission result

The construction audit passes over all 96 training bundles:

- 3,870 exact legal teammate-joint-action/opponent-style comparisons;
- minimum exhaustive factual-target terminal advantage `0.08`;
- maximum exhaustive advantage `0.148148`;
- every critical comparison strictly positive;
- every matched decoy advantage exactly zero;
- private worlds indistinguishable without the message;
- identical receiver legal-action sets across worlds and interventions;
- exact sender, receiver, and left/right-world balance;
- no train/development/frozen state-hash overlap.

This proves the curriculum contains clean counterfactual opportunity. It does
not prove the 4B model will sample nonzero advantages at every horizon; that is
measured during the run.

## Evaluation

The compact pulse uses four development bundles covering all four senders and
receivers every 20 updates. Broader development selection runs at updates 60,
120, and 180 over twelve role-complete bundles, 24 hard ordinary cases, and
base/SFT/v10-update40 opponents.

The headline is deliberately small:

1. factual versus receiver-only target-swapped return;
2. that semantic effect beyond SFT;
3. critical versus matched-decoy specificity;
4. receiver target-action switching;
5. ordinary/overall capability preservation.

Dropped, delayed, shuffled, and zero-budget interventions remain useful
diagnostics but are not five additional promotion gates. The 36-bundle frozen
final is opened once for the development-selected checkpoint.

## GPU request

Do not rent a node until the source commit, manifests, trainer config, opponent
pool, runtime certificate, and launch plan are bound together. The intended
node is 4xL40S (or 4xL40) with at least 200 GB free disk. Set auto-termination
to 16 hours initially; refine the deadline after the first ten updates establish
the measured throughput. Public compact mirroring must start at update zero.

The staged launcher must receive:

```bash
export SWARM_DATA_DIR=/workspace/blog-rl/experiments/swarm_arena/data/rl_v11
export SWARM_CURRICULUM_ARTIFACT=$SWARM_DATA_DIR/staged_curriculum_v11_4b_diverse_receiver_180.json
export SWARM_INFERENCE_CONFIG=/workspace/blog-rl/experiments/swarm_arena/configs/inference_4b_l40s.toml
export SWARM_EXPECTED_UPDATES=180
export SWARM_CHECKPOINT_INTERVAL=20
export SWARM_SHARED_RETURN_CREDIT_ASSIGNMENT=focused_agent
export SWARM_WANDB_GROUP=qwen3-4b-diverse-receiver-180
export SWARM_WANDB_MODEL_TAG=4b
```

All other required launch variables remain fail-closed in
`scripts/launch_staged_rl.sh`. The runtime certificate and final production plan
are intentionally created on the rented host because they bind its exact vLLM,
driver, GPU, and constrained-policy calibration hashes.
