# Swarm Arena research log

Last updated: 2026-08-16
Branch: `exp/swarm-arena-4b`  
Current public development checkpoint: Hugging Face revision
`1af877668ee3cdd8dd5ccd4734ce620bbe5e2aa0` (not admitted).
Status: the fresh 30-update four-policy RL v4 run and development diagnostics
are complete. The selected step-20 checkpoint is mechanically stable and
public; capability and RL-specific communication improvement are not
established, and selection/frozen final remain unopened.

Next run: the CPU-side 120-update staged curriculum and RL-specific progress
metrics are implemented and awaiting a fresh four-GPU launch preflight.

This is the durable chronological record for the Swarm Arena project. It records
the hypothesis, design decisions, data, training, evaluations, failures,
infrastructure incidents, costs when known, and the reason for every major
change. It is intentionally candid: a technically completed run is not called a
research success unless it passes its predeclared scientific gate.

## Logging contract

Every material future run must append an entry before its result is used. Each
entry must include:

- date, operator, source commit, and immutable model/data revisions;
- hypothesis and the decision the run is intended to unlock;
- exact train/development/final split and whether it was previously inspected;
- GPU type/count, wall time, and estimated cost when available;
- command or launcher, configuration, seeds, and output artifact hashes;
- protocol, capability, communication, regression, and safety results;
- failures, retries, and any change made after seeing intermediate evidence;
- verdict: `mechanical pass`, `exploratory signal`, `admitted`, or `rejected`;
- the next action and whether the paid instance can be decommissioned.

Secrets never belong in this file. Large checkpoints stay on Hugging Face Hub;
compact code, manifests, reports, and hashes stay in Git. The Mac is not a model
artifact store.

## Research question and scope

The project began from a broad question: can a group of weak or small language
models exhibit useful decentralized coordination, rather than merely producing
eight independent responses? Early ideas included hide-and-seek, social games,
Catan-like resource allocation, Hanabi, and CTF. The selected task was an
abstract 4v4 network-protection game because it retained the useful structure of
CTF while avoiding real networks, shells, exploits, Docker sandboxes, and slow
SWE-style infrastructure.

The intended research claim was narrowed to something falsifiable:

> Can four small, separately optimized language-model agents learn to use
> private, grounded messages to improve team control in a deterministic,
> partially observed, simultaneous-action game?

The project does **not** claim to demonstrate broad emergent swarm intelligence,
real cyber capability, or general alignment. Higher return alone is capability
learning. Communication requires causal message interventions.

The main constraints were:

- useful enough to suggest a scalable distributed-defense/resource-allocation
  analogue;
- discrete and deterministic so reward and replay are exactly verifiable;
- cheap enough for 1.7B-4B LoRA experiments;
- 4v4, with eight independent requests and private contexts;
- model-controlled opponents rather than an exploitable deterministic opponent;
- terminal reward rather than a large collection of hackable shaping terms;
- no learned judge and no semantic reward model;
- a frozen final evaluation that cannot be tuned on repeatedly.

## Phase 1 — Environment design and one-turn benchmark

### 2026-08-11: initial implementation

Key commit: `350ba167` (`add Swarm Arena 4B experiment`).

The first environment, `arena-core-v1`, is a deterministic graph-control game:

- two teams, BLUE and RED, with four agents each;
- nodes have ownership, value, a critical flag, fortification, exposure, and
  compromise state;
- node identifiers are randomized and carry no semantic ownership hint;
- each agent has a position, resource balance, private timestamped knowledge,
  known adjacency, and unknown neighboring state;
- agents communicate once, then all eight select actions simultaneously;
- legal actions are `SCAN`, `PROBE`, `CAPTURE`, `FORTIFY`, `RECOVER`,
  `TRANSFER`, and `WAIT`;
- resolution order is fixed: validate, transfer, recover/fortify, scan/probe,
  capture, observation refresh;
- opposing captures are contested and dictionary/worker order cannot decide the
  result.

The structured broadcast contains up to a bounded number of exact node facts,
one legal intent, and an optional one-resource request. Facts must be present in
the sender's observation with the exact timestamp and state. Unsupported facts,
future timestamps, illegal intents, extra keys, prose, and invalid action IDs
fail strict parsing.

An exact centralized joint-action solver enumerates all legal four-agent action
assignments against the fixed opponent action set. It reports the best reward,
all equivalent optima, oracle regret, and leave-one-agent-out redundancy. This
avoids false negatives when several joint actions are equally optimal and avoids
calling two actions redundant merely because they share a target.

The frozen one-turn evaluation contained 60 cases: 20 each at graph sizes 12,
13, and 14, balanced across aggressive, balanced, and defensive opponents. Each
case was evaluated with generated, dropped, reference, and sender-shuffled
messages plus action-order permutations. Manifest SHA-256:
`b53bfc523043ec71cc69f851d0819511c5a9f0b4f09520898f30954bbe874b29`.

Model-free baselines established headroom:

| Policy | Mean reward | Mean oracle regret | Optimal rate |
|---|---:|---:|---:|
| Centralized oracle | 2.645 | 0.000 | 1.000 |
| Independent local | -0.509 | 3.154 | 0.000 |
| Sequential heuristic | -0.421 | 3.066 | 0.000 |
| Random legal | -1.308 | 3.953 | 0.000 |
| Wait | -2.312 | 4.957 | 0.000 |

Verdict: **mechanical pass**. The task had a large centralized-vs-local gap and
exact verification, but the one-turn oracle/reference conditions were saturated
by construction. It was suitable for protocol/SFT checks, not the final RL task.

## Phase 2 — Initial SFT data and 4B stage-one training

### Data-generation decision

We chose solver-generated, machine-verifiable targets instead of teacher-model
rationales. The purpose of SFT was to teach the interface—not the swarm policy.
Targets contained strict JSON only; solver traces and rationales remained
metadata.

The generator:

1. sampled graph topology, node state, resources, partial observations, and
   opponent style;
2. enumerated legal actions;
3. solved the joint action exactly;
4. retained an action label only when it was unambiguous across the complete
   optimum set and prompt-identifiable locally;
5. generated broadcasts only from observed facts and legal intents;
6. re-solved under changed inboxes rather than reusing stale labels;
7. randomized node names, agent identity, action order, and irrelevant state;
8. independently reconstructed and audited every row.

`arena-sft-v2` contained 5,508 examples from 1,024 procedural seeds:

- 4,608 broadcast examples and 900 action examples;
- 4,958 train, 275 validation, and 275 synthetic test examples;
- 512 targeted silence examples and 32 targeted examples per rare mechanic;
- 1,102 ambiguous-optimum, 1,946 non-identifiable, 244 conflicting-intent,
  and 180 targeted-transfer candidates rejected;
- content SHA-256:
  `edad09bb301748621a0fab73ebf3de60d60abfd9f56c9afcc6ca02ffe12f3a80`.

The dataset was published publicly as
`CK0607/swarm-arena-sft-v2`. Split isolation was by procedural seed, and the
frozen evaluation seeds were excluded.

### Model choice

The first training target was `Qwen/Qwen3-4B-Instruct-2507`. A 4B instruct model
was large enough to have useful game competence while still fitting a single
48 GB or 24 GB-class GPU with LoRA. Qwen was selected over less integrated small
models because Prime-RL, the Qwen renderer, structured generation, tokenizer,
and LoRA serving path were available and testable end to end. Full fine-tuning
was rejected on compute and stability grounds.

### Overfit gate and stage-one recipe

An overfit run first checked packing, renderer, adapter loading, and whether the
model could learn the JSON interface. Several implementation fixes were needed:

- Hugging Face dataset schema stabilization (`e04e3f7d`);
- faster frozen evaluation (`56e95f22`);
- LoRA SFT token-packing correction (`cb79e2e7`);
- Prime-RL cat packer adoption (`86657c2c`);
- explicit adapter evaluation (`03e60ffa`);
- Qwen3 renderer alignment in both scoring and inference (`f12127a1`,
  `3809c8cf`, `24d362d7`).

The full stage-one run used:

- rank-32 LoRA, alpha 64, dropout 0.05;
- attention and MLP target modules through the default Prime configuration;
- 66.1M trainable parameters;
- learning rate `1e-4`, cosine decay, 10 warm-up steps;
- 310 packed steps, batch 32, sequence length 2,048;
- thinking disabled and separate adapter checkpoints every 40 steps.

Validation-only selection chose step 240. On the held-out 275-row SFT test:

- overall exact: 98.55%;
- schema valid, supported, and legal: 100%;
- action exact: 97.83%;
- broadcast exact: 98.69%.

On the frozen arena evaluation, SFT improved mean reward over base by `+1.005`
with 95% interval `[+0.370, +1.640]` and reduced oracle regret. Strict message
rate rose from 51.25% to 100%. However:

- generated-minus-dropped reward was `+0.386`, 95% interval
  `[-0.006, +0.777]`;
- generated-minus-shuffled was `+0.076`, interval `[-0.207, +0.359]`;
- therefore the coordination/communication claim gate failed.

Verdict: **protocol and task-capability success; communication claim rejected**.

## Phase 3 — Regression failure and diagnosis

### 2026-08-11: first major scientific failure

The 256-case non-arena regression suite tested arithmetic, instruction binding,
list transforms, and resistance to irrelevant arena-like triggers. The selected
step-240 adapter regressed overall exactness from 48.44% to 37.11% (`-11.33`
points). Instruction binding fell from 64.06% to 28.13% (`-35.94` points).
All eight stage-one checkpoints failed the fixed promotion gate; even step 40
lost 6.25 overall points and 17.19 instruction-binding points.

The diagnosis was not corrupt labels. It was excessive narrow SFT pressure:

- 4,608/5,508 examples were broadcasts;
- 464 identical empty-broadcast targets appeared in training;
- rank 32 exposed 66.1M trainable parameters;
- `1e-4` for 310 packed steps represented roughly 20M training tokens;
- no general-behavior replay was included;
- wrong answers remained well-formed JSON but bound values to incorrect keys,
  consistent with schema interference.

This changed the objective from “maximize arena exactness” to “find the minimum
interface dose that is safe enough to enter RL.” The original adapter was never
promoted as an RL initializer.

Verdict: **rejected**. Success on narrow SFT metrics was a false positive for RL
readiness.

## Phase 4 — Replay-protected warm-start iterations

### Warm start v3

`arena-warmstart-v3` mixed:

- 640 arena protocol examples;
- 640 exact general instruction-preservation examples;
- 1,280 deterministic base-model replay examples from filtered UltraChat;
- 2,560 train and 192 validation rows;
- maximum observed token length 925 under a 1,536-token limit;
- content SHA-256:
  `4f5a5f964b1c2843cd718bc9d8bd010b8010a727186c8ed99daf633fca9211fe`.

Training was deliberately weak: rank-8 Q/V-only LoRA, alpha 16, `1e-5`, 32
steps, checkpoints every eight steps. Every checkpoint passed both paired
regression suites. At step 32, regression v1 improved 0.39 points and v2 lost
only 0.39 points. But protocol learning underfit: 92.71% schema validity,
56.25% grounded broadcasts, and 93.75% legal actions. The selector retained the
base model.

Verdict: **regression-safe, protocol-ineligible**.

### Warm starts v4-v7

V4 increased only the learning dose to `5e-5` over 128 steps while retaining
the replay mix and narrow rank-8 adapter. Inspection exposed a label-design
failure: a hidden seed-index rule selected among empty, fact-only,
fact-plus-intent, and full broadcasts. Identical prompt-level tasks could have
conflicting targets.

V5 replaced those labels with one canonical, prompt-identifiable local report:
observed facts up to budget, one legal local intent, and a visible resource
request. Communication sparsity was deliberately left to RL. V6/V7 further
simplified intent formatting, required concise single-object output, filtered
base replay for complete responses, and clarified that an unknown neighbor ID
is not an observed node fact.

All datasets and validation sweeps were preserved rather than overwritten.
This series established two durable lessons:

1. replay protects general behavior but cannot repair ambiguous task labels;
2. turn-zero, inbox-free validation can look good while failing in a real
   sequential episode.

## Phase 5 — Long-horizon RL-native environment and evaluation

Commit `19da1b0c` introduced `arena-episode-v2`, the actual multi-turn research
task. Episodes run for 4-8 turns with private knowledge, stale observations,
message budgets, local events, resource transfer, opponent switches, and
terminal evaluation. Interventions include normal, dropped, sender-shuffled,
one-turn-delayed, and zero-budget communication.

The model-free 72-case audit showed:

- always wait: mean terminal return `-15.505`;
- independent local agents: `-1.399`;
- centralized collision-avoiding heuristic: `-0.860`;
- centralized-minus-independent: `+0.540`, 95% interval
  `[+0.059, +1.020]`.

The task was neither trivial nor saturated, but the coordination headroom was
modest. Learned communication interventions—not this model-free gap—remained
the claim gate.

A resumable, manifest-bound, side-swapped cross-play evaluator was built. It
records exact model/adapter identities, prompt and option permutations,
interventions, trajectories, throughput, and hashes. Opponents are models in
the headline design; deterministic opponents are not used as the learned
adversary because they invite exploitation.

The final evaluation design freezes:

- ordinary OOD capability games;
- certified communication-critical and matched-decoy cases;
- base, SFT, and historical opponent pools;
- both side assignments;
- normal/dropped/shuffled/delayed/zero-budget interventions;
- adapter permutations, role-label permutations, and action-option
  permutations;
- an action-only RL control;
- seed-level paired bootstrap with the seed—not agent or game row—as the
  independent unit.

The complete confirmatory matrix is 4,320 headline games plus a fixed 12-case
mechanism subset. It has not been run on an RL-selected checkpoint.

## Phase 6 — Sequential SFT v8 and live model diagnostics

### Failure that motivated v8

The first six-match 1.7B engineering pilot showed that turn-zero SFT validation
was not representative:

- base broadcast raw validity: 0/112 because of Markdown wrapping;
- selected v7 broadcast validity: 42/112;
- only 27/112 accepted non-empty broadcasts;
- errors included unsupported facts, illegal intents, fact-budget violations,
  and depleted message-budget violations;
- SFT and serving prompt bytes were identical, ruling out renderer or HTTP
  mismatch.

V7 covered only BLUE agents at turn zero with no inbox, private history, local
events, or depleted episodic budget. Its selector checked a repaired/grounded
fallback but omitted raw legality, causing a false-positive selection.

### V8 data and training

V8 generated complete sequential contexts from 40 training and eight validation
episodes, both teams, graph sizes 12/13, and horizons 4/6. It included every
broadcast/action phase, exact inboxes, local events, up to three private-history
turns, remaining message budget, and explicit empty targets when nothing useful
could be sent. Labels remained local and prompt-identifiable; no joint optimal
trajectory was supervised.

The pre-training audit rejected the initial 2,048-token sequence length because
the longest untruncated context was 2,385 tokens. The final limit was 2,560.

Both 4B and 1.7B recipes used rank-16 attention-only LoRA over Q/K/V/O, alpha
32, dropout 0.05, `4e-5`, 16 warm-up steps, cosine decay, 320 steps, effective
batch 16, assistant-only loss, and thinking disabled.

### 4B result

No 4B V8 checkpoint passed the strict frozen gates. Step 256 was retained only
as the best diagnostic:

- overall schema: 99.864%;
- broadcast grounded: 97.813%;
- broadcast legal: 95.000%;
- actions legal: 100%;
- frozen selection exactness: 68.438%.

Because it was ineligible, paired regression was not run for promotion. Under
structured decoding, a 24-game diagnostic forced 960/960 broadcasts and 960/960
actions valid, but that measured infrastructure enforcement, not learned
competence. Throughput was 136.0 completion tokens/s. Generated-minus-dropped
side-averaged return was `-0.519`, 95% interval `[-3.088, +2.881]`.

Verdict: **useful gameplay diagnostic; non-RL-ready**.

### 1.7B result

Qwen3-1.7B was selected for the first RL systems work because one shared
backbone plus multiple LoRA policies could be served cheaply and quickly. The
published step-320 adapter is intentionally named noneligible:

`CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible`

Pinned public revision:
`534522a8f3ff3489b1dd8318dc8e533e51264cde`; adapter SHA-256:
`2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.

The unconstrained 24-game run had only 475/960 valid broadcasts (49.5%). This
was the meaning of the earlier “broadcast validity 49.5%” caveat: the model
often failed the raw interface, so ordinary RL would waste samples or learn
around parser penalties. Dynamic structured decoding raised both broadcast and
action validity to 960/960, but must be reported as an infrastructure invariant.

The constrained 1.7B run achieved 198.8 completion tokens/s versus 455.2
unconstrained. Generated-minus-dropped was `+0.975`, interval
`[-0.0125, +1.6125]`, with a small sample/randomization `p=0.25`; this was
exploratory rather than strategic evidence. Policies emitted non-empty messages
about 40% of the time and spent near the full budget.

Verdict: **sufficient as a constrained systems warm start; not independently
promoted as learned protocol competence**.

## Phase 7 — RL v3 objective, curriculum, and final evaluation

### Reward design

RL v3 removed additive shaping. The only environment reward is normalized
terminal control-margin delta:

`(BLUE weighted control - RED weighted control) / total node weight`

minus the seed-specific initial margin. Node weight is value plus one for a
critical node. The reward is zero-sum. There are no scan, capture, defense,
collision, speaking, silence, invalid-output, or judge bonuses. Invalid protocol
output is an infrastructure failure, not a reward the policy can trade off.

### Four separate policies

The main experiment uses four distinct LoRA policies over one frozen backbone,
one stable BLUE identity per policy, with four independent optimizer states and
checkpoint slots. RED is model-controlled and frozen for an update epoch. A
shared backbone does not make this single-agent RL: ownership, contexts, token
spans, optimizer states, and policy IDs remain separate.

### Communication curriculum

Stage one uses exact critical/decoy pairs. In each critical case:

- one sender privately knows an exposed, high-value neutral target;
- one receiver can act on it but does not know its state;
- the sender cannot capture it;
- the receiver cannot legally capture without the fact;
- the message makes same-turn capture possible;
- exact joint-action enumeration certifies positive advantage.

The matched decoy changes only the receiver's prior knowledge of that target, so
the oracle communication advantage is exactly zero. All 12 ordered sender to
receiver roles are balanced.

Frozen manifests:

| Split | Pair count | Sizes | SHA-256 |
|---|---:|---|---|
| Train | 240 | 12, 13 | `86634a4da47010427add06ae9e314962d38f17ed7404f9fed80a1c64f72cc766` |
| Development | 48 | 12, 13 | `265a978a4e281b7c9bfe8e3cfe68031e2b0e5d7ae2f5172b1226149586fb8efc` |
| Frozen OOD | 72 | 14, 16 | `7c876814a2593696bea16d3cea754118a133c06b9bb93deec171960de2916b6e` |

The global audit verifies positive critical advantage, zero decoy advantage,
matched sender observations and structural state, receiver-only knowledge
change, role balance, hashes, and zero seed overlap with existing evaluation.

## Phase 8 — Prime-RL integration and safety boundary

Commits `6cfefc26`, `c539b0f3`, `97207a3a`, and `ac7a9446` built the four-policy
Prime-RL path.

Every decision records game, branch, agent, team, policy, immutable revision,
turn, phase, trajectory index, prompt/completion IDs, rollout log probabilities,
dynamic allowed-token sets, sampling key, private-context hash, and decoded
output hash.

The fail-closed CPU supervisor:

- binds source commit, manifests, base, adapters, opponent, constraints, and
  trainer parity gate in a run lock;
- independently reconstructs each private observation and inbox;
- replays every transition and recomputes terminal return;
- verifies exact policy and token ownership;
- rejects stale revisions, partial branch groups, non-finite values, unknown
  constraints, reward mismatch, or context leakage;
- signs approved envelopes with HMAC-SHA256;
- appends decisions to a hash-chained audit log.

The router verifies the signature and one-to-one policy-to-`run_*` mapping, then
creates four isolated Prime-RL batches. Rollout workers cannot write directly to
trainer queues and never receive the supervisor signing key.

Safety/containment is deliberately narrow and verifiable:

- agents can output only enumerated action IDs or bounded structured messages;
- no agent has shell, network, filesystem, arbitrary tools, code execution, or
  external-system access;
- collapse monitors track always/never speaking, repeated targets, action
  concentration, KL, single-opponent overfitting, regression, and return gains
  without intervention gains;
- diagnostics pause promotion and never become shaping rewards.

## Phase 9 — Pre-RL numerical certification

On one RTX 6000 Ada 48 GB, the original constrained four-policy path passed
pre-RL certification at commit `578a98a7`:

- 32 broadcast/action prompts;
- 1,331 completion tokens and 149 branching tokens;
- mean absolute rollout/trainer log-prob error `0.002790`;
- p99 error `0.086815` under the `0.12` gate;
- max probability error `0.073212` under `0.10`;
- p99 probability error `0.029869` under `0.05`;
- probability-error tail fraction `0.001503` under `0.005`;
- mean mismatch KL `0.000144` under `0.0005`;
- max mismatch KL `0.049982` under `0.08`;
- four LoRA optimizer parameter sets were disjoint;
- a real test update changed only `run_blue_0`.

This certified serving/training compatibility and isolation. It did not certify
throughput, strategy, communication credit, or the subsequently changed
message-only token set. Parity must be rerun for the new estimator before RL.

## Phase 10 — Live infrastructure failures and fixes

The live rollout work exposed several failures that consumed real GPU time and
are now permanent launch checks:

1. **Private artifact preflight failure.** Paid compute was provisioned before
   all source/model artifacts were anonymously accessible. Going forward, the
   run preparer verifies exact public base, adapter, and source revisions without
   credentials before creating a paid run directory. All research artifacts are
   public unless explicitly decided otherwise.
2. **Wrong curriculum path.** The first controller draft reconstructed ordinary
   random states instead of the certified curriculum. The controller now
   reconstructs the committed manifest scenario and verifies its canonical
   state hash.
3. **Tuple/list hash mismatch.** JSON reconstruction changed Python tuples to
   lists and produced false state mismatches. State identity now uses canonical
   JSON hashing.
4. **Stale same-name LoRA registration.** vLLM could acknowledge a second load
   while retaining the old path. Adapter refresh is now unload, load, then exact
   `/v1/models` registry-path verification on every server.
5. **Serving/trainer revision ambiguity.** Initial trainable policy, opponent,
   and comparison revisions are bound separately.
6. **Canonical-token trie mismatch.** A choice string can have multiple valid
   tokenizations. A trie made from one `tokenizer.encode(choice)` path rejected
   valid vLLM samples. The rollout client now reconstructs xgrammar's exact
   allowed-token bitmask for the installed vocabulary at every sampled token.
   Protocol version became
   `arena-structured-protocol-v2-xgrammar-choice-mask`.
7. **Multi-vLLM cache collision.** Multiple processes sharing compilation and
   Triton caches could race on missing shared objects. Each server now receives
   unique vLLM, Triton, TorchInductor, RPC, and API-port namespaces.
8. **Process lifecycle ambiguity.** `tmux` panes owned by a `tee` pipeline could
   close even when the server was healthy. The model server is now the pane's
   `exec` process and logging uses `tmux pipe-pane`.
9. **Mac storage risk.** Checkpoints and model caches are not copied to the Mac.
   Only compact reports, hashes, code, and manifests are retained locally/Git;
   selected model artifacts go to public Hugging Face repositories.

These were infrastructure failures, not model findings. They are documented
because omitting them would make the final cost and reproducibility story false.

## Phase 11 — Rejected whole-policy credit estimator

### Initial idea

The first credit estimator replaced one BLUE agent at a time with a frozen base
or SFT comparator and assigned:

`actual terminal return - replacement-branch terminal return`

to that agent's actual tokens. This preserved terminal-only reward but changed
the agent's messages **and** actions, so it was only valid if effects localized.

### 24-case base-counterfactual audit

Twelve critical and 12 matched-decoy scenarios ran with no optimizer:

- critical nonzero credit: 7/12; decoy: 5/12;
- critical mean absolute agent credit: 0.05045; decoy: 0.02841;
- intended sender nonzero: 5/12; receiver: 3/12;
- off-role credit: 6/12 critical and 3/12 decoy.

This proved optimizer signal existed, but critical/decoy pairs had different
sampling namespaces, so paired differences mixed information and sampling
noise. Verdict: **rejected**.

### Common-randomness paired audit

The audit was repeated with one shared sampling namespace per pair and per-
branch common random numbers:

- critical nonzero-credit cases: 9/12; decoys: 8/12;
- mean absolute credit: 0.05350 critical vs 0.04975 decoy;
- intended-sender paired nonzero: 3/12;
- intended-receiver paired nonzero: 4/12;
- off-role paired nonzero: 12/12;
- paired signs: 15 positive, seven negative, 26 zero.

Common randomness removed one noise source but did not fix the causal problem.
Changing an entire policy caused broad downstream behavioral changes that
dominated the private-information intervention.

Verdict: **scientifically rejected; no optimizer started**. Environment,
serving, replay, and terminal reward remained mechanically usable.

## Phase 12 — Local message-edge credit estimator

Commit `567bc139` replaced the default whole-policy comparator with a narrower
causal intervention:

1. actual and counterfactual branches use the same eight policies, revisions,
   initial state, dynamic constraints, and per-decision sampling keys;
2. one counterfactual branch per BLUE sender changes only delivery of that
   sender's first-turn broadcast to the empty message;
3. every other message is delivered normally;
4. downstream actions are regenerated from the altered private inbox;
5. sender advantage is actual terminal return minus message-dropped terminal
   return;
6. only that sender's corresponding actual first-turn BROADCAST span enters its
   optimizer batch;
7. action, later-message, opponent, and counterfactual tokens receive no
   gradient.

The supervisor additionally proves:

- intervention and actual branches did not change policy routing;
- emitted first-turn broadcasts match before delivery is changed;
- only the named sender/turn delivery edge changed;
- identical policy, context, constraint, and sampling key cannot produce a
  different output;
- dropping an already-empty message reproduces the exact trajectory and zero
  advantage;
- policy-replacement evidence cannot be admitted under a message-drop run lock
  and vice versa.

CPU result at implementation time: 35 dependency-light regressions passed,
including explicit tamper rejection; Ruff and compile checks passed. Commit was
pushed and anonymously verified public.

This is **not yet RL-admitted**. It still requires a Linux/live inference smoke,
new broadcast-only rollout/trainer parity certification, and a critical/decoy
localization audit.

The decision rule was frozen before live results in
`MESSAGE_CREDIT_AUDIT_PLAN.md`: a two-pair mechanical smoke, a role-balanced
12-pair diagnostic, and—only if that is promising—a fresh 52-pair/104-scenario
confirmation. Thresholds cannot be relaxed after inspection.

## Phase 13 — Active GPU validation

Started: 2026-08-14.  
Host: paid remote instance supplied by the user; host address is intentionally
not treated as a durable artifact.  
Planned sequence:

1. inspect GPU, storage, processes, repository, and public artifact reachability;
2. checkout exact public commit `567bc139...`;
3. run complete Linux tests and import/runtime smoke;
4. run two pairs (two critical, two matched decoys), rollout-only;
5. if mechanically clean, run the fixed 24-scenario audit;
6. publish compact hash-chained results;
7. decommission immediately after artifacts are safely public.

No optimizer is authorized in this phase. The result and cost will be appended
below when complete.

Early host checkpoint:

- one NVIDIA RTX A6000, 49,140 MiB reported VRAM, idle at inspection;
- 209 GB free of 251 GB local disk;
- no stale training, rollout, or inference process;
- exact public checkout `567bc139...`;
- public submodules initialized anonymously over HTTPS;
- host initially lacked `uv`; resolved with an isolated
  `/workspace/uv-runtime/uv` 0.12.4 installation;
- submodule metadata initially referenced SSH remotes; local checkout URLs were
  changed to public HTTPS without changing committed source;
- environment synchronization and Linux/runtime tests in progress;
- no optimizer or result-producing rollout started at this checkpoint.

Linux test checkpoint:

- the first isolated-project invocation failed during collection because the
  Swarm Arena project metadata did not include runtime imports `httpx` and
  `huggingface_hub`;
- running against the synchronized root environment then exposed that the
  scripts-namespace test also requires `experiments/swarm_arena` on
  `PYTHONPATH`;
- these were invocation/packaging defects, not estimator failures, and no GPU
  workload or optimizer was started while they were unresolved;
- with the correct root overlay and Python path, the complete Linux Swarm suite
  passed: **61 passed**, with two third-party `SwigPy` deprecation warnings, in
  40.39 seconds with the GPU disabled;
- the exact corrected invocation, run from the repository root, was
  `CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/experiments/swarm_arena" uv run
  --with ./experiments/swarm_arena --with pytest pytest
  experiments/swarm_arena/tests -q`;
- `README.md` and `GPU_HANDOFF.md` were corrected so a clean checkout does not
  repeat this detour.

A local macOS recheck did not enter test collection: the installed Homebrew
`uv` was 0.9.2, older than the repository's required `>=0.11.1`, and the
sandbox also denied that binary's shared cache path. No dependency or machine
configuration was changed to mask this; the Linux result above is the executed
validation of the corrected command.

At 2026-08-14 18:06 IST the A6000 remained idle (1 MiB / 49,140 MiB,
0% utilization, 29 C) with no vLLM, inference, RL, or `torchrun` process. Live
runtime/vLLM validation was authorized only after the corrected suite passed.

Pinned public inference assets were then staged without using private tokens:

- base: `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, 3.8 GB, approximately nine
  seconds to download;
- SFT adapter:
  `CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible`, 25 MB,
  approximately two seconds to download, verified SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`;
- 189 GB host storage remained free after staging.

No model process or optimizer had started at this checkpoint. The next paid-GPU
operation was the live vLLM startup/smoke, followed by Stage A only if serving
and structured generation were clean.

Live serving then passed its admission smoke:

- vLLM weight load: 9.45 seconds;
- `torch.compile`: 62.52 seconds;
- CUDA graph capture: 31 seconds;
- steady reserved memory: 40,602 / 49,140 MiB under the configured 0.82 memory
  fraction;
- adapter registry binding passed;
- constrained generation passed 8/8 samples: four `BROADCAST`, four `ACT`, 335
  completion tokens total;
- the live parser separately accepted one legal 53-token broadcast and one
  legal eight-token action;
- no HTTP, grammar/trie, structured-generation, OOM, or retry error occurred.

### Stage A message-edge smoke — stopped fail-closed

At 2026-08-14 18:16 IST the frozen two-pair audit stopped during group zero,
before producing a scientific result or starting Stage B. The supervisor found
that two decisions with identical policy, private context, dynamic constraint,
and sampling key produced different broadcasts:

`message-credit-stage-a-567bc139:step-0:group-0:curriculum:3000003:drop-message-blue-2:blue-1:1:BROADCAST`

This violates the common-randomness invariant. Any terminal-return difference
from those branches would mix the delivery-edge intervention with sampling
noise and therefore cannot be used as message credit. The controller correctly
exited; only the 32-byte supervisor key existed and no Stage A result artifact,
Stage B run, or optimizer was started. The API remained healthy at 40,602 MiB,
33 C, with no serving or parsing error.

Initial status: **mechanically rejected pending diagnosis**. The required
diagnosis was
to preserve and compare exact duplicate request payloads, response hashes,
request order/concurrency, seeds, grammar state, and vLLM logs; reproduce the
smallest failing duplicate request; and establish the cause before editing the
implementation. Thresholds will not be changed around this failure.

The minimized reproduction then rejected the initial concurrency-only
hypothesis. Among 16 duplicate decisions, concurrent replay happened to have
zero mismatches but serial replay had two. The mismatching pairs were identical
in policy/revision/model, sampling key and derived seed, prompt hash and length,
private-context hash, phase-level constraint hash, and sampling parameters:

- `blue-3`: prompt hash prefix/suffix `c7d1...a9dd`, seed `3388348335`, output
  hashes `dceb5a...f7f1` and `788dbb...4920`;
- `red-0`: prompt hash prefix/suffix `97ffff...5f61c`, seed `3858053319`, output
  hashes `a3f344...4598` and `1c3abb...4920`; the decoded choices changed
  `request_resource` from 1 to 0.

Therefore the live vLLM serving path's request seed is not a sufficient
bit-reproducibility guarantee, even serially. One attempted shell command for
summarizing the reproduction had a quoting error; it produced no result and was
replaced by direct extraction of the recorded pairs.

Code review also found that the existing `constraint_sha256` represented only
the protocol phase/version, while the ordered legal choices were dynamic. The
candidate correction does not weaken the supervisor. It:

1. hashes the complete inference identity: selected URL, immutable policy and
   revision, sampling key, exact rendered token IDs, ordered legal choices, and
   normalized sampling parameters;
2. coalesces concurrent identical requests and memoizes completed results only
   within one atomic credit group;
3. records the request hash on every decision;
4. rejects either one common-random semantic key mapping to different request
   hashes or one exact request hash mapping to different outputs;
5. clears the cache at the group boundary so distinct games cannot reuse a
   completion.

This candidate passed Python bytecode compilation locally. Local pytest could
not start because the macOS subproject environment had no pytest executable;
the Linux worker owns the focused, complete, and live validation before any
new scientific run.

The first remote focused-test command named a nonexistent
`test_live_request_identity.py`, so no test ran. The corrected focused run then
passed the supervisor request-hash tamper case but stopped in the new cache test
before exercising the cache: its `EpisodeConfig` accidentally retained the
default nonzero protocol costs, which RL v3 correctly rejects because it uses
hard constraints rather than additive communication/protocol penalties. The
test fixture was corrected to set all three costs to zero. This was a test-only
failure; no inference process or scientific run had started.

The next focused rerun exposed a second fixture-only setup error before the
cache was exercised: `horizon=1`, while RL v3 requires a horizon of at least
two. The fixture was corrected to `horizon=2`; the request-hash tamper test
continued to pass, and no static, full-suite, live-inference, or scientific run
had started at this point.

Validation of commit `bd40ea931c8aab9f5e62288bda4e7ccf44b860fb` then
passed:

- focused request-coalescing and request-hash tamper tests: 2/2 in 20.45
  seconds;
- Ruff: all checks passed;
- compileall: status zero;
- complete Linux Swarm suite: 62 passed in 40.58 seconds, with only the two
  existing third-party SWIG deprecation warnings;
- live exact-request check: two concurrent identical requests caused exactly
  one server POST and returned request hash `3bfe17...d033` and completion hash
  `7e50d1...6cc3`; the same request in a fresh group caused a second POST,
  proving cache reset.

The first live adapter reload after restart returned HTTP 400 because its shell
JSON was escaped incorrectly. The registry proved no adapter had loaded. The
corrected request bound `smoke-adapter` to the frozen warm start, after which
the live check above passed. Expected LoRA-tokenizer deprecation and first-shape
Triton JIT warnings were the only live warnings.

Fresh Stage A ran from 13:21:45 to 13:22:43 UTC, approximately 58 seconds. Both
the critical and decoy groups passed signed replay, delivery, common-random
request, and token-ownership checks. Each approval owned exactly four actual
turn-one `BROADCAST` spans—one per BLUE policy—and zero action spans. There
were no errors or retries. Evidence hashes were `178d06...b2b6` (critical) and
`6a031d...3f39` (decoy); run-lock hash was `69c485...eff5`. This one smoke pair
was not localization evidence: BLUE-2 had paired effect `D=-0.142857` and the
other three effects were zero.

Stage B was stopped before launch because the runner persisted approvals,
returns, credits, and scenario metadata but discarded raw branch messages,
deliveries, actions, and transitions. Running it in that form could not answer
the frozen target-fact, receiver-action/capture, and transition-evidence gates
without repeating all 24 scenarios. The GPU was stopped and remained at 1 MiB.
The remote worker mapped an evidence schema but made no source change and then
stopped responding; direct host inspection confirmed a clean worktree, no live
process, and zero GPU utilization.

A local candidate evidence writer now emits one compact hash-chained record per
message-credit scenario. It includes scenario roles/target, initial-state and
approval hashes, emitted and delivered broadcasts, replay-derived legal action
sets, chosen actions, events, target before/after state and capture events,
returns/credits, receiver effects, and per-decision request/output/context
hashes. It must pass Linux, replay, and two-scenario live validation before
Stage B starts.

The first public paired-summary commit `ba518f39` failed its focused test before
reading Stage B evidence because `message_credit_audit.py` called
`canonical_sha256` without importing it. No summary or scientific verdict was
produced; the immutable Stage B evidence was unaffected. The missing import was
added and must pass focused validation before analysis.

### 2026-08-14 — compact evidence-writer preflight

- Status: completed
- Verdict: mechanical pass; Stage B evidence capture admitted
- Source commit: `05290c7feca5679a2baf735b61b29329020ab06a`
- Base / adapter / opponent revisions: Qwen3-1.7B
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; frozen adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
  for all initial BLUE policies and the frozen opponent.
- CPU checks: focused evidence-writer/message-credit test passed (1/1);
  Ruff and compileall passed; full suite passed (62 tests, 42.81 s). The two
  warnings are pre-existing third-party SWIG deprecations.
- Live preflight: A6000 local vLLM was restarted and healthy (about
  41.0/49.1 GiB VRAM, 37 C). A fresh rollout-only critical/decoy smoke ran
  14:28:16--14:29:15 UTC (about 59 s), with no optimizer step, OOM, HTTP
  retry, structured-generation error, or supervisor failure.
- Evidence result: `message_credit_evidence.jsonl` is 199,886 bytes and its
  hash chain verifies exactly two records. Each record has five branches
  (actual plus four message drops), 80/80 nonempty request hashes, raw emitted
  and delivered broadcasts, replay-derived legal action sets, chosen actions,
  events, target before/after state and capture events, role labels, target
  fact presence, returns, credits, and signed approval/evidence hashes.
- Invocation-only detours: an initial nonexistent focused-test selector, two
  invalid test-fixture settings (nonzero additive costs and horizon one), one
  malformed adapter-load JSON payload (HTTP 400), and an early verifier that
  looked for old field names were corrected. None was a model, inference, or
  evidence-writer failure; the final corrected checks above passed.
- Artifacts (server only; no local download):
  `/workspace/runs/swarm-message-credit-stage-a-evidence-05290c7f/audit/message_credit_evidence.jsonl`,
  `admission.jsonl`, and `live_rl_diagnostic.json`.
- Next action: fixed 12-pair / 24-scenario rollout-only Stage B diagnostic;
  no optimizer unless all frozen localization gates pass.

### 2026-08-14 — Stage B message-credit diagnostic

- Status: completed
- Verdict: rejected for RL; no Stage C or optimizer
- Rollout source commit:
  `ffad14954f4f7b8e695bc394a79ae3f4f5b39ffb`
- Analyzer source commit:
  `19fd2c68a946c440b2b168ac44ecea5150ca1e92`
- Fixed audit: 12 role-balanced pairs / 24 critical-decoy scenarios,
  message-drop, horizon two, rollout-only.
- Timing and GPU: 14:36:47--14:38:57 UTC (about 130 s); A6000 reservation
  41.0/49.1 GiB, observed full-load temperature 51 C. vLLM was stopped after
  the run (1 MiB, 0%, 32 C).
- Mechanical result: 24 hash-chained records, all five branches per scenario,
  with raw messages, delivery, legal/chosen actions, events, target
  transitions, credits, and request/output hashes. No OOM, HTTP retry, or
  supervisor/invariant failure.
- Frozen gates: sender messages identical 12/12 (pass); target fact present
  5/12 (fail); intended mean D +0.0342928735 but 4 positive / 1 negative /
  7 zero (fail); intended/off-role absolute-effect ratio 1.41817 (fail,
  threshold 2); 5/12 pairs with nonzero off-role effects (fail, maximum 4);
  receiver target effects critical 7/12 and decoy 7/12 (fail).
- Interpretation: positive mean alone does not localize credit. The
  message-drop estimator is rejected for RL under the frozen gates.
- Invocation-only detours: the analyzer evidence argument is positional, not
  `--input`; the first analyzer revision lacked a `canonical_sha256`
  import. The fixed analyzer passed its focused test and produced the final
  16,692-byte summary.
- Compact public artifacts:
  `results/pre_rl_1_7b/message_credit_stage_b_ffad1495/`; no keys, tokens,
  model files, caches, or verbose server logs were included.
- Final host state: vLLM and audit sessions stopped; A6000 at 1 MiB, 0%, 29 C;
  189 GB free of 251 GB. The instance is decommission-ready but termination is
  a user-controlled external action.
- Estimated total host cost for this resumed validation window: approximately
  USD 1.70--1.90 at the user-reported USD 0.75/hour. This is a range because the
  provider's exact boot/billing timestamp was not available in the run
  artifacts; measured experiment timings remain exact above.

## Current decision gates

The Linux/live-serving, replay, delivery-edge, request-identity, and
broadcast-only ownership gates pass mechanically. The scientific localization
gate fails: target facts are too rare, effects are insufficiently concentrated
on intended senders, off-role effects are too common, and receiver changes are
no more critical-specific than decoy-specific. Therefore:

- no Stage C confirmation, rollout/trainer parity recertification, or RL
  optimizer run is authorized for this estimator;
- thresholds must not be relaxed and no additive communication reward should
  be introduced to force a pass;
- the next work is CPU-only evidence diagnosis: compare the five target-fact
  pairs with the seven omissions, inspect why two positive sender effects occur
  without the target fact, and determine whether the failure is primarily
  warm-start capability, curriculum construction, or a genuinely nonlocal
  causal effect;
- any revised task, warm start, or estimator must receive a new frozen audit
  plan and fresh development seeds while leaving the existing frozen OOD
  evaluation unchanged.

### 2026-08-14 — broadcast-priority capability probe

- Status: planned; frozen before results
- Verdict scope: prompt/capability diagnosis only, never RL admission
- Motivation: in the rejected Stage B evidence, every certified target was an
  older turn-zero observation. The model frequently selected unrelated
  turn-one facts instead. The current prompt's broad “prefer newer” sentence
  may be incorrectly making recency compete across different nodes, rather
  than resolving conflicting reports about the same node.
- Inputs: the same 12 already-inspected training-manifest pairs, sender
  broadcast only, five common-random repetitions per pair. No final/OOD cases.
- Variants: current prompt; recency scoped to conflicts about the same node;
  and a generic actionable-priority rule that prefers EXPOSED/COMPROMISED,
  critical, and high-value facts. No variant receives the certified target ID,
  receiver answer, oracle action, or reward.
- Fixed diagnostic gates for a promising variant: 100% protocol validity;
  target fact in at least 45/60 samples; majority emission on at least 9/12
  pairs; and at least +0.20 absolute target-fact rate over the current prompt.
- If a variant passes, confirm it on a fresh seed/pair subset before changing
  the versioned production prompt or rerunning any causal audit. If none passes,
  diagnose or improve the warm start rather than training on the rejected
  credit signal.
- No optimizer is authorized by this probe.
- Preflight failures before sampling: public commit `77f8df1f` accidentally
  placed the focused priority test inside the preceding fail-closed test, so it
  raised `NameError: actual_decisions is not defined`. Commit `e68c826a` moves
  the test to module scope. A subsequent remote checkout failed because the
  orchestrator relayed an incorrect expanded SHA; anonymous `ls-remote` and
  local `rev-parse` agree on the corrected exact commit
  `e68c826aab419d3e93fe5f492faccb356e420211`. Neither detour issued inference
  requests or optimizer steps, and the healthy vLLM process was left intact.
- A third invocation-only detour passed obsolete variant names to the probe's
  CLI; `argparse` rejected the command before sampling. The canonical launch
  omitted `--variant` flags and therefore used the three versioned defaults.
- Completed result: 180 broadcasts in about 139 seconds on one A6000, with no
  inference or protocol errors and zero optimizer steps. `current` achieved
  60/60 valid, 44/60 target facts (73.33%), and majority on 10/12 pairs;
  `recency_scoped` achieved 60/60, 44/60, and 10/12; `actionable_priority`
  achieved 60/60, 57/60 (95%), and 12/12. The actionable variant therefore
  passes all four frozen capability gates, including a +21.67 percentage-point
  improvement over current. This is capability evidence, not RL admission.
- Compact artifact:
  `results/pre_rl_1_7b/broadcast_priority_probe_e68c826a/probe.json`, 222,527
  bytes, SHA-256
  `1f127c43d10358ffcc11bfe78ade5e2999ab5f4d4783b398d67f3e8e82d6e952`.

### 2026-08-14 — fresh broadcast-priority confirmation

- Status: planned; frozen before results
- Verdict scope: capability replication only, never RL admission
- Inputs: untouched training-manifest pairs 12--23, sender broadcast only,
  five repetitions per pair. These pairs were not used to create or choose the
  prompt variants and no final/OOD examples are exposed.
- Variants: `current` and the already-selected `actionable_priority` only.
- Fixed confirmation gates: 100% protocol validity; target fact in at least
  45/60 actionable samples; majority emission on at least 9/12 pairs; and at
  least +0.20 absolute target-fact rate over `current` on this same fresh
  slice. Failure on any gate rejects production-prompt promotion.
- No optimizer is authorized. Passing permits a versioned prompt update and a
  newly frozen causal re-audit on unused development scenarios; it does not by
  itself authorize RL.
- Completed result: 120 broadcasts in about 121 seconds on the A6000, no
  errors and zero optimizer steps. `current` achieved 60/60 valid, 43/60
  target facts (71.67%), and majority on 9/12 pairs. The preselected
  `actionable_priority` achieved 60/60 valid, 58/60 target facts (96.67%), and
  majority on 12/12 pairs. Its +25 percentage-point gain passes the frozen
  confirmation threshold. Misses were isolated to pairs 17 and 22, each 4/5.
- Verdict: capability replication passes. This permits versioning the exact
  generic instruction into the BROADCAST prompt, but does not admit RL.
- Compact artifact:
  `results/pre_rl_1_7b/broadcast_priority_confirm_a1c3740a/probe.json`, 147,767
  bytes, SHA-256
  `c2da6a11a4cd4cd5118881ea9a3e84d817936eacafd3c963d82e3b492f3e41f7`.

### 2026-08-14 — actionable-prompt causal re-audit

- Status: planned; frozen before results
- Verdict scope: message-drop estimator admission, no optimizer
- Prompt: version `arena-episode-v5-actionable-broadcast-priority`, containing
  exactly the independently confirmed generic instruction. The prompt never
  receives the certified target, receiver answer, oracle action, or reward.
- Stage A: development-manifest pairs 0--1, four alternating critical/decoy
  scenarios, mechanical smoke only. All original exact replay, delivery,
  common-random request, constraint, token-ownership, and supervisor gates must
  pass before Stage B.
- Stage B: if Stage A passes, development-manifest pairs 12--23, 24 alternating
  critical/decoy scenarios. This slice is disjoint from Stage A and from both
  train-pair prompt probes. Apply all eight original frozen Stage B conditions
  in `MESSAGE_CREDIT_AUDIT_PLAN.md` unchanged; conditions 3, 5, or 6 failing
  rejects the estimator, and thresholds will not be relaxed after inspection.
- Stage C remains prohibited unless Stage B is promising. If admitted, it will
  start at development pair 24 to remain disjoint. No final/OOD cases are used.
- No optimizer or trainer process is authorized by Stage A or B.
- Preflight: the first focused command named nonexistent
  `tests/test_broadcast_priority.py` and therefore ran zero tests. The corrected
  selectors passed 3/3; Ruff and compileall passed; the full Linux suite passed
  63 tests in 41.46 seconds with only the known SWIG deprecation warnings.
- Stage A completed 15:16:58--15:17:56 UTC (about 58 seconds): four records,
  five replayed branches per record, 80/80 request hashes, exact hash-chain and
  complete-evidence verification. Evidence SHA-256 is
  `d18cdebb2f34e7c4bd4ba741d8f7567f1f3e1bfca75c924ed40c9b49cf501b73`;
  admission SHA-256 is
  `8e6ba6a4122c24157228bd1ed26cc7ee190d66a19412ac2a8a0111138ee23564`.
  Mechanical Stage A passes and authorizes Stage B, not RL.
- The first Stage B launch stopped at 15:20:57 UTC after 18/24 evidence
  records when the inference client raised `httpx.ReadError` while awaiting an
  action completion. It produced no analyzer verdict and no optimizer step.
  The partial directory is immutable failed-run evidence and will not be
  resumed, truncated, combined with another run, or scored.
- Retry policy frozen before diagnosis/result: permit exactly one clean Stage B
  rerun from group zero only if API/process/GPU/disk/log inspection finds no
  OOM or model/grammar failure and a pinned structured-generation smoke passes.
  Use the identical prompt, policy, development pairs 12--23, sampling keys,
  and original gates. Do not add request-level retries. A second transport or
  mechanical failure stops this audit rather than selecting a convenient run.
- Diagnosis passed the retry prerequisites: vLLM stayed alive; disk had 189 GB
  free; no OOM, Xid, NCCL, engine, model, or grammar failure appeared; all
  visible server requests returned 200; and a fresh 83-token constrained smoke
  was legal. The failure was isolated to the client connection while receiving
  headers. The original run ID was retained so sampling namespaces stayed
  unchanged; only the output directory and documentation-only source commit
  changed.
- The single authorized retry completed all 24 records in 104.26 seconds from
  launcher to log close (80.88 seconds from run-directory creation to evidence
  close). All hash chains, supervisor approvals, replay fields, and request
  identities verify; no optimizer ran.
- Final gates: identical sender messages 12/12 pass; target fact 11/12 pass;
  intended mean D +0.0705009 pass; receiver target effects critical 12 versus
  decoy 3 pass; signs 7 positive / 1 negative / 4 zero fail; localization
  1.0767x fail; off-role nonzero 8/12 fail. Verdict: rejected; Stage C and RL
  remain prohibited.
- Interpretation: the prompt repair succeeded and the receiver's immediate
  action is message-sensitive, but a two-turn team return is too coarse for
  reliable per-sender localization. The result argues against more prompt/SFT
  work as the immediate fix; the next issue is the credit estimand itself.
- Compact public results:
  `results/pre_rl_1_7b/actionable_message_credit_stage_a_f54e72b4/`,
  `results/pre_rl_1_7b/actionable_message_credit_stage_b_partial_f54e72b4/`,
  and
  `results/pre_rl_1_7b/actionable_message_credit_stage_b_retry_92cce5f2/`.

### 2026-08-14 — actionable-prompt communication cross-play

- Status: planned; frozen before results
- Verdict scope: collective communication efficacy only, never credit-estimator
  or RL admission
- Focal policy: pinned 1.7B SFT adapter; opponent: its pinned 1.7B base parent.
  Both are model-controlled, sides are swapped, and no deterministic opponent
  is used.
- Inputs: 12 new development cases beginning at seed 5,000,003; horizons 4/6;
  history window 3 for both teams. No training-manifest, prior audit, or frozen
  final/OOD case is used.
- Conditions on the focal team: generated, all messages dropped,
  sender-shuffled, and one-turn delayed; opponent messages remain generated.
  Dynamic protocol constraints remain enabled. With both side assignments this
  is 96 complete games and 12 independent seed-level comparisons per
  intervention.
- Primary diagnostic: side-averaged generated-minus-dropped mean return must be
  positive with at least 8/12 positive seeds to call collective communication
  promising. Generated-minus-shuffled and generated-minus-delayed means, exact
  paired intervals/randomization tests, protocol/grounding validity,
  communication spend, duplicate targets, and throughput are mandatory
  supporting reports. No threshold will be revised after inspection.
- A pass shows this policy's messages help team play against a model opponent;
  it does not repair the rejected per-sender estimator or authorize RL. A
  manifest-matched resume is allowed only for a transport interruption because
  games are independent; no failed row may be retained.
- Completed cleanly: all 96/96 rows and 48/48 side-swapped pairs verified.
  Run source `679e60a57a35abecc989a5a654236f19bbb9182b`; analyzer source
  `274bbe87643ff36f2eb3c239b7d8bfd9dfb7fe0b`. Runtime was
  15:44:39--15:57:32 UTC (about 773 seconds); 7,680 requests and 182,608
  completion tokens ran at 242.02 completion tokens/s. No error, resume, OOM,
  invalid output, optimizer, or RL step occurred.
- Generated-minus-dropped: +0.3167 mean, 95% interval
  [-0.8792,+1.3542], p=.6162, 8/12 positive. This meets the frozen exploratory
  screen exactly but is statistically weak and is not confirmatory.
- Generated-minus-sender-shuffled: -0.0250, interval
  [-1.3625,+1.3000], p=.9712, 6/12 positive. No detectable sender-routing
  benefit.
- Generated-minus-delayed: +1.2042, interval [+0.2208,+2.2833], p=.0474,
  7/12 positive. Timely messages carry measurable value even though the effect
  is heterogeneous across seeds.
- All focal protocol, grounding, and action rates were 1.0 with zero invalid
  outputs. Generated duplicate-target turns were 0.4375 versus 0.2361 dropped,
  0.3507 shuffled, and 0.3611 delayed; communication is not yet reliably
  reducing redundant actions.
- Verdict: marginal collective communication signal, strongest for timeliness;
  no strong sender-specific coordination and no repair of the rejected credit
  estimator. Do not run Stage C or the optimizer. The next implementation
  target is the source- and opponent-matched shared-terminal-return candidate
  in `MULTI_POLICY_RL.md`, followed by parity and collapse gates.
- Public compact artifacts:
  `results/pre_rl_1_7b/actionable_crossplay_96_679e60a5/`. The 54.5 MB raw
  matrix is stored as a deterministic 2.32 MB gzip; both compressed and raw
  hashes are recorded in its `RUN.md`.

### 2026-08-14 — shared-terminal-return CPU admission path

- Status: implementation complete; live parity validation pending
- Verdict scope: systems admission only, no optimizer and no scientific RL
  admission
- Hypothesis: a scenario- and opponent-matched leave-one-out baseline can
  preserve the verified terminal team objective while avoiding the rejected
  per-sender localization estimator and any additive shaped reward.
- Frozen bootstrap contract: `K=4` independent joint trajectories from one
  initial state; advantage
  `A_k = G_k - mean(G_j for j != k)`; no additional normalization; train only
  first-turn `BROADCAST` completion spans; share `A_k` across the four agents in
  trajectory `k` while keeping four policy IDs, contexts, token spans,
  gradients, optimizer states, and checkpoints separate.
- Implemented CPU path: immutable estimator-spec hash in `RunLock`; independent
  game/sampling namespaces per replica; concurrent full-game rollout builder;
  independent replay of every state transition and terminal return; complete
  private-context/output, dynamic-constraint, policy-revision, delivery, and
  namespace validation; exact zero-sum leave-one-out invariant; four signed
  approvals per replica group sharing one complete-evidence hash; four-policy
  Prime-RL batch routing and atomic replica merging; full hash-chained evidence
  persistence. The rejected message-drop and replacement estimators remain
  available only as explicitly selected legacy diagnostics; shared return is
  now the controller default.
- Safety tests added for exact LOO values, replay-return tampering, reused
  namespaces, stale estimator config, signed four-policy routing, selected-span
  ownership, and merged replica batches. Ruff lint passed on all changed files;
  the existing message-credit regression passed 2/2 locally. The local Mac's
  `uv 0.9.2` cannot parse the repository's `uv>=0.11.1` friendly-duration and
  universal-lock syntax, and the deliberately minimal local environment lacks
  the live rollout stack. No large dependencies or checkpoints were downloaded
  to work around that; the focused full integration suite will run on the
  already-warm Linux A6000 environment.
- GPU use so far for this entry: none; the existing vLLM process was left warm
  and untouched.
- No optimizer step is authorized. Next action: run the focused Linux tests,
  then one `--rollout-only` live group to certify serving/replay/routing and the
  deferred trainer parity handoff. Only a clean result permits the separate
  real trainer-parity certificate.
- Linux validation completed on the warm A6000 without restarting vLLM. The
  focused shared-return plus message-credit tests passed 4/4 in 19.98 seconds;
  the complete Swarm Arena suite passed 65/65 in 44.27 seconds. The first test
  command exposed a stale cached wheel and imported older source; the exact
  commit passed after switching the local package to editable mode.
- One bounded rollout-only critical curriculum group then completed with four
  independent replicas, 16 decisions and one terminal replay transition each,
  four signed approvals, exact replay/context/output/routing checks, and both
  hash chains verified. Returns were `[+0.166667, -0.083333, 0, 0]`; LOO
  advantages were `[+0.194444, -0.138889, -0.027778, -0.027778]` and summed to
  zero. The controller constructed all four isolated policy batches and merged
  replicas, but `--rollout-only` prevented queue admission. The first launcher
  omitted the editable-package flag and failed at import before contacting
  vLLM; the corrected frozen command completed cleanly.
- Compact public evidence is in
  `results/pre_rl_1_7b/shared_return_smoke_c267f479/` (453,301 bytes total).
  Complete-evidence SHA-256:
  `84cf512166c23262645c555e73f473df5e0eec69b73b358a7e4d23dbf061b304`;
  admission SHA-256:
  `8c85cf8533de0236934199b11021bf62cb869228956bd4cc5c14ad414497a70b`.
  The supervisor signing key is excluded. GPU remained healthy at 41,032 MiB,
  0% idle utilization, 33 C after the run.
- Mechanical verdict: pass. Scientific/training verdict: still not admitted;
  trainer-side constrained log-prob parity and the predeclared opponent-pool,
  intervention, KL, and collapse gates remain before any optimizer step.
- Post-smoke adversarial review found that the v3 approval committed decision
  IDs and completion-token counts but the router did not cryptographically bind
  the exact in-memory `TrainingSample` content. A malicious worker could have
  substituted same-length token or log-prob arrays after approval. No optimizer
  had run. Supervisor/router/contract v4 now persist allowed-token rows in each
  decision, hash the complete trusted sample projection into each shared-return
  envelope, recompute that hash from the untrusted Prime sample, and reject any
  mismatch before queue admission. A same-length prompt substitution is a
  required fail-closed test. The c267 smoke remains valid rollout/replay
  evidence but is explicitly superseded for queue-admission purposes; a v4
  rollout-only smoke is required before trainer parity.
- The required v4 smoke completed under source
  `ab981247772c66cff5f5b00922fc1fa9c8f1aea0`. The immutable source guard first
  rejected a mistyped full SHA before any model request; the corrected command
  completed four replicas with returns `[-0.083333, 0, +0.083333, 0]` and LOO
  advantages `[-0.111111, ~0, +0.111111, ~0]`. All 64 decisions persisted
  complete allowed-token rows; all 16 envelopes contained a non-empty exact
  sample hash; replay, context, output, signature, sample, routing, and both
  audit-chain checks passed. Focused tests passed 4/4 and the full suite passed
  65/65 in 41.07 seconds. No trainer or optimizer ran.
- Authoritative v4 evidence:
  `results/pre_rl_1_7b/shared_return_smoke_ab981247/` (488,081 bytes).
  Complete-evidence SHA-256:
  `6fae21eba1438282023df10ef808abe26b24ac75364ca9acd940f89c03c0c0d0`;
  admission SHA-256:
  `1c2115b593556ef75cf08b7785d50b00cd123671d7b3da7634c2e49ac93dfd58`.
  The v3 directory is marked superseded for queue admission.

### 2026-08-14 — approved-rollout trainer parity diagnosis

- Status: failed gate; diagnosis in progress; no RL optimizer
- Source: `fb272aed`; probe SHA-256
  `df9a1213e6cd94e5e25a8e7801a6afdb8e15edd37dee6cc46e08d35a745138d1`
- The verified v4 evidence deterministically produced 16 policy-bound
  first-turn broadcast samples (1,002 completion tokens; 857 branching
  tokens). The probe builder verifies the evidence hash chain, selected phase
  and turn, policy slot, sampled-token legality, and exact allowed rows.
- A co-resident certificate attempt failed while materializing the trainer:
  warm vLLM held 40.06 GiB, the trainer held 7.31 GiB, and a final 48 MiB
  allocation exceeded the A6000. This is a capacity failure, not a parity
  result. The failed directory remains immutable.
- With stateless vLLM stopped, the single clean retry completed. Four optimizer
  parameter sets were disjoint and the disposable isolation step changed only
  `run_blue_0`, so policy isolation passed. Aggregate mean absolute log-prob
  error 0.002184, p99 0.051755, p99 probability error 0.028206, tail fraction
  0.003992, mean mismatch-KL 0.0001447, and max mismatch-KL 0.033244 all passed
  frozen limits. Maximum probability error was 0.140238 versus the frozen 0.10
  ceiling, so the certificate correctly failed and no training admission is
  allowed. The threshold will not be relaxed.
- Current hypothesis: a small number of branching tokens, concentrated in
  `blue-0` broadcasts, amplify serving/trainer numerical differences. The next
  diagnostic adds token-level outlier identity and tests the predeclared
  serving-side eager execution variant. Any variant requires fresh rollout
  evidence and the same unchanged parity limits.
- The original compiled vLLM server and all six pinned LoRA registrations were
  restored after the retry.

### 2026-08-14 — strict serving, mask audit, and broad parity rejection

- Status: completed; rejected for RL admission; no promoted optimizer output
- Decision unlocked: the CPU-side multi-policy rollout/admission implementation
  is complete and fail-closed, but the current vLLM/Prime numerical pair cannot
  start RL under the frozen parity contract.
- Source progression: `b74a8f8b` selected native eager serving; `0010a013`
  made sampling filters explicitly neutral; `7dc6a41a` bounded serving at four
  sequences; `8aaeaaa0` added live server-mask verification; `ae857d9d` treated
  only vLLM's `-9999` masked sentinels as non-finite; `5e0e87ed` added bounded,
  auditable transport recovery after a real broad-run connection reset.
- Run-lock v5 binds the complete trainer and serving config hashes in addition
  to source, model, adapter, opponent, data, estimator and constraint identity.
  Each approval binds the exact prompt IDs, completion IDs, rollout log
  probabilities, allowed-token rows, sample masks, temperatures, environment,
  policy slot and revision. The router recomputes the complete sample hash.
- Serving requests explicitly set temperature `1`, top-p `1`, top-k `0`,
  min-p `0`, maximum 128 tokens, seed, top-20 token-ID logprobs, and the exact
  structured choice list. Whenever an allowed set contains at most 20 tokens,
  the finite server top-logprob IDs must exactly equal the independently
  reconstructed xgrammar set. Model/HTTP/timeout/parser/mask failures are never
  retried. Network/protocol retries are capped at three identical seed-bound
  requests and persisted as `transport_attempts`.
- Diagnostics were deliberately progressive. The original compiled 16-sample
  probe failed maximum probability error (`0.140238 > 0.10`). One eager
  16-sample diagnostic passed every gate (`0.078062` max probability error,
  `0.018078` max mismatch-KL), showing that the path could agree narrowly but
  not establishing broad admission. A strict one-sequence server exceeded the
  frozen 180-second client timeout and was rejected as operationally unusable.
- The first four-group mask-audit attempt completed two groups, then received
  `httpx.ReadError` while the still-healthy server drained a large connection
  pool. No model, mask or HTTP error occurred. The client now disables local
  HTTP keep-alive and permits only bounded identical-request retries for
  network/protocol errors. A unit test proves byte-identical request bodies;
  focused Linux tests passed 4/4 before relaunch.
- The fresh authoritative run at `5e0e87ed` completed four groups: two
  communication-critical and two matched decoys over curriculum seeds 4000008
  and 4000009, four independent replicas per group. It produced 256 decisions;
  all 256 passed the exact mask audit and completed on transport attempt one.
  Replay, return, private context, output, revision, sampling namespace, sample
  content, four-policy routing, signatures and both audit chains passed.
- Probe SHA-256:
  `fe0ae52d78c3e85607bd1c74a265a7f7721df917fedb407c1d67d75b28d3162d`;
  64 first-turn broadcast samples, 4,305 completion tokens and 3,672 branching
  tokens. Four optimizer parameter sets were disjoint and the disposable
  isolation update changed only `run_blue_0`.
- Broad parity result: mean absolute log-probability error `0.002640`, p99
  `0.069428`, p99 probability error `0.029992`, probability tail fraction
  `0.003949`, and mean mismatch-KL `0.0001818` all passed. Maximum probability
  error `0.140238` exceeded `0.10`; maximum mismatch-KL `0.100478` exceeded
  `0.08`. The certificate exited nonzero and no trainable checkpoint was
  written or promoted. Thresholds were not relaxed.
- Interpretation: exact live mask equality rules out a hidden legal-token or
  normalization-set mismatch. The remaining failure is rare numerical drift
  between vLLM decode log-probabilities and Prime's teacher-forced forward path,
  concentrated at a few branching tokens. Narrow probes can miss it; a broad
  pre-step gate is necessary.
- Validation: complete Swarm Arena tests passed 69/69 in 43.25 seconds. Ruff
  passed on every changed file. Whole-experiment Ruff still reports 12 older,
  unrelated import/unused-variable findings; they were not mixed into this
  safety change.
- GPU/cost: the decisive strict-serving through trainer-certificate window was
  approximately 19 minutes on the $0.48/hour A6000 (about $0.15). vLLM and all
  trainer/rollout processes were stopped afterward; final GPU state was 1 MiB,
  0% utilization, 32 C.
- Public evidence:
  `results/pre_rl_1_7b/parity_compiled_fb272aed/`,
  `results/pre_rl_1_7b/parity_eager_313c1aa7/`, and
  `results/pre_rl_1_7b/parity_mask_audited_broad4_5e0e87ed/`. The preceding
  transport failure and its two complete groups are preserved in
  `results/pre_rl_1_7b/transport_failure_ae857d9d/`. Supervisor keys and
  checkpoints are excluded; raw evidence/probe hashes are recorded before
  compression.
- Next action: do not spend more single-A6000 time on serving knobs. On the
  eventual multi-GPU topology, make inference and trainer use a genuinely
  matched forward/precision implementation (or generate rollout
  log-probabilities through the trainer-compatible policy service), then rerun
  this unchanged four-group gate. Only a pass can enable the first bounded RL
  optimizer step and its communication/collapse evaluations.

### 2026-08-15 — CPU-only parity recovery implementation

- Status: implementation complete; GPU execution and Linux integration tests
  pending; no RL admission
- Hypothesis: the rare vLLM/Prime mismatch may be isolated to the trainer model
  or attention implementation. Test four predeclared trainer variants against
  the same published broad probe that exposed the failure before undertaking a
  cached Prime-native actor.
- Scientific constraint: behavior log-probabilities are never replaced with
  trainer values. Such relabeling would hide the off-policy distribution and
  make the parity gate meaningless. Prime's custom Qwen3 path currently rejects
  KV-cache generation, so using repeated full trainer forwards as an actor was
  rejected as operationally infeasible.
- Rollout evidence now persists each finite server top-logprob distribution,
  excluding only masked `-9999` sentinels. Decisions validate row count, token
  uniqueness, finiteness, sampled-token presence and membership in the trusted
  allowed set. Fresh parity probes use schema v2; the published v1 evidence
  remains readable without pretending it contains distributions.
- The trainer certificate now accepts and hashes the actual trainer TOML,
  verifies the model and pinned adapter identity, and reports full constrained
  distribution normalization error, maximum probability error, total
  variation, serving-to-trainer KL and trainer-to-serving KL on every complete
  row. The original frozen sampled-token admission gates remain unchanged.
- Added a predeclared four-variant matrix: custom+FA2 baseline, HF+FA2,
  HF+SDPA, and custom+eager. One variant is isolated per visible GPU with a
  unique rendezvous endpoint. The selection rule is checked in as
  `first_passing_variant_in_declared_order`; every command, source/probe/config
  hash, log, return code and report is persisted. No passing variant causes a
  nonzero exit.
- A matrix pass is diagnostic only. The selected trainer config must be bound
  into a fresh shared-return run lock, receive new four-group rollout evidence,
  and pass a fresh certificate before any optimizer pilot.
- Validation: all changed Python files compiled; changed-file Ruff passed; the
  three TOML variants and JSON matrix parsed and matched their declared
  implementation/attention values; the four-command matrix dry run passed and
  recorded unique GPUs/rendezvous endpoints plus stable hashes. The local
  Homebrew `uv 0.9.2` intermittently panics in macOS dynamic-store discovery,
  so the full Prime/xgrammar pytest suite remains a mandatory first action on
  the Linux GPU host before model loading.
- GPU/cost: none.
- Cost ordering: run the four trainer variants immediately on the published
  failing probe, without starting vLLM. Only a passing variant earns a fresh
  schema-v2 rollout and bound recertificate. This avoids holding three idle GPUs
  during an unnecessary initial rollout.
- Operator decision: do not provision an unattended overnight instance for the
  first attempt. Provision 4x L40S when the first 10--15 minutes can be watched;
  the parallel matrix should quickly determine whether to create fresh evidence
  or terminate. Full sequence and stop conditions are in
  `PARITY_RECOVERY_PLAN.md`.

### 2026-08-15 — L40S parity recovery, matched actor, and first optimizer OOM

- Status: retry in progress; first optimizer attempt failed closed; no learned
  checkpoint was written
- Verdict: parity and four-policy isolation admitted the matched actor; the
  16,384-token trainer layout was rejected as operationally infeasible
- GPU: 4x NVIDIA L40S 48 GB at `64.247.196.177`; provider hourly price was not
  supplied, so wall time is recorded without inventing a cost estimate.
- Public inputs: Qwen3-1.7B revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; warm-start revision
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`; adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.
  Anonymous public preflight succeeded. The complete Linux Swarm suite passed
  69/69 twice (22.42 s and 17.05 s).
- The predeclared four-GPU trainer matrix against the published broad probe
  selected custom Qwen3 + FlashAttention 2 by the declared first-pass rule.
  HF + FlashAttention 2 also passed; HF + SDPA failed maximum mismatch-KL
  (`0.101038 > 0.08`); custom + eager was an invalid Prime configuration and
  failed at startup with `KeyError: 'eager'`. These outcomes are defects and
  negative results, not missing arms.
- A fresh strict-vLLM four-group recertificate failed maximum probability error
  (`0.13877 > 0.10`) and maximum mismatch-KL (`0.114236 > 0.08`). Repeating the
  certificate with the HF trainer produced the same outlier, locating the
  mismatch in the serving/training boundary rather than the trainer variant.
- Source `01519aa9a019871d9560971b11b80826ad1cfac1` added a truthful local HF
  constrained actor. It samples from its own masked distribution, records the
  actual behavior log-probabilities plus the complete allowed distribution,
  uses the same BF16-by-BF16-to-FP32 LM-head path, supports five isolated LoRA
  adapters, and hot-reloads the four trainable adapters. Trainer log-probabilities
  are never substituted after sampling.
- A one-group certificate passed. The authoritative fresh four-group probe had
  SHA-256 `6f0ee9c06d53e75a915c7c24a800afefed855a255609b71ecaa91358739b30ef`,
  64 policy samples and 4,346 completion tokens. Against exact trainer config
  SHA-256 `1c0d354583ee7571b96c4bdab350e2dbe19effa162d5ea6d96ca4745d98747bc`,
  mean absolute log-prob error was `0.002261`, p99 `0.079917`, maximum
  probability error `0.099317`, p99 probability error `0.027553`, probability
  tail fraction `0.003682`, mean mismatch-KL `0.0001510`, and maximum
  mismatch-KL `0.074599`; every unchanged gate passed. Four optimizer parameter
  sets were disjoint, and a disposable update changed only `run_blue_0`.
- The first real eight-step run generated and signed four step-zero groups, then
  failed closed before `optimizer.step()`. The unfused vocabulary head attempted
  an additional 7.57 GiB allocation after 36.96 GiB was resident on GPU 0.
  No `STABLE` update or learned checkpoint exists. Evidence, admission records,
  routed batches, and the failed config remain immutable under
  `/workspace/runs/rl-hf-actor-pilot-01519aa9`.
- Root cause: the trainer advertised `seq_len = 16384` even though every
  per-policy orchestrator and admitted Swarm sample is capped at 4,096. The
  vocabulary-logit path therefore reserved four times the supported sequence
  capacity. The correction sets the trainer cap to 4,096; it does not change
  reward, precision, optimizer dtype, reduction dtype, policy ownership, or
  parity thresholds.
- Retry rule: create a new immutable run directory and source commit, regenerate
  four-group evidence bound to the corrected trainer TOML, rerun the complete
  parity/isolation certificate, and only then attempt optimization. The failed
  run is never resumed or overwritten.

### 2026-08-15 — First 4,096-token retry exposed packing-step mismatch

- Status: failed closed; no optimizer update and no learned checkpoint
- Preserved run: `/workspace/runs/rl-hf-actor-seq4096-acb242bb`
- The fresh four-group certificate for resolved trainer SHA-256
  `788218207782687d6a339912d50426b21f1c00cf9564f73359c9e68df25e197f`
  passed on 64 samples and 4,154 completion tokens: mean absolute log-prob
  error `0.0019235`, p99 `0.055065`, maximum probability error `0.071481`,
  maximum mismatch-KL `0.048064`, probability-tail fraction `0.0031295`,
  and four-policy isolation passed.
- The controller then produced four signed shared-return groups. Each policy
  batch contained 16 samples and 13,192–14,206 total prompt-plus-completion
  tokens. Prime's trainer `max_steps` counts 4,096-token packing slices, not
  logical Swarm controller updates. The finite value `8` therefore exhausted
  the trainer after only 32,768 packed tokens, before any complete per-policy
  batch (54,716 tokens total) became ready.
- Every pre-step parity gate passed, but all eight trainer log rows reported
  `No runs are ready to update` and `LR 0.00e+00`. The nonzero gradient norms
  were only accumulated partial-batch gradients; no optimizer owned a complete
  logical batch, no adapter was stepped, and no `STABLE` broadcast was written.
  The controller was stopped after preserving its evidence and routed inputs.
- Correction: the dedicated Swarm multi-run trainer must run without a finite
  global `max_steps`. `prepare_live_rl_run.py` now requires an explicit
  `--policy-steps` value for each per-policy orchestrator config, and both run
  preparation and the live controller reject a finite trainer packing limit.
  This separates the controller's scientific update horizon from Prime's
  internal token-packing cadence and makes the failure mode impossible to
  repeat silently.

### 2026-08-15 — Infinite-packing pilot rejected a rare PEFT/Prime outlier

- Status: failed closed; no optimizer update and no learned checkpoint
- Preserved run: `/workspace/runs/rl-hf-actor-stability-7a22e4a6`
- Source `7a22e4a6ed259b0f748b66bf6e492526fdd27e8b` separated the
  controller's policy horizon from Prime's internal packing horizon. The full
  Linux suite passed 72/72. A fresh four-group certificate bound to trainer
  config SHA-256
  `3c02c244965dd1c1ab840606dc4f86c414e6f85778d2aeaa098a582569c29cb9`
  passed on 64 samples and 3,902 completion tokens: mean absolute log-prob
  error `0.0016575`, p99 `0.050959`, maximum probability error `0.077751`,
  maximum mismatch-KL `0.049735`, probability-tail fraction `0.002563`, and
  four-policy isolation passed.
- The real pilot generated four new signed groups. Their four policy batches
  contained 16 samples each and 13,228–14,025 tokens. The infinite trainer
  correctly continued past eight packing slices, proving the horizon fix.
- Packing slices 0–10 passed every pre-step parity check. Slice 11 then failed
  before `optimizer.step()` because a previously unseen constrained row had
  maximum mismatch-KL `0.095285 > 0.08`. No policy batch had completed, every
  logged learning rate remained zero, no `STABLE` broadcast existed, and the
  waiting controller was stopped. Gates were not loosened and the run was not
  resumed.
- Diagnosis: certificate samples were forwarded individually, but the live
  4,096-token trainer slice co-packed several 799–975-token trajectories. The
  certificate therefore had not bound the actual packed-forward numerical path.
- Correction under test: use 1,024-token trainer slices and reject a policy
  batch before queue admission unless every sample fits and the two shortest
  samples cannot fit together. This guarantees exactly one trajectory per
  trainer forward, matching the certified path while retaining the truthful HF
  actor. It also reduces memory. No reward, dtype, optimizer, or parity threshold
  changes.

### 2026-08-15 — Single-trajectory pilot exposed slice-local aggregate gate

- Status: failed closed; correction under Linux test
- Preserved run: `/workspace/runs/rl-hf-singletraj-stability-30ad49d9`
- Source `30ad49d9e3d330872a0596b50b63742d036c719d` passed the complete
  Linux suite (73/73). A new four-group, 64-sample, 4,158-completion-token
  certificate for the exact 1,024-token HF+FA2 trainer passed every numerical
  gate and four-policy isolation. Mean absolute log-probability error was
  `0.001699`, p99 was `0.051024`, maximum probability error was `0.077751`,
  mean mismatch-KL was `0.00008281`, maximum mismatch-KL was `0.025083`, and
  probability-tail fraction was `0.002886 < 0.005`.
- The real controller produced and signed all four fresh shared-return groups.
  The first one-trajectory trainer slice then failed before any optimizer step:
  one probability error above `0.05` among 43 completion tokens produced a
  slice-local tail fraction of `1/43 = 0.023256`, above the certified logical-
  batch threshold of `0.005`. No `STABLE` marker or learned checkpoint exists.
- Root cause: single-trajectory packing fixed the forward-path mismatch, but
  Prime applied distribution-level mean, p99, and tail-fraction gates to every
  arbitrary packing slice. Those gates were certified over the complete
  logical policy batch. A small slice turns one allowed rare tail into a false
  rejection even when the full batch passes.
- Correction: accumulate exact parity tensors separately for each isolated
  policy run across packing slices, validate the unchanged gates when that
  logical run becomes ready, and only then permit its optimizer step. Maximum
  errors remain part of the same fail-closed validation; rejected batches still
  cannot update weights. No threshold, reward, optimizer, optimization dtype,
  or reduction dtype was changed.

### 2026-08-15 — Logical-batch FA2 pilot found a genuine cached-forward tail

- Status: failed closed after one isolated policy update; rejected as a
  four-policy stability run
- Preserved run: `/workspace/runs/rl-hf-logicalbatch-stability-cd9bc506`
- Source `cd9bc5064f2b26e8c45dba22349ede99232171b1` passed the full
  Linux suite (84/84). Its fresh 64-sample, 4,141-token certificate passed all
  unchanged gates: mean absolute log-probability error `0.001607`, p99
  `0.051024`, maximum probability error `0.092168`, tail fraction `0.002898`,
  mean mismatch-KL `0.00006677`, maximum mismatch-KL `0.039641`, and all four
  policies remained isolated.
- The first complete logical policy batch passed live parity and made one real
  optimizer step at learning rate `5e-6`. Its gate values included p99
  probability error `0.024223`, tail fraction `0.001017`, and maximum
  mismatch-KL `0.002927`; a stable adapter broadcast was written.
- The next isolated policy batch failed before its optimizer step with maximum
  probability error `0.119982 > 0.1`, p99 probability error
  `0.067708 > 0.05`, and tail fraction `0.012371 > 0.005`. The controller and
  trainer were stopped. Because only one of four shared policies updated, this
  partial checkpoint is not an accepted pilot or a usable RL result.
- Interpretation: logical-batch accounting removed the earlier false positive,
  leaving a genuine rare mismatch between cached token-by-token FA2 actor
  generation and the trainer's full-sequence FA2 forward. Thresholds were not
  loosened. The next diagnostic matches both sides on Hugging Face SDPA; the
  earlier SDPA matrix failure compared against a vLLM actor and does not answer
  this matched-forward question.

### 2026-08-15 — Full-prefix SDPA isolates KV-cache parity failure

- Status: cached diagnostic rejected; full-prefix numerical path admitted for a
  fresh-rollout test
- Preserved run: `/workspace/runs/rl-hf-sdpa-probe-16245067`
- Source `16245067ab4e7d9890cb16bad3c5e353c4ac50b6` passed 89/89
  project-scoped Linux tests. A fresh critical-curriculum group produced 16
  policy samples and 954 completion tokens with nonzero shared-return
  advantages (`+0.190476`, `-0.190476`, `0`, `0`).
- The cached HF-SDPA actor versus HF-SDPA trainer certificate failed only
  maximum mismatch-KL: `0.092147 > 0.08`. Mean absolute log-probability error
  (`0.001923`), p99 (`0.049209`), maximum probability error (`0.074383`), p99
  probability error (`0.021608`), tail fraction (`0.002096`), and mean
  mismatch-KL (`0.0001504`) all passed. No training was attempted.
- Source `b9847abf` added a reusable diagnostic that recomputes the same probe's
  PEFT actor distributions with full-prefix SDPA and no KV cache. This is
  explicitly labeled numerical evidence rather than a valid replacement for
  the behavior-policy probabilities from the original cached rollout.
- The full-prefix certificate passed every unchanged gate and policy-isolation
  check: mean absolute log-probability error `0.001945`, p99 `0.060908`, maximum
  probability error `0.068723`, p99 probability error `0.027233`, tail fraction
  `0.001048`, mean mismatch-KL `0.0001040`, and maximum mismatch-KL `0.050092`.
- Decision: implement an explicitly configured no-KV-cache actor and require
  fresh rollout evidence from that actual behavior policy. This trades actor
  throughput for correct on-policy probabilities; cached probabilities will
  not be relabeled or used for training.

### 2026-08-15 — Four-group probe rejects PEFT as the live behavior backend

- Status: failed closed; no training attempted
- Preserved evidence: `/workspace/runs/hf-parity-nocache-0b1519c4`
- The actual full-prefix HF-SDPA behavior actor completed four balanced
  critical/decoy groups (64 samples, 4,129 completion tokens) in 391 seconds.
  A wider certificate found an outlier absent from the 16-sample probe:
  maximum probability error `0.124935 > 0.1`. The PEFT actor and Prime trainer
  therefore remain numerically different even without KV caching.
- Disabling Prime's grouped LoRA GEMM produced identical certificate metrics,
  ruling out grouped-versus-looped adapter matmul as the cause. Gates remained
  unchanged and neither rejected certificate was used for optimization.
- Correction under test: serve constrained actions from the exact Prime model,
  multi-LoRA slot, SDPA attention, BF16-to-FP32 LM head, and full-prefix path
  used by the trainer. Actor slots remain isolated and accept only
  checksum-pinned PEFT adapter broadcasts. The controller must run as a
  one-rank `torchrun` process so this truthful actor initializes the same
  distributed/FSDP model path as certification and training.
- First integration attempt failed before producing any sample because calling
  the FSDP-sharded backbone directly mixed local token tensors with DTensor
  parameters. The actor now explicitly replicates token and position tensors
  onto the model device mesh, then localizes the world-size-one hidden state and
  LM-head weight only for the final BF16-to-FP32 projection.
- The distributed-input retry reached causal-mask construction, then failed
  before sampling because Transformers mutates DTensor version counters and
  PyTorch `inference_mode` forbids that operation. The Prime actor uses
  `no_grad` instead, matching the certificate and trainer's graph-free
  evaluation semantics without the stricter version-counter prohibition.
- The next retry showed that direct backbone calls also bypass the FSDP root's
  placement hook, leaving rotary buffers local beside DTensor activations. The
  actor now enters through the normal fully-sharded model root and installs a
  scoped LM-head pre-hook that retains only the final hidden token. This keeps
  FSDP placement exact and avoids projecting every prefix token over the full
  vocabulary during autoregressive generation.

### 2026-08-15 — Exact Prime actor passes bound four-group recert

- Status: recertification admitted; one-step optimizer stability pilot running
- Source: `471684263ffd7f13272ee6a2647cbf9b48b80d5b`.
- The first complete one-group rollout from the exact Prime FSDP actor produced
  16 policy samples and 1,040 completion tokens in 308.6 seconds. Its fresh
  certificate passed every unchanged numerical gate: mean absolute
  log-probability error `0.0005634`, p99 `0.021916`, maximum probability error
  `0.022617`, p99 probability error `0.014307`, probability-tail fraction `0`,
  mean mismatch-KL `0.00000909`, and maximum mismatch-KL `0.0040605`. All four
  policy adapter slots passed isolation.
- The first four-group recertification directory,
  `/workspace/runs/prime-parity-4g-47168426`, was terminated by an external
  `SIGTERM` after one of four groups. The controller emitted no diagnostic and
  all CUDA memory was released. This was not a model, parity, or simulator
  failure: the concurrently launched repository pytest session's autouse
  `cleanup_zombies` fixture executes `pkill -f torchrun` at module setup.
- The full project-scoped Linux suite nevertheless completed successfully:
  90 passed with 31 warnings in 20.59 seconds. The interrupted evidence remains
  preserved and is ineligible for certification or training.
- Operational correction: complete pytest before starting any live
  `torchrun`; never use pytest as a concurrent health check on the same host.
  The run-start, monitor, and Swarm asynchronous-operation instructions now
  state this explicitly.
- A new immutable public-preflighted directory,
  `/workspace/runs/prime-parity-4g-r2-47168426`, was created with the same
  source, inputs, trainer config, thresholds, four groups, and balanced
  alternating curriculum. It completed all four groups and 64 policy samples
  in 1,203 seconds. Probe SHA-256:
  `160cd12152e57968ad40cf04f3d92744bd0dc140a95d251874903b1989dd67d1`.
- The bound certificate passed parity and isolation over 3,954 completion
  tokens: mean absolute log-probability error `0.0006103`, p99 `0.0195237`,
  maximum probability error `0.0281846`, p99 probability error `0.0115359`,
  probability-error tail fraction `0`, mean mismatch-KL `0.0000164615`, and
  maximum mismatch-KL `0.00813663`. A disposable optimizer step changed only
  `run_blue_0`. Certificate SHA-256:
  `9468217b71f6d2cd481b279201807ead940374ff4a108b43bf4b7e4c151a6a37`.
- The exact public inputs and unchanged parity gate are now bound into
  `/workspace/runs/rl-prime-stability-47168426`. A one-step four-policy
  optimizer pilot is running with the trainer on GPU 0 and exact Prime actor on
  GPU 1. The model-controlled RED opponent is the frozen SFT adapter; it is not
  a deterministic policy.
- GPU/cost: 4x NVIDIA L40S host; only GPU 1 was used by each rollout actor.
  Provider hourly price remains unknown, so no fabricated dollar estimate is
  recorded. Instance decommissioned: no; required stability and development
  work remains.

### 2026-08-15 — First complete four-policy optimizer stability pass

- Status: completed and admitted as a mechanical RL stability pass
- Run: `/workspace/runs/rl-prime-stability-47168426`; source
  `471684263ffd7f13272ee6a2647cbf9b48b80d5b`; exact Prime actor on GPU 1,
  Prime multi-run trainer on GPU 0, frozen sampled SFT opponent, four balanced
  critical/decoy groups, 16 independent joint-trajectory replicas, and one
  logical update for each of four separately optimized BLUE LoRA policies.
- All four complete logical batches passed the unchanged parity gate immediately
  before `optimizer.step()`. Their maximum probability errors were at most
  `0.0278276`; their maximum mismatch-KL values were at most `0.0140302`.
  Every optimizer step used learning rate `5e-6` and produced a `STABLE`
  filesystem broadcast.
- The training summary passed every mechanical check: four groups, 16 replicas,
  balanced critical/decoy cases, paired sampling namespaces, nonzero learning
  signal, four distinct adapters, and complete policy updates. Mean absolute
  advantage was `0.0751488`; 62.5% of replica advantages were nonzero.
- Final step-one adapter SHA-256 values were `195f1e20...96f8`,
  `cb0f1ff7...15a6`, `e43953d8...ae3`, and `ae715efc...36ae` for BLUE policies
  zero through three. Training-summary SHA-256:
  `11638fa50dec032ce901203853b8eae64a0d7cfb03cf12b797c8901f712f4848`.
- The controller exited normally after recording progress. The trainer is
  deliberately configured with an infinite packing horizon, so it was stopped
  only after all four stable updates were verified; the resulting elastic
  SIGHUP in its tail is expected teardown, not a training failure.
- Verdict: **mechanical pass**. This establishes a working, non-collapsing
  four-policy RL update path. It is not evidence of improved return or learned
  communication; those require the longer run and held-out interventions.
- Next action: a fresh four-step run from the same warm start is now active on
  GPUs 0-1 while the isolated development-evaluation server remains healthy on
  GPU 2. Instance decommissioned: no.

### 2026-08-15 — Four-policy RL v1: mechanically valid updates, scientific admission rejected

- Status: completed; source `471684263ffd7f13272ee6a2647cbf9b48b80d5b`.
  GPU: four NVIDIA L40S host. Exact Prime actor used GPU 1; the Prime trainer
  used GPU 0; an isolated evaluation server used GPU 2. The provider rate was
  not recorded, so no cost is inferred.
- Preflight and parity: a Linux run of the project-scoped suite passed 90 tests
  (31 warnings, 20.59 seconds) before live `torchrun` work. A fresh four-group
  exact-actor certificate (`prime-parity-4g-r2-47168426`) covered 64 samples
  and 3,954 completion tokens. It passed all unchanged gates and policy
  isolation: mean absolute log-probability error `0.0006103`, p99 `0.0195237`,
  max probability error `0.0281846`, p99 probability error `0.0115359`, tail
  fraction `0`, mean mismatch-KL `0.0000164615`, and max mismatch-KL
  `0.00813663`. Probe SHA-256:
  `160cd12152e57968ad40cf04f3d92744bd0dc140a95d251874903b1989dd67d1`;
  certificate SHA-256:
  `9468217b71f6d2cd481b279201807ead940374ff4a108b43bf4b7e4c151a6a37`.
- Training: the long run (`rl-prime-long-4step-47168426`) completed exactly
  three valid logical updates. Every valid update contained four distinct,
  independently optimized BLUE LoRA policies, four balanced critical/decoy
  groups, 16 independent joint-trajectory replicas, shared terminal team
  reward, paired sampling namespaces, and nonzero learning signal. Aggregate
  mean absolute advantage was `0.0713141`, nonzero-advantage rate `0.8333333`,
  and mean return `0.0361149`. The complete three-step summary SHA-256 is
  `bfef06fe4568f55fcb70827ff35ace9af4196011b49c92bde8d0d775ec202163`.
- Fail-closed step four: a fourth proposed update was rejected *before*
  optimizer execution because `p99_probability_error=0.062207937 > 0.05` and
  `probability_tail_fraction=0.01183432 > 0.005`. No threshold, reward,
  optimizer, optimization dtype, or reduction dtype changed. Its partial
  output is excluded from every baseline, checkpoint selection, and claim.
- Development-only evaluation: the 210-row model-pool/intervention suite had
  perfect action protocol and grounded broadcast rates, but candidate-minus-SFT
  ordinary return was `-0.00610169` across 24 paired cells. Candidate normal
  minus dropped, shuffled, delayed, and zero-budget messages was exactly zero
  in all 18 critical paired cells. Opponent-conditioned normal returns were
  `+0.09937` vs base, `-0.01996` vs historical step 1, and `-0.01502` vs SFT.
  This does not pass general capability or communication gates; the frozen OOD
  final evaluation remains unopened.
- Safeguards: candidate-to-start KL was low (overall mean `0.00156537`, p99
  `0.0388221`). Both frozen non-arena regression suites passed with no leakage.
  The initial collapse audit had a serving-label versus stable-policy-label
  alias bug; the audit now resolves explicit `SERVED_MODEL=POLICY_ID` aliases,
  and unit coverage prevents recurrence. The corrected audit found no speech,
  action, repeated-target, excess-KL, or return-without-communication collapse.
  It did correctly flag that model-pool performance was not broad.
- Communication-credit gate, horizon 2: four exact Prime paired rollout-only
  groups found zero intended-sender effects despite target facts and some
  receiver action changes. The proposed message-drop RL repeat was not
  launched.
- Communication-credit gate, horizon 4 Stage B: 12 independent, hash-chained
  critical/decoy pairs were collected across five fresh shards. All 12 had
  identical sender messages and grounded target facts. The intended sender
  mean effect was `+0.0631044` (absolute mean `0.1117224`), with seven positive,
  three negative, and two zero effects. Critical receiver target effects were
  `10/12` versus `2/12` for decoys. However, the predeclared sign gate failed,
  localization was `1.938x < 2x`, and nonzero off-role effects occurred in
  `6/12 > 4/12` pairs. The estimator is rejected: there is information-sensitive
  behavior, but not sufficiently localized counterfactual sender credit for
  small-model per-agent reward assignment. Summary SHA-256:
  `986cbfd771fa5cd5356f99e65ebbe956f45cf74d6441bb66a11dfee9b03cd3ea`.
- Verdict: **mechanical pass; RL-improvement and communication admission
  rejected**. The current environment is not justified for another targeted
  message-credit RL campaign. Redesign the frequency and exclusivity of
  communication-critical states, then rerun the frozen credit gate before
  spending more RL compute.
- Public compact evidence: `results/rl_prime_1_7b_v1/`. No full checkpoint is
  copied to the Mac; only code, reports, manifests, and hashes are retained in
  Git. Final source validation after the audit hardening: `ruff check` passed
  and the full Linux project-scoped suite passed **93 tests** (31 warnings,
  20.05 seconds). The exact cached runtime interpreter was used because the
  detached tools checkout lacks its editable `deps/pydantic-config` dependency;
  no package set, code path, or numerical configuration was substituted.
  The four distinct step-three adapters and all compact reports are public at
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v1`, revision
  `ad51ef261f3e7b7b2d3c6433106bd667ba1da81c`. The publisher anonymously
  downloaded and checksum-verified all 21 public files. Its model card and
  provenance mark the artifact `not-admitted`; it is a reproducibility record,
  not a promoted checkpoint. Instance decommissioned: yes, after final Git
  publication; evaluator and all GPU allocations have been shut down.

### 2026-08-15 — Next-iteration design decision: fast bounded-lag RL and information handoffs

- Status: planned design decision; no new data collection, training, or
  evaluation was run for this entry. It is deliberately separated from the
  rejected v1 result.
- Async execution: the exact Prime actor/trainer route remains a reference and
  calibration harness, not the preferred high-throughput rollout backend. The
  next implementation should use Prime-RL's multi-agent abstraction with a
  community-optimized actor backend and kernels (for example a vLLM-class
  server with FlashAttention) plus asynchronous rollout queues.
- Bounded off-policyness: every accepted trajectory must retain its behavior
  adapter/version hash, behavior log-probabilities, sampling settings, dynamic
  constraint hash, and per-agent token span. Before the next run, predeclare a
  maximum policy-version lag, a behavior-versus-trainer divergence/importance
  bound, and a discard rule for stale or divergent batches. These bounds solve
  rollout staleness; a fixed calibration probe still checks that the actor and
  trainer implement the same constrained distribution closely enough. Exact
  bitwise parity is not the goal, and the old thresholds will not be loosened
  retrospectively to relabel rejected v1 evidence.
- Environment preservation: retain the current 4v4 graph-control game,
  simultaneous actions, strict grounded broadcasts, randomized node names,
  safe discrete simulator, and terminal control-margin reward. Do **not** add a
  speaking bonus, supervised message target, reward shaping, or an action that
  is mechanically unlocked by chat.
- New optional scenario family: add randomized **information handoff** states.
  In such a state, one dynamically selected agent can privately observe a
  valuable exposure/critical fact but cannot exploit it immediately because of
  legitimate map geometry, budget, or current commitment. A different nearby
  teammate can choose among otherwise plausible legal targets but cannot see
  which one is valuable. The sender's grounded fact can therefore improve the
  teammate's next decision, while an unsupported, dropped, delayed, shuffled,
  or irrelevant message cannot be rewarded by construction. Roles must rotate
  with state rather than becoming fixed agent identities.
- Controls: mix ordinary maps, genuine handoff maps, and matched decoy maps in
  the training curriculum so that silence is sometimes correct. The original
  frozen game remains a regression/gameplay suite. After the new scenario is
  specified and tested, generate a new seed-disjoint frozen final for the new
  communication mechanism; the original frozen OOD suite is not silently
  repurposed as its final test.
- Interpretation: three valid v1 updates are enough to establish mechanical
  feasibility, not enough to assess learning. The horizon-4 audit is evidence
  that messages can affect behavior, but it rejects the current sender-credit
  estimator as a per-agent reward signal. A larger or redesigned audit may be
  run only from a new, predeclared plan.
- Next CPU-only work, in order: (1) formalize the handoff state generator and
  invariants; (2) extend the exact solver and replay oracle; (3) construct
  train/development/frozen-final seed manifests plus ordinary and decoy
  controls; (4) run model-free and base/SFT message-intervention audits; and
  (5) implement the async rollout record, calibration probe, and stale-batch
  rejection logic. GPU work begins only after these gates are fixed.
- Next GPU work, if the CPU gate passes: first a no-update fast-backend
  calibration/throughput pilot, then a longer shared-terminal-reward baseline
  against a model-controlled opponent pool. A credit-aware comparison is
  permitted only if the new counterfactual credit audit passes its frozen gate.
  Instance decommissioned: yes; no GPU is required for the next CPU phase.

### 2026-08-15 — RL v4 CPU gate: information handoffs, harder eval, and async admission

- Status: completed CPU design, generation, audit, and unit verification. No
  GPU was allocated and cost was zero.
- Hypothesis: communication should improve a teammate's selection among actions
  that are already legal, rather than mechanically unlock an action. This
  directly addresses the v3 audit's weak/localized sender signal without adding
  a message reward or changing the game.
- Environment: `arena-information-handoff-v2` creates balanced two-world
  bundles. The receiver has identical stale observations and the same legal
  actions in `left_exposed` and `right_exposed`; a remote sender privately sees
  the active exposure and cannot exploit either candidate. The exact
  decentralized solver permits one receiver action across the dropped-message
  information set and separate actions after the message. A matched decoy
  changes only receiver knowledge, making both worlds distinguishable without
  chat and giving exact zero message value.
- Generated manifests: 240 train bundles at 12/13 nodes and horizons 4/5; 48
  development bundles at 14/16 nodes and horizons 6/8; and 24 frozen OOD
  bundles at 18/20 nodes and horizons 8/10. All 12 ordered sender/receiver roles
  are exactly balanced. The hard ordinary development and frozen suites each
  contain 24 seed-disjoint cases; the frozen suite uses 18/20 nodes and 8/10
  turns. Curriculum stages mix ordinary, critical handoff, and matched-decoy
  cases at 50/25/25 then 70/15/15.
- Handoff manifest SHA-256 values: train
  `665408b25a62eb276e70be1ca2716472c0c53abe3cbcc0c1a4f84d4a10ef9681`;
  development
  `1c70c8ff8253535853e8684c1d2c01a76e9b36dd34d68f3f695e12fa617fc031`;
  frozen OOD
  `be928d297d49954e4a29c1343e13df40580edb39a810d297d66b06b492402bc2`.
  Hard ordinary SHA-256 values: development
  `0e60c12e10c322ee081d666aa02d6237cd65e6e8992e19722b225a4c35b3bf4a`;
  frozen OOD
  `73f58e8fec965569ccb4e190e7c65ac1e3fe87f2806c8cbacac1af85e66efb6e`.
- Exact audit: every scenario reconstructs from its manifest and content hash;
  all old/new split-overlap sets are empty; all action sets match across latent
  worlds; both captures are already legal; critical receiver worlds are
  indistinguishable without the message; decoy worlds are distinguishable;
  sender observations and structural worlds match critical to decoy; and the
  sender cannot act on either candidate. Every invariant rate is `1.0`.
  Certified minimum critical advantage is positive against balanced,
  aggressive, and defensive opponent anchors: train mean `0.06625352`
  (minimum `0.05128205`), development mean `0.05685693` (minimum
  `0.04651163`), and frozen mean `0.04569230` (minimum `0.03921569`). Every
  matched decoy has exact zero advantage.
- Evaluation: `arena-rl-progress-eval-v4` separates paired candidate-RL minus
  SFT capability on legacy and hard ordinary maps from causal communication on
  critical handoffs. It reports normal minus dropped, sender-shuffled, delayed,
  and zero-budget effects, opponent slices, and a null decoy control. Bootstrap
  units are whole procedural seeds or two-world bundles—not agents, turns,
  sides, or repeated interventions. Online, checkpoint-selection, and frozen
  tiers prevent training-time peeking at the final suite.
- The resumable runner now executes the complete matrix. Online is 96 games
  with one SFT opponent and normal/dropped messages; selection is 1,296 games
  across three opponent families and all interventions; frozen final is 3,168
  games. The legacy final uses 24 map seeds under three legal-option orders
  (72 paired cells), while the new hard and handoff finals use 18/20 nodes and
  8/10-turn horizons. Frozen execution requires an explicit canonical design
  digest, and raw trajectories are intended to stay on remote storage rather
  than being copied to the Mac.
- Async implementation: the exact Prime actor is now explicitly a reference
  and calibration route. `swarm_ctf_eval.async_admission` permits a fast
  community actor backend only with immutable behavior adapter hashes,
  revisions, update indices, exact constraints, complete current-policy
  log-probabilities, an approved no-update calibration, and explicit
  precommitted bounds for lag, log ratios, importance ratios, and probability
  drift. Limits have no arbitrary defaults and are hashed into the run lock.
  Stale/divergent batches are discarded atomically; frozen-opponent changes and
  malformed evidence fail admission. Rejected v1 evidence remains rejected.
- Verification: the broader dependency-free CPU suite passed 51 tests after
  adding async admission, handoff invariants, manifest balancing, tier locking,
  progress endpoints, and resume tests. Ruff passed every changed Python file.
  A clean regeneration matched every committed generated file byte-for-byte,
  and the independent global audit passed with SHA-256
  `940447c0feb9eef484ec088a239125075e851b9cec35ea86280acd3bbde99052`.
  Torch/vLLM-dependent Linux tests were not installed on the Mac and remain a
  pre-GPU-host gate.
- Existing evidence boundary: the v3 frozen OOD suite is unchanged. V4 has not
  trained or evaluated a model yet, so this entry establishes task validity and
  measurement design, not learned capability or communication.
- Next GPU gate: run a no-update production-backend calibration/throughput
  probe, freeze its calibration digest and admission limits, then launch a small
  shared-terminal-reward stability pilot against the base/SFT/historical model
  pool. Escalate to a longer run only if development capability,
  communication-intervention, regression, and collapse gates remain healthy.

### 2026-08-15 — RL v4 SFT baseline and stage-1 reward-density run

- Status: completed pre-training evaluation and rollout-only diagnostic; no
  optimizer update. Verdict: the 1.7B SFT is RL-usable, but no communication or
  RL-improvement claim is admitted.
- Hardware/runtime: one user-provided NVIDIA RTX A6000 48 GB host. vLLM 0.22
  served the pinned Qwen3-1.7B BF16
  backbone plus dynamic rank-16 LoRAs. The first evaluation process started at
  approximately 11:02 UTC and the last model rollout completed at approximately
  11:53 UTC. Using the earlier user-reported `$0.48/hour` rate for this same
  host class gives roughly `$0.41` for that 51-minute GPU-active window; later
  artifact handling is not included. Instance decommissioned: not yet at this
  log checkpoint; inference has been stopped and GPU state is 1 MiB / 0%.
- Immutable models: base revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; SFT revision
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`; adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.
- Linux gate: after all live-v4 fixes and the rollout-summary implementation,
  the complete project-scoped suite passed **90 tests** with two third-party
  SWIG deprecation warnings in 35.80 seconds. Ruff passed the changed files.
- Serving compatibility: the first static LoRA launch exposed a vLLM 0.22
  loader-signature change (`moe_ep_spec`). Commit `3fdf72b1` updated Prime's
  key-prefix wrapper and the repaired server registered both `base` and `sft`.
  A production dynamic-adapter broadcast/action probe then passed.
- Development baseline: `run_progress_eval_v4.py --tier online` completed all
  96 games. Its fixed historical label `candidate_rl` is the SFT adapter in
  this pre-training run, and `sft_init` is the untouched base; model revisions
  in the manifest remove ambiguity. SFT-minus-base return was `+0.2868501`
  (95% interval `[+0.1191502,+0.4653010]`) on legacy ordinary maps and
  `+0.2031388` (`[+0.1572712,+0.2720588]`) on hard ordinary maps. Action,
  broadcast, and grounded-broadcast rates were all `1.0`.
- Communication baseline: critical normal-minus-dropped was directionally
  positive at `+0.0597636`, but its four-bundle interval
  `[-0.0022629,+0.1079741]` crossed zero. The matched-decoy effect was
  `+0.0388356` with interval `[-0.0334280,+0.1110991]`. The online tier
  correctly rejected a communication claim and left the larger selection and
  frozen tiers unopened.
- Live v4 integration failures: the first rollout called the v3 scenario
  reconstructor; the retry reached `HandoffScenario` but assumed a single
  `.state` instead of its two latent worlds. Both failed before any model
  decision, evidence admission, or optimizer input. Commits `c1c2fdb2` and
  `7af9d102` dispatch to the correct reconstructor and choose worlds
  deterministically by pair index, pairing critical and decoy on the same
  world. The v4 run lock now binds composite train/curriculum, development, and
  frozen-evaluation hashes (`535c29bd...`, `e1a46a3b...`, `440bc41e...`) and
  uses the scenario-certified horizon unless an ablation explicitly overrides
  it.
- Reward-density diagnostic: 24 signed shared-return groups used the exact
  stage-1 mixture: 12 ordinary, six critical, six matched decoy. Four
  independently sampled complete-game replicas per group produced 96 games.
  Overall mean return was `+0.0261111`, range `[-0.30,+0.6206897]`, mean
  absolute leave-one-out advantage `0.1042770`, nonzero-advantage rate
  `0.9583333`, and within-group return-variance rate `0.9583333`. Mean absolute
  advantage was `0.1488890` ordinary, `0.0597869` critical, and `0.0595430`
  decoy. Critical-minus-decoy mean return was `-0.0212387`; this is a
  pre-training diagnostic, not a communication estimate.
- Decision unlocked: pure verified terminal team return is sufficiently dense
  for the next optimizer run. Do not add speaking, capture, or message bonuses.
  Start multi-GPU asynchronous RL with the 50/25/25 stage-1 mixture, four
  independent BLUE LoRA policies, a model-controlled opponent pool, bounded
  policy lag, and the existing signed replay/admission boundary. Use the
  96-game online tier for directional monitoring; keep selection and frozen
  final closed until checkpoint selection.
- Compact public artifacts: `results/rl_v4_pretrain_1_7b/`. Baseline file
  hashes are `5c8bb163...` manifest, `dc290a42...` rows, and `876e38a2...`
  summary. Content-addressed reward-density summary SHA-256 is `8618fb4b...`.
  The 59.8 MB raw
  evaluation trace remained off the Mac and compressed to 2.6 MB for public
  release attachment. The complete 7.4 MB raw/log/evidence archive has SHA-256
  `162eae39d36cc7906c2d865ed924d8eaa140237b8d10910c50f966ac695f8fd0`
  and excludes supervisor signing keys. No model checkpoint was copied locally.
  Public release:
  `https://github.com/ChinmayK0607/blog-rl/releases/tag/swarm-arena-v4-pretrain-2026-08-15`.

### 2026-08-15 — RL v4 production orchestration and fresh L40S allocation

- Status: completed eight-update development run, three-opponent online
  evaluation, regressions, KL and collapse audit; selection and frozen final
  remained closed.
- User directive: finish the CPU production pass, then use a newly provisioned
  4x NVIDIA L40S host for approximately three to four hours to obtain useful RL
  evidence. The provider rate was not supplied, so cost will not be invented.
- Fresh-host inspection: all four 46,068 MiB GPUs were idle at 0 MiB/0%,
  temperatures 24--25 C; the 2.5 TB filesystem had 2.3 TB free; no trainer,
  rollout, vLLM, or stale tmux process existed. The reused address presented a
  new ED25519 fingerprint, so its key was isolated in a task-specific temporary
  known-hosts file instead of replacing the developer's normal SSH trust file.
- CPU contract under implementation: train each BLUE policy's own `BROADCAST`
  and `ACT` spans across every episode turn; generate the exact 50/25/25
  ordinary/critical/matched-decoy schedule before sampling; rotate base, SFT,
  historical, and current model-controlled opponents exactly; bind the complete
  plan into the run lock; and admit/reject only complete four-policy groups.
- Async policy: begin with an optimized vLLM-class actor at explicit lag zero.
  A separate same-backend worker must prove current adapter hashes equal the
  behavior snapshot; Prime still checks the independently calibrated numerical
  envelope immediately before optimization. This is not called lag-one/two
  async RL. A real current-policy constrained rescorer is required before
  admitting stale rollouts.
- Reward remains only verified normalized terminal team return with the
  leave-one-out replica baseline. No message, action, capture, silence, judge,
  or other additive shaping term is introduced.
- Promotion automation under implementation: resume-safe checkpoint export,
  online 96-game monitoring, both non-arena regression suites, policy KL,
  collapse audit, model-pool cross-play, and public artifact publication. The
  monitor is prohibited from opening selection/frozen evaluation during
  training.
- The fresh clone initially repeated the repository's SSH-submodule problem for
  `renderers` and `verifiers`; only the clone-local submodule URLs are being
  changed to public HTTPS. This is an invocation/preflight issue, not an RL
  result, and no GPU process has started.
- Linux CPU validation completed before any `torchrun`: changed-file Ruff
  passed; 15 focused production/async/shared-return tests passed in 36.46 s;
  and the complete project-scoped Swarm suite passed **98 tests** with only the
  two known third-party SWIG deprecation warnings in 39.06 s.
- Three independent vLLM 0.22.0 servers on GPUs 1--3 passed the exact live
  structured probe with the pinned SFT adapter: one valid 83-token broadcast
  and one valid 8-token legal action. The probe SHA-256 is
  `74cb20fef8c3e11c5ca416eb1185744d558eb4f825ef36bf4512bfbdae94ec46`;
  inference-config SHA-256 is
  `300f6f5910456fae7aa93c8c7c97c34caf2b5ca2d028433782ef3d8daffe4420`.
  Both enter immutable production-plan hash
  `65f5e0f719ab5b13524d5c12c3e3c41844d011727bee08ce83a7639b1eb01d43`.
- Rejected canary `/workspace/runs/rl-v4-canary-57aff316`: its first complete
  group passed replay, signing, four-policy ownership, and lag-zero async
  admission with 2,848 selected BLUE tokens and exactly zero behavior/current
  log-ratio. Before the remaining groups were generated, the controller raised
  `KeyError: pair_index`. A production ordinary assignment had incorrectly
  fallen through to the legacy alternating-pair namespace branch. The trainer
  was healthy and waiting; no batch was sent, no optimizer ran, and no learned
  checkpoint exists. This is rejected orchestration evidence, not an RL result.
- Correction: sampling-namespace selection is now a tested pure helper.
  Production ordinary games never receive a paired namespace; matched critical
  and decoy scenarios retain their shared namespace; the old non-production
  alternating path keeps its explicit fallback. No reward, curriculum mix,
  numerical bound, optimizer, dtype, or model configuration changed. After the
  fix, changed-file Ruff and 5 focused tests passed, followed by the complete
  Linux suite: **100 passed**, two known third-party SWIG warnings, 38.35 s.
- Rejected retry `/workspace/runs/rl-v4-canary-bc7376ec`: all four groups
  passed replay and lag-zero admission, then three of four policy batches made
  isolated step-1 updates. Their full logical-batch parity summaries passed;
  examples include mean log-probability error `0.00143--0.00228`, p99
  `0.0382--0.0699`, probability-tail fraction `0.00122--0.00273`, mean
  mismatch-KL `0.0000688--0.000137`, and maximum mismatch-KL
  `0.0631--0.0940`. The fourth batch was rejected before its update solely
  because one token had maximum probability error `0.251636 > 0.16`; every
  other predeclared aggregate and KL bound passed. The three partial adapters
  are rejected and will never seed another run.
- Prospective v4b decision: the single-token maximum probability difference is
  now diagnostic (`1.0`, its mathematical upper bound). Mean and p99 log-prob
  error, p99 probability error, probability-tail frequency, and mean/maximum
  mismatch-KL remain unchanged and fail closed. This is a new source/config
  hash and new run, not a retroactive pass. The change follows the previously
  documented optimized-backend design: sparse kernel outliers are controlled
  by distributional bounds and DPPO's trust-region mask rather than allowing
  one isolated maximum to veto an otherwise bounded logical batch.
- Rejected v4b retry `/workspace/runs/rl-v4b-canary-d90d7cc8`: all four
  rollout groups passed replay, signing, four-policy ownership, and lag-zero
  admission. The trainer processed the complete logical batch, then rejected
  it before producing a valid four-policy checkpoint solely because one token
  had maximum mismatch-KL `0.206909 > 0.13`. Every aggregate bound passed,
  including mean and p99 log-probability error, p99 probability error,
  probability-tail frequency, and mean mismatch-KL. No adapter from this run
  is eligible to seed training.
- Prospective v4c decision: both raw single-token maxima are diagnostic on the
  optimized actor. `max_probability_error` remains `1.0` and
  `max_mismatch_kl` becomes `1.0`; the latter is a catastrophic-outlier
  ceiling, not a mathematical KL bound. The aggregate gates remain strict and
  are now locked: mean log-probability error `0.01`, p99 log-probability error
  `0.15`, p99 probability error `0.07`, probability-tail fraction `0.02` above
  error `0.05`, and mean mismatch-KL `0.001`. If a fresh run fails any of these
  aggregate gates, the vLLM route is rejected rather than relaxing them.
- Accepted v4c canary `/workspace/runs/rl-v4c-canary-1379f9c2`: all four
  rollout groups passed replay, lag-zero rescore/admission, and the locked
  aggregate parity bounds; all four independently optimized policies wrote a
  stable step-1 adapter. Across the four logical policy batches, mean
  log-probability error was `0.00154--0.00208`, p99 log-probability error
  `0.0425--0.0678`, p99 probability error `0.0187--0.0237`, probability-tail
  fraction `0.000310--0.002473`, and mean mismatch-KL
  `0.0000576--0.000168`. The largest raw mismatch-KL was `0.138117`, confirming
  it was a sparse maximum rather than an aggregate drift failure. The canary
  adapter-set revision is
  `eb4ff30b0235872c0750eb2a153754c8a0618e2ebd118c058b16eabf35d07ce1`.
- A fresh eight-update production run
  `/workspace/runs/rl-v4-production-1379f9c2` was then initialized from the
  original pinned SFT adapter, not from the canary. It uses the same source,
  production-plan, data, opponent rotation, reward, and locked parity hashes.
- The fresh production run completed all eight optimizer updates: 32 signed,
  replayed, rescored, and admitted groups and four separately optimized LoRA
  policies. Update-level `(mean return, mean absolute advantage, nonzero
  advantages / 16)` was: `(0.00622, 0.09758, 12)`, `(-0.02053, 0.11518, 13)`,
  `(0.09083, 0.08444, 12)`, `(0.00332, 0.06992, 16)`,
  `(0.07407, 0.12654, 16)`, `(0.00721, 0.10907, 16)`,
  `(0.03251, 0.07850, 8)`, and `(0.11687, 0.10485, 15)`. This demonstrates
  dense, nondegenerate learning signal; it is not a monotonic learning curve.
- The final adapter-set revision is
  `d60f358d32448f26f7cce7ecc6f9f53f212906303e0deeda779354a44129b156`.
  Role adapter SHA-256 values are `b2ee1004...` (blue-0), `aa032a5e...`
  (blue-1), `9735e411...` (blue-2), and `a27efc4a...` (blue-3). Final update
  aggregate parity remained within every locked bound: mean log-probability
  error `0.00256--0.00343`, p99 log-probability error `0.0768--0.0990`, p99
  probability error `0.0285--0.0343`, probability-tail fraction
  `0.00352--0.00534`, and mean mismatch-KL `0.000179--0.000350`.
- Three disjoint 96-game online development monitors used the base, SFT, and
  historical opponent families. Candidate action validity, broadcast protocol
  validity, and broadcast grounding were all `1.0`. Capability differences
  versus SFT were mixed: hard-suite differences were `+0.0306`, `+0.0193`, and
  `-0.0280`; legacy differences were `-0.0351`, `+0.0187`, and `+0.0307` for
  base, SFT, and historical opponents respectively. With only four independent
  paired units per cell, these are development signals, not a capability claim.
- Critical normal-minus-dropped message effects were positive against every
  opponent: `+0.0846` (base, 95% CI `[+0.0397,+0.1295]`), `+0.0315` (SFT,
  `[-0.0063,+0.0711]`), and `+0.0904` (historical,
  `[+0.0219,+0.1677]`). Matched-decoy effects were `+0.0583`, `-0.00525`, and
  `+0.0393`. The base decoy confidence interval excludes zero, so the run shows
  promising communication sensitivity but does **not** establish
  information-specific communication or emergent swarm cooperation.
- The reference-state candidate-to-SFT KL probe passed comfortably: overall
  mean `0.000410`, p99 `0.01261`, maximum `0.03790`; per-policy mean KL was
  `0.0000557--0.000879`. Scope is 16 constrained reference-state broadcast
  samples, not the full on-policy state distribution. Both 128-case non-arena
  regressions passed for all four policies: v1 exact match stayed `0.17969`; v2
  improved from SFT `0.51172` to `0.51953--0.53125`; leakage remained zero.
- The first collapse-audit invocation exposed a summary-schema assumption
  (`KeyError: candidate_normal_return_by_opponent`). The audit now derives its
  paired metrics from immutable rows; four focused tests and the full Linux
  suite passed (**102 tests**, two known SWIG deprecation warnings, 40.00 s).
  The corrected audit passed all stop/inspect gates: no action, speaking,
  repeated-target, KL, single-opponent, or return-without-message collapse.
  Per-role speaking rates were `0.3529--0.3566`.
- The four standard PEFT adapters and seven compact reports were uploaded to a
  public, anonymously downloaded, checksum-verified Hugging Face bundle at
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-step8-development`,
  revision `6a660a3fabfebd3270753155a131d2148d463b82`. It is explicitly labelled
  **not admitted**. No checkpoint was copied to the Mac.
- Operational note: the first KL diagnostic failed before model loading because
  PEFT was absent. The remote environment was repaired with `uv`, the retry
  passed, and the failed log was retained. At completion there were no vLLM,
  trainer, rollout, or evaluation processes and every GPU was idle. Provider
  rate was not supplied, so cost is not estimated.
- Next scientific action: do not open selection/frozen evaluation for this
  checkpoint. Redesign the next training mix to require the recipient to use
  private sender information while matched decoys remain causally inert, then
  run a longer predeclared training schedule and select exactly one checkpoint
  before opening the next tier.

### 2026-08-16 — RL v4 fresh 30-update run and compact checkpoint pulse

- Status: completed, published, and GPU processes stopped. Verdict:
  **mechanical pass and exploratory information-specific communication signal;
  capability improvement and RL-specific communication improvement not
  established; not admitted**.
- Hypothesis and decision: test whether the exact v4 production path remains
  stable for 30 updates and whether later checkpoints improve task return or
  causal message use. The run retained verified terminal team return only,
  four distinct BLUE LoRA policies, and the exact 50/25/25
  ordinary/critical/matched-decoy schedule. No shaping reward or gate was
  relaxed.
- Immutable training identity: source commit
  `12e0c461a28c3d0311d0353ab1ed45bcffb0b569`; production-plan semantic
  SHA-256 `65f5e0f719ab5b13524d5c12c3e3c41844d011727bee08ce83a7639b1eb01d43`;
  base revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; SFT revision
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`; SFT adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.
  The offset-capable development evaluator is public at commit `6d6fe88e`.
- Hardware/runtime: four NVIDIA L40S 46,068 MiB GPUs. Actor startup was
  approximately 18:19:41 UTC, update 30 completed at approximately 21:06:37
  UTC, and public verification plus GPU shutdown completed at 21:22:05 UTC:
  about 3 h 02 min of allocated-node wall time for this run segment. The
  provider rate was not supplied, so cost is not invented.
- Launch retries: preparation first failed before run creation because
  `PYTHONPATH` was absent. The first trainer entrypoint then failed before model
  loading because single-rank distributed variables were unset; it was
  relaunched under `torchrun --standalone`. Both failures were retained as
  invocation evidence and produced no accepted update.
- Training result: all 30 updates completed, comprising 120 complete groups and
  480 game replicas. Mean return was `+0.0173472`, mean absolute advantage was
  `0.0964001`, and mean nonzero-advantage rate was `0.8625`; every update had
  nonzero learning signal. All four adapters were distinct and all four changed
  at every update. The final step-30 adapter-set revision was
  `32d3e0d71c5c05c2d3592826067b75eb9b7c5fbd125c516acc8e3fdc36c20880`.
- Admission/parity: 120/120 logical policy batches passed and no admission
  record was rejected. Worst aggregate values across the run were mean
  log-probability error `0.00391042`, p99 log-probability error `0.12181`, p99
  probability error `0.0488602`, probability-tail fraction `0.00952931`, and
  mean mismatch KL `0.000456511`, all inside the locked bounds.
- Checkpoint preservation: Prime retained only the two newest broadcaster
  snapshots, so a fresh same-lineage step 10 could not be recovered after that
  retention window. Fresh steps 18, 20, and 30 were exported before pruning.
  The public earlier step-8 checkpoint was included only as an explicitly
  independent baseline, never represented as part of the fresh learning curve.
- Pulse design: a fixed 66-game development pulse used one ordinary case and
  one critical/decoy pair, base/SFT/historical opponents, both sides, all five
  critical message conditions, and normal/dropped decoys. The first pulse
  invocation used `data/rl_v4`, but the compact v3 curriculum runner requires
  `data/rl_v3/development.json`; all four jobs failed before their first model
  request. The empty failed directories were archived, the exact path was
  corrected, and the rerun completed. This was an orchestration failure, not a
  model result.
- Pulse result: all action, broadcast, and grounding rates were `1.0`. The
  predeclared score selected fresh step 20. Ordinary candidate-minus-SFT was
  `-0.05357`, `+0.10714`, and `+0.04762` at fresh steps 18, 20, and 30. The
  independent earlier step 8 was `-0.05952`. The tiny pulse had no positive
  information-specific message effect and was used only for selection.
- Non-overlapping development holdout: three new ordinary cases and three new
  critical/decoy pairs produced 198 games across the three inference GPUs.
  Step-20 ordinary candidate-minus-SFT was `+0.00344379`, effectively flat.
  Critical normal-minus-dropped, sender-shuffled, delayed, and zero-budget were
  each `+0.0444444`; matched-decoy normal-minus-dropped was `-0.0579710`.
  Protocol and grounding stayed `1.0`. This supports information-specific
  message sensitivity on these held-out cases, but the SFT critical
  normal-minus-dropped effect was also `+0.0444444`. RL preserved the behavior;
  it did not demonstrably improve it.
- Opponent result: the collapse audit's ordinary-only mean candidate returns
  were `+0.18697` against base, `+0.04114` against SFT, and `-0.06972` against
  the historical league. This is not single-opponent collapse, but it rules out
  a claim of uniform opponent-pool improvement.
- Regression/KL/collapse: both 256-case frozen non-arena suites passed for all
  four policies with zero leakage. V1 deltas versus SFT ranged from `0` to
  `-0.0078125`; v2 ranged from `+0.0078125` to `+0.01171875`. Overall
  constrained candidate-to-SFT KL mean was `0.0013720` and p99 `0.0248497`.
  BLUE-3 p99 was `0.216858`, below the fixed `0.30` limit but a role asymmetry to
  monitor. No action, speech, repeated-target, excessive-KL, opponent, or
  return-without-message collapse flag fired; per-role speaking rate was
  `0.54545`.
- Summary-tool failure and fix: the existing shared-return training summarizer
  assumed all four groups had curriculum `kind` and `pair_index` fields. The
  production 50/25/25 mix includes two ordinary groups, so it raised
  `KeyError`. Ordinary groups are now handled explicitly and both the legacy
  all-curriculum and production mixtures are tested. Linux Ruff passed and the
  two focused tests passed.
- Selected artifact: step-20 revision
  `7fe8ff458b73ea055b9f28b5b95db13961b3bc839298c294f12e2a81c527223d`;
  policy SHA-256 values are `4e25a6c8...`, `18ce7da6...`, `37092538...`, and
  `280d9b13...`. The public bundle was anonymously downloaded and all 17 files
  checksum-verified at
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-long-development`,
  revision `1af877668ee3cdd8dd5ccd4734ce620bbe5e2aa0`. It is labelled
  `not-admitted`; the temporary Hub credential was removed after publication.
- Compact Git artifacts: `results/rl_v4_1_7b_long/` contains the full 30-update
  progress/summary, pulse, selection rule, non-overlapping holdout, KL,
  regression, collapse, publication record, and human-readable report. No
  checkpoint was copied to the Mac.
- Final state: all vLLM, trainer, controller, rescorer, and evaluation processes
  stopped; all four GPUs reported 0 MiB. The node is safe to decommission.
- Next action: do not open selection or frozen final for this run. The strongest
  justified conclusion is that the RL system is stable and the task measures
  information-specific communication, while 30 updates did not improve either
  holdout capability or communication over SFT. A future run should use a
  longer schedule or a stronger optimization change and compare RL-minus-SFT
  intervention effects directly.

### 2026-08-15/16 — RL v4 learning-rate and curriculum ablation

- Status: completed; compact evidence copied locally; all GPU processes stopped.
  Verdict: **ordinary-capability positive, communication negative, not
  admitted**.
- Hypothesis and decision: the mechanically stable 30-update v4 run was flat
  against SFT. Test whether stronger optimization alone helps, then whether a
  communication-heavier curriculum helps. Keep the frozen selection/final
  suites unopened, terminal team return unchanged, four independently
  optimized BLUE LoRAs, and the model-controlled opponent pool unchanged.
- Immutable identity: training source commit
  `12e0c461a28c3d0311d0353ab1ed45bcffb0b569`; offset evaluator commit
  `6d6fe88e`; base revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
  SFT revision `534522a8f3ff3489b1dd8318dc8e533e51264cde`; SFT adapter
  SHA-256 `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.
- Hardware/runtime: four NVIDIA L40S 46,068 MiB GPUs. Actor processes were
  ready from approximately 21:53 UTC. Variant A trained from approximately
  21:55--22:57 UTC; Variant B ran from approximately 23:00--23:43 UTC; matched
  evaluation and diagnostics completed by approximately 00:22 UTC. The
  provider price was not supplied, so cost is not invented.
- Variant A: learning rate `1e-5`, exact 2 ordinary / 1 critical / 1 matched
  decoy update mix. All 12 updates completed: 48 groups and 192 replicas. Mean
  return `+0.00717239`, mean absolute advantage `0.107124`, and mean nonzero
  advantage rate `0.822917`. All four adapters were distinct, changed every
  update, and every update had nonzero signal.
- Variant B: learning rate `1e-5`, exact 2 ordinary / 3 critical / 3 matched
  decoy mix. Four updates were valid: 32 groups and 128 replicas, mean return
  `+0.0149210`, mean absolute advantage `0.0861203`, and mean nonzero advantage
  rate `0.7890625`. All eight update-5 rollout groups were admitted, but one
  policy hit the unchanged per-token mismatch-KL ceiling
  (`1.0974264 > 1.0`) before its optimizer step. The other three policies
  passed that update. The gate was not relaxed; only steps 1--4 are evidence.
- Matched pulse: identical offset-6, 66-game development windows compared
  Variant A steps 8 and 12 and Variant B step 4. Ordinary candidate-minus-SFT
  was respectively `+0.154762`, `+0.0833333`, and `+0.0357143`, so the
  predeclared capability rule selected Variant A step 8. Every critical
  intervention effect was `0`; the pulse made no communication claim.
- Non-overlapping development holdout: offsets 9--11 yielded 198 unique games.
  Variant A step-8 ordinary candidate-minus-SFT was `+0.0707407` over 18 paired
  cells. Ordinary-only candidate returns were `+0.28` against base, `+0.06`
  against SFT, and `+0.0166667` against the historical league. All four
  candidate message-intervention effects were exactly `0`; SFT dropped-message
  and matched-decoy effects were also `0`. Protocol and grounding stayed `1.0`.
- Regression/KL/collapse: both 256-case frozen non-arena suites passed for all
  four policies with zero arena leakage. V1 deltas ranged from `-0.00390625` to
  `0`; v2 from `-0.00390625` to `+0.0078125`. Overall constrained
  candidate-to-SFT KL mean was `0.000782232`, p99 `0.0161926`, and max
  `0.0924678`. No action, speech, repeated-target, excessive-KL, or
  single-opponent collapse flag fired. The audit correctly stopped on the sole
  `return_gain_without_message_gain` flag.
- Startup and orchestration failures, all before consuming relevant model
  requests unless stated otherwise: a broad `XDG_CACHE_HOME` rewired UV and
  caused three CPU-only actor exits; the UV user install/cache was repaired;
  config tokenization required `inference @ FILE`; shared vLLM and Triton
  compile caches raced and were split per actor; `/v1` controller URLs produced
  `/v1/v1/load_lora_adapter` and were changed to server roots; Variant B first
  lacked its log directory; the first pulse used an evaluator without offset
  flags; two holdout launches failed on a `set -u` local expansion and a missing
  background-worker `cd`. The first Variant-A summary repeated the already
  fixed ordinary-group `pair_index` error and was rerun with the fixed
  summarizer. No failed launcher was counted as model evidence.
- Diagnostic wrapper failures: the first KL attempt used an interpreter without
  PEFT and loaded no model. PEFT `0.20.0` was added only to the cached trainer
  environment, which also resolved its cuDNN package from `9.22.0.52` to
  `9.19.0.56`; no training occurred afterward. The first collapse invocation
  used stale `holdout-blue-*` aliases and failed identity validation; the
  trajectory-discovered `ablate-a8-blue-*` aliases were then supplied, with no
  metric/gate change.
- Checkpoints: Variant A step-8 role SHA-256 values begin `9912e5c6`,
  `927b7f32`, `8a1aa6de`, and `845d93ae`; Variant B step-4 values begin
  `126f1904`, `c2a1beaa`, `b8f554db`, and `c81cb370`. No checkpoint was copied
  to the Mac. Variant A step 8 was published as an explicitly non-admitted
  development model at
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-lr1e5-step8-development`,
  revision `395f933ea16185c0f18087269cc8cc850e55257d`. An anonymous download
  checksum-verified all four adapters; Hub metadata reported `private=false`.
- Compact artifacts: `results/rl_v4_1_7b_lr_ablation/` contains exact configs,
  progress and training summaries, the matched pulse, 198-game holdout summary,
  policy KL, all regression comparisons, collapse audit, hashes, launchers, and
  a human-readable report. Raw trajectories and adapters remain excluded.
- Interpretation: doubling the learning rate produced a useful ordinary-game
  gain by step 8 without regression or policy collapse. Simply increasing
  communication-case density at the same learning rate was unstable, and none
  of the evaluated policies showed causal message use on the new holdout. The
  next run should change the communication-critical curriculum so that a
  private fact must reach a different role to affect the terminal transition,
  while preserving terminal team return and reporting capability and message
  effects separately.
- Instance decommissioned: GPU workloads stopped and all four GPUs reported
  `0 MiB` at approximately 00:22 UTC. Final Git publication followed from the
  local compact copy.

### 2026-08-16 — staged 120-update curriculum and publishable metric logging

- Status: CPU implementation completed; GPU run not started.
- Verdict: implementation evidence only; no new model or scientific result.
- Hypothesis: gradually increasing matched information-handoff density can
  produce RL-specific message use without the instability seen in the abrupt
  `2/3/3` run and without sacrificing ordinary game capability.
- Decision unlocked: one predeclared 1.7B run can now test the curriculum before
  deciding whether to move to 4B.
- Source commit: pending at entry creation; record the final commit after push.
- Data: reuses oracle-certified `data/rl_v4/handoff_train.json` pairs 0--151.
  Development and frozen v4 manifests were not edited or regenerated.
- Reward and policy structure: unchanged terminal control-delta team return,
  no additive message/action bonus, four independent LoRA policy routes and
  optimizers, exact base/SFT/historical/current opponent rotation.
- Curriculum: 120 updates x 4 groups. Stage counts are 20 tactical updates at
  `2/1/1`; 40 introduction updates using four `2/1/1` updates then one `0/2/2`;
  40 communication-heavy updates using three `2/1/1` then two `0/2/2`; and 20
  harder self-play updates with the same latter pattern. Aggregate schedule:
  176 ordinary, 152 critical, 152 matched-decoy groups. Every update has equal
  critical/decoy counts. Generated audit: plan SHA-256
  `56cda2e8c193905c1cb05b551409877aaa134dffc61c3379b36994bd12ef7f34`,
  schedule SHA-256
  `462c4ff6fcbabbd715de290f125f603d9f25939b076c9ddcdfc1e05079f5fb66`,
  curriculum SHA-256
  `02f3a2308c71bef45fdddd665655d338473f0d685378bbf985f82cf4e481aa14`,
  handoff manifest SHA-256
  `665408b25a62eb276e70be1ca2716472c0c53abe3cbcc0c1a4f84d4a10ef9681`.
  These audit hashes used the existing Variant-A runtime plan as a local
  fixture; rebuild and record fresh hashes after binding paths/revisions on the
  next GPU host.
- Optimization: LoRA rank 16, constant `7.5e-6` learning rate. This is between
  the stable/weak `5e-6` run and fast but abrupt `1e-5` communication-heavy
  failure. No fictitious within-run LR changes are declared.
- Evaluation change: optional `--rl-specific-communication` adds SFT
  normal/dropped critical rollouts and reports both
  `(RL critical effect - SFT critical effect)` and
  `(RL critical effect - RL matched-decoy effect)`. Default v4 evaluation
  behavior is unchanged.
- W&B change: trainer config now logs Prime trainer telemetry to
  `swarm-arena-rl`. A separate failure-isolated sidecar logs logical update,
  terminal return, nonzero/absolute advantage, exact mix, curriculum stage,
  return by scenario/opponent, protocol, capability, RL-specific communication
  lift, and critical-minus-decoy specificity. It uploads only compact progress
  and summary JSON; credentials remain environment-only.
- Verification: all edited Python files passed `compileall`; the focused
  curriculum/progress/W&B pytest set passed `11/11`. The dependency-light wider
  suite passed `69` tests; its sole executed failure was an environment import
  error for `prime_rl`, while five collection files could not import optional
  Mac-absent `httpx`, `torch`, or `huggingface_hub` dependencies. No behavioral
  assertion failed. Deterministic staged-schedule, plan-builder, and
  W&B-metric smoke assertions also passed. The generated schedule has 480
  groups, maximum handoff pair index 151, and 176 unique ordinary seeds.
- Local tooling failure: the first dependency-resolving pytest invocation did
  not run tests because local Homebrew `uv 0.9.2` cannot parse the repository's
  newer `exclude-newer = "7 days"` setting and crashed during macOS system
  configuration discovery. Offline `uv` has no cached pytest. This is recorded
  as an environment failure; the complete Linux pytest gate remains mandatory
  before any `torchrun` or paid rollout.
- GPU, wall time, cost: no GPU allocated; zero GPU cost. No checkpoint or model
  cache was downloaded to the Mac.
- Artifacts: `STAGED_RL_PLAN.md`,
  `data/rl_v4/staged_curriculum_v1.json`,
  `configs/rl_v4_1_7b_staged.toml`,
  `scripts/build_staged_rl_plan.py`, `scripts/log_live_rl_wandb.py`, and
  `swarm_ctf_eval/progress_eval_v5.py`.
- Next action: on 4x4090 or 4xL40S, run the full Linux tests, public-input
  preflight, staged-plan binding/audit, a four-update mechanical smoke, and then
  the 120-update run with online development measurements every ten updates.
  Do not open selection/frozen tiers during the run.
- Instance decommissioned: not applicable; no rented instance used.

## Artifact index

- Public source branch:
  `https://github.com/ChinmayK0607/blog-rl/tree/exp/swarm-arena-4b`
- Initial SFT dataset:
  `https://huggingface.co/datasets/CK0607/swarm-arena-sft-v2`
- Pinned 1.7B constrained warm start:
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible`
- Beginner gameplay explainer:
  `https://swarm-arena-gameplay.chinmayk.chatgpt.site/`
- Compact replay:
  `results/replay/seed0-full-match.json`
- Pre-RL parity evidence:
  `results/pre_rl_1_7b/`
- Rejected counterfactual audits:
  `results/pre_rl_1_7b/base_counterfactual_24/` and
  `results/pre_rl_1_7b/paired_counterfactual_24/`
- RL v3 manifests and audits:
  `data/rl_v3/`
- RL v4 task, progress evaluation, manifests, and audits:
  `RL_TASK_V4.md`, `PROGRESS_EVAL_V4.md`, and `data/rl_v4/`
- RL v4 pre-training baseline and stage-1 reward-density evidence:
  `results/rl_v4_pretrain_1_7b/`
- RL v4 30-update development result:
  `results/rl_v4_1_7b_long/` and
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-long-development`
- RL v4 stronger-learning-rate development result:
  `results/rl_v4_1_7b_lr_ablation/` and
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-lr1e5-step8-development`
- Frozen message-credit admission plan:
  `MESSAGE_CREDIT_AUDIT_PLAN.md`
- Public, non-admitted mechanical RL artifact:
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v1`

## Future entry template

Copy this block for each material run:

```markdown
### YYYY-MM-DD — short run name

- Status: planned | running | completed | failed | stopped
- Verdict: mechanical pass | exploratory signal | admitted | rejected
- Hypothesis:
- Decision unlocked:
- Source commit:
- Base / adapter / opponent revisions:
- Data split and manifest SHA-256:
- GPU, wall time, and estimated cost:
- Exact launcher/config:
- Predeclared gates:
- Results:
- Failures and retries:
- Artifact paths and hashes:
- Interpretation:
- Next action:
- Instance decommissioned: yes/no/time
```
