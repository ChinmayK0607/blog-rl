# Swarm Arena research log

Last updated: 2026-08-26
Branch: `exp/swarm-arena-4b`  
Current public development checkpoint: focused atomic update 50, Hugging Face
revision `049e95062903501a8a50efac09d1b2caab393364` (not admitted).
Status: the immutable 80-update focused-credit run was prospectively capped at
update 60. It produced a directional capability gain but degraded the causal
communication measures, so it is useful negative evidence rather than a swarm-
communication result.

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
- Implementation commit: `88a5fc34f1fa6f5d471f8badeae44498a5291062`.
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

### 2026-08-16 — second-pass audit of the staged 120-update run

- Status: CPU-side implementation and audit completed; GPU certification and
  training not started. Verdict: the long run now has a fail-closed launch path,
  but it is not authorized until the exact rented host produces a fresh passed
  runtime certificate and Linux test result.
- Audit motivation: the first staged design left several operational failure
  modes that could waste a long rental. Prime removes ordinary broadcast
  snapshots aggressively; the W&B sidecar could stop before the last
  evaluation; online trainer logging could kill training during an outage;
  later curriculum stages used unnecessarily long episodes; the pulse used one
  rollout server and 32 rows; and the old backend identity did not bind serving
  and numerical parity into one host-specific artifact.
- Curriculum correction before GPU use: all 120 training updates now remain at
  12/13 nodes and 4/5 turns. Difficulty increases through matched
  communication density and the adaptive current-policy opponent, while the
  larger 18/20-node, 8/10-turn cases remain evaluation-only. The immutable
  schedule still contains 480 groups: 176 ordinary, 152 critical, and 152
  matched decoys, with 176 unique ordinary seeds and handoff pairs 0--151.
  Canonical curriculum SHA-256 is
  `2dd25a990ddae837502dcdbe261dd87e67e5ca6657cb0dc9e775fb211283adcd`;
  schedule SHA-256 is
  `73f9e5b888282cdd1a311a1f5d1581ad2fae5868be1c6f1723091ec814a197de`.
  File SHA-256 is
  `b69484f10c58b563b3fb6b930a1c496969743f92755f529cae802abb3bcb145c`.
- Runtime binding: `capture_runtime_parity_probe.py` captures 32 fresh
  constrained decisions, balanced across all four policy slots and exercised
  across all three rollout servers. `certify_prime_parity.py` now records the
  exact probe digest. `bind_runtime_certificate.py` refuses dirty source or a
  failed/mismatched report and binds source commit, base revision, adapter
  bytes, resolved trainer config, inference config, vLLM version, driver and
  four-GPU inventory, the three-server broadcast/action probe, and passing
  numerical parity plus four-policy optimizer isolation. The production-plan
  builder replaces inherited backend identity with this certificate; a
  machine-specific production-plan hash therefore does not exist until the GPU
  host passes certification.
- Public fail-closed preflight: `preflight_staged_rl.py` independently checks a
  clean exact commit, anonymous public base/adapter/source availability, local
  model and adapter bytes, resolved config and parity-gate hashes, certificate,
  installed vLLM, exact GPU/driver inventory, three serving registries, at
  least 20 GiB free disk, 120 updates, four independent trainer slots, LoRA
  rank 16, `7.5e-6` LR, 2,560-token trainer versus 4,096-token actor context,
  every reconstructed handoff pair, unique ordinary seeds, the exact
  base/SFT/historical/current opponent rotation, and all opponent adapter
  hashes. It writes `swarm-staged-rl-preflight-v2`; the launcher cannot start
  without a pass.
- Evaluation/checkpoint synchronization: the controller now blocks before
  update 0 and after updates 10--120 on content-hashed ready/continue files.
  The step-zero 16-row BLUE-only pulse must show exact SFT-vs-SFT invariance
  before any optimizer step. Subsequent pulses measure ordinary capability,
  candidate critical normal-minus-dropped, RL-minus-SFT communication lift,
  and critical-minus-decoy specificity. Agent rosters are distributed across
  all three actor servers. Before a continuation is written, the pulse process
  requires all four per-policy checkpoints to contain `STABLE`, optimizer
  state, adapter config, and adapter bytes whose SHA-256 exactly matches the
  policy evaluated at that barrier. This catches Prime multi-run checkpoint
  errors that are logged rather than raised.
- Storage/logging: every tenth per-policy checkpoint is permanently retained;
  only two intervening checkpoints are kept. The trainer W&B run is offline so
  network loss cannot affect optimization; the controller/evaluation sidecar
  remains failure-isolated and waits for the explicit pulse-completion marker.
  No checkpoint, model, or raw rollout was downloaded to the Mac.
- Exact checked-in trainer-config SHA-256 is
  `6cef3afd20df3a15ab34f53d00270bdff66d68cf39fa93db648c5e3f03f95597`;
  inference-config SHA-256 remains
  `300f6f5910456fae7aa93c8c7c97c34caf2b5ca2d028433782ef3d8daffe4420`.
  The generated staged plan before host binding had canonical SHA-256
  `c82ba8e19d6bff18fb58009ad08b3225b2b71bc35c44347803f1f99245823369`;
  it is superseded for execution because the final plan must include the fresh
  host certificate.
- Verification: changed Python files passed `compileall` and Ruff; the launch
  shell passed `bash -n`; `git diff --check` passed; CLI help smoke tests passed
  for the certificate binder, plan builder, preflight, and pulse runner. The
  focused staged curriculum/progress/checkpoint set passed 15 tests. The wider
  dependency-light set passed 69 tests; its sole executed failure was the
  already-recorded Mac environment absence of `prime_rl`, not a behavioral
  assertion. Six optional-runtime collection files were excluded because the
  lightweight Mac environment lacks `httpx`, Torch, or Hugging Face packages.
  A dependency-resolving `uv` attempt also failed before testing because the
  local clone has intentionally unpopulated workspace submodules; an attempted
  isolated Ruff resolution could not reach PyPI under the sandbox. Cached
  tools were used successfully instead. The complete repository Linux suite
  remains the first paid-host action and must run before `torchrun`.
- Runtime/cost estimate: the prior 4x L40S run took approximately three hours
  for 30 updates. With short training horizons and thirteen distributed
  16-game pulses, budget roughly 12--14 hours for 120 updates on the same class,
  then measure actual throughput over the first five updates. A materially
  slower trajectory is a stop-and-debug condition, not permission to continue
  unattended.
- GPU, wall time, cost: no GPU allocated; zero GPU cost. Instance
  decommissioned: not applicable.
- Next action: on the exact four-GPU host, run the full Linux suite, prepare a
  fresh run directory, start the three isolated actor servers, create the fresh
  serving/parity certificate, build the host-bound plan, require preflight v2,
  then launch. Do not reuse an older certificate or open selection/frozen data
  during the learning curve.

### 2026-08-16 — staged 120-update L40 launch-control failure and fix

- Status: stopped before update 1; corrected restart pending.
- Verdict: rejected as training evidence; useful launch-harness finding.
- Hypothesis: the fresh 4x L40 runtime can execute the predeclared staged
  120-update, four-policy LoRA run with causal communication pulses every ten
  updates.
- Decision unlocked: keep the runtime/parity configuration, but correct the
  step-zero evaluator control and make the failure-isolated W&B sidecar
  explicitly offline before restarting from a new immutable source commit and
  run directory.
- Source commit: `ff2cf86417a3a12c819eb6015196eeb16e217c68`.
- Base / adapter / opponent revisions: Qwen3-1.7B
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; SFT step 320
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`, adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`;
  historical RL-v1 `ad51ef261f3e7b7b2d3c6433106bd667ba1da81c`, blue-0
  adapter SHA-256
  `1004e012cd96a6377006c334d997825e3ebb25828b482a4644b7149a823d873a`.
- Data split and manifest SHA-256: staged curriculum
  `2dd25a990ddae837502dcdbe261dd87e67e5ca6657cb0dc9e775fb211283adcd`;
  handoff training manifest
  `665408b25a62eb276e70be1ca2716472c0c53abe3cbcc0c1a4f84d4a10ef9681`;
  generated 480-group schedule
  `73f9e5b888282cdd1a311a1f5d1581ad2fae5868be1c6f1723091ec814a197de`.
- GPU, wall time, and estimated cost: host `216.81.248.55:40300`, four NVIDIA
  L40 46,068 MiB, driver `580.126.09`; setup/certification and the stopped
  zero-step launch consumed about 30 minutes. Exact provider price was not
  supplied, so no fabricated dollar total is reported.
- Exact launcher/config: `scripts/launch_staged_rl.sh`, prepared trainer SHA-256
  `8cc214002599cb6f557205f7b7fb02f8cebed21c268500f38abada13f26fb05d`,
  inference config SHA-256
  `300f6f5910456fae7aa93c8c7c97c34caf2b5ca2d028433782ef3d8daffe4420`,
  generated plan SHA-256
  `249282737a727e49e726c0a101fcb19e604fa89b843c6220f3bdc73ab64064be`.
- Predeclared gates: all 108 Linux tests passed. The exact three-server serving
  probe passed. A fresh 32-sample parity corpus contained 16 BROADCAST and 16
  ACT decisions. Runtime certificate
  `9455b13f9ac6008a3cf5985ead436e63329410108f7cdd608a45f185355407ae`
  passed: mean absolute log-probability error `0.00228549`, p99 `0.0749279`,
  probability-tail fraction `0.00219459`, mean mismatch KL `0.000121251`, max
  mismatch KL `0.0353732`; optimizer parameter sets were disjoint and a test
  update changed only `run_blue_0`.
- Results: public-input verification, preflight v2, serving, parity, and
  four-policy isolation all passed. The controller reached only the step-zero
  checkpoint barrier. Trainer logs show `Starting from step 0` and no optimizer
  metric or progress record was written; therefore there were zero optimizer
  steps and no model result to retain.
- Failures and retries: the first parity command omitted `torchrun` and failed
  before model load because `RANK` was unset; its directory/log was preserved,
  and the unchanged command passed under single-rank `torchrun`. The update-0
  pulse then rejected ordinary-hard SFT-vs-SFT behavioral invariance
  (`-0.1388889`) while legacy and RL-specific communication differences were
  exactly zero. Root cause: the candidate arm used four byte-identical LoRA
  aliases while the baseline used one shared alias. vLLM may batch those paths
  differently, so a greedy choice near a numerical tie can diverge even though
  no weights changed. This tested alias-kernel equivalence rather than the
  intended evaluator invariant. The fix routes both step-zero arms through the
  same registered SFT alias; adapter checksums, controller registration checks,
  and fresh trainer/serving parity continue to cover the four trainable aliases.
  All post-zero pulses still use `blue-0` through `blue-3`. The W&B sidecar also
  attempted online login because a tmux server did not inherit `WANDB_MODE`;
  the launcher now passes its explicit `--offline` flag.
- Artifact paths and hashes: failed run remained only on the paid host at
  `/workspace/runs/rl-v4-staged-120-ff2cf864-l40-20260816`; no checkpoint or raw
  trajectory was copied to the Mac. Compact certificate/config evidence will
  be copied only after the corrected run establishes progress.
- Interpretation: fail-closed synchronization worked as intended and prevented
  the first optimizer step. Exact model-behavior equality is a valid harness
  control only when both arms use the same serving alias; numerical alias-path
  equality belongs in the separately bound parity/registration evidence.
- Next action: test and publish the two-line launch behavior fix, update the
  remote exact commit, create a new run directory, recapture/rebind parity to
  that commit, and require a passing step-zero pulse plus observed update-1
  progress before leaving the long run unattended.
- Instance decommissioned: no; inference GPUs remain live for the corrected
  restart, trainer GPU is idle.

### 2026-08-16 — rejected exact-behavior null pulse on corrected L40 launch

- Status: stopped before update 1; second corrected restart pending.
- Verdict: rejected as training evidence; evaluator-control diagnosis.
- Hypothesis: routing both step-zero arms through the same registered SFT alias
  would make two independently generated temperature-zero games behaviorally
  identical.
- Decision unlocked: do not use full-game behavioral identity as a serving
  determinism gate. Require exact four-model roster identity and complete valid
  protocols at step zero; retain the separately bound token-level parity test
  as the numerical trainer/serving gate.
- Source commit: `37adfe2bf4a6b69f21f482e0fada259caf9901a6`.
- Base / adapter / opponent revisions: unchanged from the preceding entry.
- Data split and manifest SHA-256: curriculum
  `2dd25a990ddae837502dcdbe261dd87e67e5ca6657cb0dc9e775fb211283adcd`;
  handoff train
  `665408b25a62eb276e70be1ca2716472c0c53abe3cbcc0c1a4f84d4a10ef9681`;
  schedule `73f9e5b888282cdd1a311a1f5d1581ad2fae5868be1c6f1723091ec814a197de`.
- GPU, wall time, and estimated cost: four L40s on
  `216.81.248.55:40300`; about 15 minutes for fresh certification, preflight,
  and the stopped pulse. Provider price was not supplied.
- Exact launcher/config: runtime certificate body SHA-256
  `67edc6d4571085c8bf4e01b27939b2f7eff00c42c03c8c8f6b308e1ff20e7c82`;
  production plan body SHA-256
  `ccd03763fb24871b9a7b4398a70a78cf318b26971104d27f7c6d8db65230e3f7`;
  trainer config SHA-256
  `a74ab9bccce2140536323bd0c16ebe54d7db61faf30dc3f205198c4e7c2da25d`.
- Predeclared gates: 109 Linux tests passed. Fresh 32-sample parity passed
  with mean absolute log-probability error `0.00228372`, p99 `0.0743217`,
  probability-tail fraction `0.00146306`, mean mismatch KL `0.000116482`,
  max mismatch KL `0.0353720`, disjoint optimizer sets, and single-policy
  isolation.
- Results: public-input preflight passed and both W&B streams opened explicit
  offline runs. The step-zero pulse completed 16 games. Legacy and hard
  capability differences were exactly zero and all action/broadcast/grounding
  protocol rates were 1.0. One of four critical handoff outcomes diverged,
  yielding RL-specific communication lift `-0.0909091` despite both arms using
  the exact same four `sft-opponent` aliases.
- Failures and retries: temperature-zero greedy generation is not a promise of
  bitwise identical multi-turn trajectories on GPU. A near-tied token can
  change under nondeterministic CUDA/vLLM reduction or batching order; one
  changed action then changes later observations. Exact full-game equality was
  therefore an invalid null gate. The trainer remained blocked at step zero;
  logs contain no optimizer metric or live progress record.
- Artifact paths and hashes: rejected raw trajectories stay only in
  `/workspace/runs/rl-v4-staged-120-37adfe2b-l40-20260816`; no large artifact
  was copied to the Mac.
- Interpretation: configuration identity is deterministic and appropriate for
  this integration smoke test; independent game returns are not. Token-level
  numerical parity, adapter hashes, alias registration, and policy-isolation
  checks continue to cover the failure modes that matter before optimization.
- Next action: test and publish the corrected null-control semantics, recapture
  a certificate bound to the new commit, and require observed update-1 metrics.
- Instance decommissioned: no; inference servers remain resident, GPU 0 idle.

### 2026-08-16 — admitted 120-update staged L40 run (running)

- Status: running; one of 120 logical updates completed at this checkpoint.
- Verdict: admitted paid run after end-to-end step-zero and first-update gates.
- Hypothesis: a four-policy shared-return LoRA curriculum can improve game
  capability while communication-critical and matched-decoy pulses distinguish
  information use from generic tactical learning.
- Decision unlocked: leave the staged run active through the 10-update pulse
  cadence; do not interpret the first update as a learning result.
- Source commit: `504f97ceaab21c1a531b52ad42849362ddf4266a`.
- Base / adapter / opponent revisions: Qwen3-1.7B
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; SFT step 320
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`, adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`;
  historical RL-v1
  `ad51ef261f3e7b7b2d3c6433106bd667ba1da81c`, adapter SHA-256
  `1004e012cd96a6377006c334d997825e3ebb25828b482a4644b7149a823d873a`.
- Data split and manifest SHA-256: task version
  `arena-rl-v4-information-handoff`; train
  `535c29bd5be475c615554c0fbd4aedab8566f16baf9ee0a2d69a42d92800ec38`;
  development
  `e1a46a3b3a54b7410e330f49e59fae6d4ea0a7764d081e9df050fd6f27bd89eb`;
  frozen OOD
  `440bc41ef06a755a0174245640fb6a837d31af39372fa4bf90994574feb63dec`;
  staged curriculum
  `2dd25a990ddae837502dcdbe261dd87e67e5ca6657cb0dc9e775fb211283adcd`;
  handoff train
  `665408b25a62eb276e70be1ca2716472c0c53abe3cbcc0c1a4f84d4a10ef9681`;
  480-group schedule
  `73f9e5b888282cdd1a311a1f5d1581ad2fae5868be1c6f1723091ec814a197de`.
- GPU, wall time, and estimated cost: host `216.81.248.55:40300`, four NVIDIA
  L40 46,068 MiB, driver `580.126.09`. Cold setup through admitted update 1
  took about one hour; the first logical update took about five minutes after
  the step-zero pulse. Current projection is 12--14 hours plus preservation
  buffer. Provider price was not supplied, so no dollar total is fabricated.
- Exact launcher/config: run directory
  `/workspace/runs/rl-v4-staged-120-504f97ce-l40-20260816`; launcher
  `scripts/launch_staged_rl.sh`; trainer config SHA-256
  `9697346463ca18ed9a900aca9896efba36901b2c4a9dff212c783b1583b69250`;
  inference config SHA-256
  `300f6f5910456fae7aa93c8c7c97c34caf2b5ca2d028433782ef3d8daffe4420`;
  runtime certificate body SHA-256
  `0a84893ebded5e013674924fb74dd21b312526bf3a7e73df3b69b12b956144a7`;
  production plan body SHA-256
  `fe7b1f7673135dc467f0f61a1f5a4782cdb0eef0fadecd5f989ef2e937892ee9`.
- Predeclared gates: all 110 Linux tests passed. Fresh three-server serving
  probe passed. The 32-sample production parity certificate passed with mean
  absolute log-probability error `0.00230364`, p99 `0.0743217`, max probability
  error `0.0939987`, p99 probability error `0.0244452`, probability-tail
  fraction `0.00219459`, mean mismatch KL `0.000118507`, max mismatch KL
  `0.0353719`; optimizer parameter sets were disjoint and a single-policy test
  changed only `run_blue_0`. Public-input preflight passed with 2.27 TiB free.
- Results: the 16-game step-zero pulse completed with action, broadcast, and
  grounding protocol rates all `1.0`; the descriptive RL-specific
  communication difference was `0.0`. The first logical update admitted four
  groups and 16 stochastic replicas. Mean return was `-0.00183132`, with 50%
  nonzero returns; means by case kind were decoy `0.0322581`, ordinary
  `-0.0197917`, and critical `0.0`. These are workload-density diagnostics,
  not an improvement claim. Immediately before optimization, all four policy
  parity gates passed. Four optimizer steps reported LR `7.5e-6`, and the
  resulting adapters had distinct SHA-256 values: blue-0
  `cb1ad4c0fa9379dfde4df232872b428ff70f58ad3d02bc1c52831bf6a607b49a`,
  blue-1 `733e87cb94edfe8739b37cfd7c9a1a6f0162cf4210741d2ef8379e07f0082e78`,
  blue-2 `f7bf95e4651750e5ac80b5e81e39dfd406f2f4de591db3def24e1a86755f2b40`,
  and blue-3 `3cb57f79fdecacd896cac9e225c296b2b2b6d70ff923880b3c330ffd28010b31`.
- Failures and retries: fresh preparation initially failed because the wrapper
  pre-created an output directory that the fail-closed preparer requires to be
  absent; the empty directory was removed and preparation was rerun unchanged.
  Certification was then mistakenly invoked first with one visible GPU, which
  the four-slot isolation matrix rejected, and once with the broader
  kernel-variant matrix whose thresholds do not bind the production trainer.
  Both non-binding attempts were preserved; the correct trainer-config-bound
  certifier then passed. Prime-RL's aggregate step logger often prints LR
  `0.00e+00` after a packed slice because it queries `ready_to_update_idxs`
  after that slice is consumed. Source inspection confirmed optimizer creation
  uses each run's configured LR before `optimizer.step`; the four actual
  per-policy readiness/parity events at the end of logical update 1 each logged
  LR `7.50e-06`. Treat aggregate zero-LR lines as a monitor bug, not optimizer
  evidence.
- W&B: trainer and controller/evaluation telemetry are captured in explicit
  offline runs at
  `wandb/offline-run-20260816_091653-kfbl8uil` under the run directory and
  `/workspace/blog-rl/wandb/offline-run-20260816_091624-rl-v4-staged-120-504f97ce-l40-20260816-controller-v1`.
  The controller run logs stage/opponent returns, communication interventions,
  collapse diagnostics, and the immutable production plan/curriculum as compact
  artifacts. Cloud sync remains pending because the rented host has no W&B
  credential; do not auto-delete until these two directories are synced.
- Artifact paths and hashes: compact evidence remains in the run directory;
  raw trajectories and checkpoints stay on the paid host and were not copied to
  the Mac. Public source is the exact commit above.
- Interpretation: mechanical admission is complete and actual learning has
  begun. No claim about capability or communication improvement is valid until
  the predeclared longitudinal pulses and final cross-play/intervention suite.
- Next action: monitor updates 2--10, inspect the first causal communication
  pulse and collapse diagnostics, then continue only while all components
  remain healthy. Sync W&B and publish compact results before decommissioning.
- Instance decommissioned: no; recommended auto-delete no earlier than
  `2026-08-17 09:45 IST` (`04:15 UTC`) to leave a preservation buffer.

### 2026-08-16 — staged run stopped at update 8 by unbounded max statistic

- Status: failed closed after eight complete logical updates; fresh restart
  pending.
- Verdict: rejected as a final learning result; valid systems and throughput
  evidence through update 8.
- Hypothesis: the `max_mismatch_kl = 1.0` catastrophic-outlier ceiling would
  catch genuine serving/trainer divergence without vetoing an otherwise
  bounded logical policy batch.
- Decision unlocked: make raw single-token mismatch-KL a finite catastrophic
  sentinel (`100.0`) rather than a scientific quality threshold. Keep the
  strict mean, p99, and tail-frequency gates unchanged and start a fresh run;
  do not retroactively admit the stopped batch.
- Source commit: `504f97ceaab21c1a531b52ad42849362ddf4266a`.
- Base / adapter / opponent revisions: unchanged from the admitted launch entry
  above.
- Data split and manifest SHA-256: unchanged from the admitted launch entry;
  only schedule ordinals 0--31 produced complete logical updates.
- GPU, wall time, and estimated cost: four L40s on
  `216.81.248.55:40300`; about 57 minutes from admitted update 1 to the trainer
  stop. Provider price was not supplied.
- Exact launcher/config: run directory
  `/workspace/runs/rl-v4-staged-120-504f97ce-l40-20260816`; runtime certificate
  `0a84893ebded5e013674924fb74dd21b312526bf3a7e73df3b69b12b956144a7`;
  production plan
  `fe7b1f7673135dc467f0f61a1f5a4782cdb0eef0fadecd5f989ef2e937892ee9`.
- Predeclared gates: all aggregate gates remained under their limits at the
  rejected batch. The only reported violation was raw single-token
  `max_mismatch_kl = 1.4104266 > 1.0`; therefore the trainer rejected before
  that policy optimizer step and the controller timed out rather than
  publishing a partial logical update.
- Results: eight complete logical updates were recorded (steps 0--7). Mean
  returns by update were `-0.001831`, `0.059769`, `0.063333`, `0.013136`,
  `0.064815`, `-0.019692`, `0.025107`, and `0.114834`. The last update had 15
  of 16 nonzero returns. This short, changing-case sequence is promising reward
  density but is not a comparable evaluation curve and makes no communication
  claim. No update-10 pulse or retained checkpoint was reached.
- Failures and retries: after about 1,976 packed trainer slices, one near-zero
  chosen token exceeded the unbounded maximum mismatch statistic. Mean
  mismatch-KL, p99 errors, and tail fraction did not fail. With DPPO masking,
  the isolated token is not evidence that the batch distribution is unsafe.
  Leaving `1.0` as a hard maximum would make longer runs more likely to fail
  solely as a function of token count. The new prospective config changes only
  this maximum to `100.0`; aggregate thresholds and LR remain locked.
- W&B: the trainer and controller records were finalized and synced to
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/kfbl8uil` and
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v4-staged-120-504f97ce-l40-20260816-controller-v1`.
- Artifact paths and hashes: the failed run occupies 579 MiB on the host. Raw
  data was not copied to the Mac. Compact evidence remains in the public log
  and the synced W&B runs.
- Interpretation: fail-closed behavior was mechanically correct, but the
  maximum statistic was not length-robust. This is a parity-gate design failure,
  not evidence of reward collapse or a failed curriculum.
- Next action: publish/test the prospective gate semantics, recertify a new
  immutable source/config hash, restart from the common SFT initializer, and
  require update 10 plus its first communication pulse before judging promise.
- Instance decommissioned: no; inference remains resident for a cheap restart.

### 2026-08-16 — fresh certified staged 120-update restart

- Status: stopped fail-closed after 25 complete logical updates.
- Verdict: valid exploratory systems/curriculum evidence through update 25;
  rejected as a completed learning run.
- Hypothesis: a length-robust distributional parity gate permits the staged
  shared-return run to continue while still rejecting systematic
  serving/trainer drift.
- Decision unlocked: require the first complete logical update and the
  predeclared update-10 causal-communication pulse before interpreting the
  run; retain the same learning rate and curriculum.
- Source commit: `683a19212ac327c368935fa934449497bb04fb28`.
- Base / adapter / opponent revisions: base `Qwen/Qwen3-1.7B` at
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; initial adapter
  `CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible` at
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`, local adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.
- Data split and manifest SHA-256: task version
  `arena-rl-v4-information-handoff`; train
  `535c29bd5be475c615554c0fbd4aedab8566f16baf9ee0a2d69a42d92800ec38`,
  development
  `e1a46a3b3a54b7410e330f49e59fae6d4ea0a7764d081e9df050fd6f27bd89eb`,
  final `440bc41ef06a755a0174245640fb6a837d31af39372fa4bf90994574feb63dec`.
  The immutable schedule contains 480 groups over 120 logical updates: 152
  critical, 152 paired decoy, and 176 ordinary cases.
- GPU, wall time, and estimated cost: four NVIDIA L40 46,068 MiB GPUs on
  `216.81.248.55:40300`; provider price was not supplied. Three GPUs serve
  rollouts and GPU 0 trains four independent LoRA policy slots.
- Exact launcher/config: run directory
  `/workspace/runs/rl-v4-staged-120-683a1921-l40-20260816`, session prefix
  `swarm-120-683a`, config `configs/rl_v4_1_7b_staged.toml`, constant LR
  `7.5e-6`, checkpoint/pulse interval 10.
- Predeclared gates: fresh 32-decision certificate passed with mean absolute
  log-probability error `0.0020536655`, p99 absolute error `0.0598721877`,
  p99 probability error `0.0225713346`, mean mismatch-KL `0.0001295755`, raw
  maximum mismatch-KL `0.0508227348`, and disjoint four-policy optimizer
  parameter sets. Runtime certificate SHA-256
  `96bbe2d9b03af51bdcd1eec53a18bbe8cb6bc2125e64ecb60899619233eb1145`;
  production-plan SHA-256
  `44ea3918e1e09ace953aa8d4f000ef4dc81a2ff6a79a4c3efa7601de18b70aac`.
- Results: logical update 1 (schedule step 0) completed. All four policy slots
  passed their online pre-optimizer parity gates and stepped at the configured
  LR `7.5e-6`. Across the 16 game replicas, mean return was
  `0.0304603495`, 8/16 returns were nonzero, and the range was
  `[-0.0416666667, 0.2]`. The resulting joint policy revision is
  `acf2f97d95b0e0a2855c854a2630db9c94b3bfa4e8df768357d5d7b43b8483cb`.
  At the eight-update checkpoint, per-update mean training returns were
  `0.030460`, `0.092689`, `0.041667`, `0.012842`, `0.039352`, `0.023250`,
  `-0.019835`, and `0.035278`; 8--13 of 16 replicas per update had nonzero
  return. This establishes reward density and optimizer activity, not a
  learning trend, because the scheduled cases and opponents change by update.
  Each of the four policy slots had a distinct adapter hash after every update.
  The frozen update-0 pulse confirmed the initializer's protocol and grounded
  broadcast rates were both `1.0`; critical normal-minus-dropped return was
  `0.151515`, while the matched-decoy effect was `0.0`. RL-specific lift was
  correctly `0.0` at initialization. This one-independent-unit pulse validates
  the causal evaluation wiring but is not a communication-learning claim.
  At update 10, protocol and grounded-broadcast rates remained `1.0`; legacy
  capability versus SFT was `+0.214286`, hard capability was unchanged, and
  critical normal-minus-dropped remained `+0.151515`. Because the SFT baseline
  had the same critical effect, RL-specific communication lift was `0.0`; the
  decoy effect also rose to `+0.060606`, so the communication claim did not
  pass. At update 20, legacy capability remained `+0.214286`, but hard
  capability was `-0.166667`, critical normal-minus-dropped was `-0.030303`,
  and RL-specific communication lift was `-0.181818`. These pulse cells each
  contain only one independent unit and are directional diagnostics, not final
  estimates. They nevertheless argue against blindly resuming the identical
  optimization configuration.
- Failures and retries: the first public-source preflight attempt used an
  incorrectly expanded commit URL and failed with HTTP 404 before creating a
  run directory or starting a trainer. The exact full commit URL passed on the
  immediate retry. Linux tests passed `110/110` before certification. During
  the attempted 26th logical update, the trainer rejected a batch because
  p99 absolute rollout/trainer log-probability error was `0.15295246`, narrowly
  above the predeclared `0.15` distributional bound. No partial logical update
  was published; the controller later timed out waiting for all four policies.
  Unlike the previously rejected raw-maximum gate, p99 is a robust aggregate
  statistic, so this stop must not be waved through without diagnosing whether
  drift is localized to a policy/phase/token regime.
- W&B: credentials were forwarded through netrc without printing the token.
  Trainer telemetry is in
  `wandb/offline-run-20260816_114828-ingmc3tn` under the run directory;
  controller/evaluation telemetry is in
  `/workspace/blog-rl/wandb/offline-run-20260816_114755-rl-v4-staged-120-683a1921-l40-20260816-controller-v1`.
  The run-scoped finalizer closed the sidecars and synced both records to
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/ingmc3tn` and
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v4-staged-120-683a1921-l40-20260816-controller-v1`.
- Artifact paths and hashes: raw checkpoints/trajectories remain only on the
  paid host; compact source and protocol evidence are public. Nothing was
  copied to the Mac.
- Interpretation: this is a clean prospective restart, not a continuation of
  or retrospective repair to the rejected eight-update run.
- Next action: preserve the compact update-10/update-20 evidence and retained
  four-policy checkpoints; diagnose the failed batch by policy, phase, and
  token regime. Decide whether the update-10 checkpoint merits a larger frozen
  evaluation before changing optimization or parity configuration. Do not
  resume the update-20 trajectory unchanged.
- Instance decommissioned: no.

### 2026-08-16 — meaningful progress pulse and minimal stability checks

- Status: validation completed; a new RL trajectory is running from logical
  update zero and no previous optimizer state was resumed.
- Verdict: the previous 16-game pulse is retired as a learning-progress
  measure. Its one independent unit per endpoint was useful only as an
  end-to-end smoke test and could not distinguish a trend from map noise.
- Hypothesis: a fixed, repeated development subset with multiple independent
  cases can show whether RL changes gameplay and causal message use, while
  mean-distribution stability checks protect the run without aborting on a
  single harmless tail token.
- Decision unlocked: a fresh run may start from the pinned SFT adapter only
  after the expanded update-zero pulse completes and passes its structural
  validation. No step-20 optimizer or model state will be resumed.
- Source commit: `07edbb90` on public branch `exp/swarm-arena-4b`.
- Progress subset: six legacy ordinary maps, six harder 18/20-node maps, and
  six two-world information-handoff bundles. Every case is played from both
  BLUE and RED against the immutable SFT opponent. Critical and matched-decoy
  cases each use normal and dropped messages. Update zero therefore executes
  192 complete games and stores 72 exact SFT baseline rows; later checkpoints
  generate 120 candidate games and reuse those content-verified baseline rows.
  Each core capability and communication endpoint has six independent units.
- Reported progress signals: candidate-minus-SFT return on legacy maps, hard
  maps, and handoff games; an equal-family overall gameplay delta; critical
  normal-minus-dropped message value; RL-specific message lift above the SFT
  initializer; critical-minus-decoy specificity; and protocol/grounding rates.
  This repeated subset is a development curve, not the frozen final claim set.
- Stability checks: finite tensors, exact sampled-token spans, legal constrained
  choices, adapter identity, four-policy roster, and disjoint optimizer state
  remain mandatory structural invariants. Online numerical admission now
  aborts only on widespread mean absolute log-probability error above `0.05`
  or mean mismatch-KL above `0.002`; p99, maximum, and tail-fraction metrics
  remain logged diagnostics and DPPO still masks ratio outliers. The admission
  plan permits at most one update of policy lag and uses the same two mean
  distribution bounds, but the current lightweight rescore worker only emits
  lag-zero batches; this run is therefore on-policy in practice rather than a
  claim of fully pipelined async RL. This deliberately removes the p99 `0.15`
  hard stop that rejected the previous run over one localized tail event.
- Validation: Linux lint passes and the combined Swarm Arena plus trainer-loss
  suite passes `122/122` tests. A three-replica serving probe bound the pinned
  SFT adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
  and produced valid constrained broadcast and action JSON. The full 192-game
  SFT-vs-SFT null pulse completed in 10 minutes 56 seconds. Overall gameplay
  was `+0.002962` with bootstrap interval `[-0.017927, +0.020076]`;
  RL-specific communication lift was `+0.012120` with interval
  `[-0.006170, +0.029683]`; and both legacy and hard capability intervals
  contained zero. Critical normal-minus-dropped message value was `+0.071094`
  for the candidate arm and `+0.058974` for the SFT arm, while the matched
  decoy interval contained zero. Action protocol, broadcast protocol, and
  grounded-broadcast rates were all `1.0`. A handoff-only capability
  diagnostic showed a small `+0.016431` serving-path null offset, so that raw
  endpoint is interpreted only as change from the run's own update-zero point;
  it is not an absolute learning claim.
- Failures and retries: the first simultaneous three-server startup exposed a
  shared Triton/AOT cache race: two replicas attempted to import temporary
  compiled objects removed by another process. The two failed replicas were
  restarted with isolated vLLM, Triton, and TorchInductor cache directories;
  all three then passed health and adapter-binding checks. The first detached
  eval wrapper also exited before any request because `/usr/bin/time` is absent
  from the image; the untouched eval was relaunched without that optional
  wrapper.
- Fresh launch: run
  `/workspace/runs/rl-v4-staged-120-07edbb90-l40-progress-v2-20260816`
  started from the pinned SFT adapter at source `07edbb90`, LR `7.5e-6`, with
  four independent LoRA policy slots. Runtime certificate SHA-256 is
  `d2cc16333380136da760961ce77fb6597a4aaa69eaa79c0332f951dbe1dbdff4`;
  its mean absolute serving/trainer log-probability mismatch was `0.0021003`,
  mean mismatch-KL was `0.000108122`, and the single-slot optimizer isolation
  probe passed. Production plan SHA-256 is
  `ae69f834b5dc7e5ff06465be49fc93b42dc131af574a126ec48aa35bf52986c7`.
  The run has a fixed 192-game update-zero pulse and 120-new-game cached pulses
  every ten updates, run-scoped health logging, completion-only sidecar cleanup,
  and automatic sync of the exact trainer/controller W&B offline runs.
- Next action: require the fresh update-zero pulse, first complete optimizer
  update, and update-10 learning pulse; use the new curves rather than the
  retired one-unit pulse to make curriculum/LR decisions.
- Instance decommissioned: no.

### 2026-08-17 — live update-40 trajectory diagnosis

- Status: running; 41 complete logical updates were recorded when this audit
  finished, and the unchanged run was proceeding toward update 50.
- Verdict: mechanically stable with a negative development signal through
  update 40. This is not evidence of capability or communication improvement.
- Source/config: unchanged from the fresh certified staged restart at source
  `07edbb90`, production-plan SHA-256
  `ae69f834b5dc7e5ff06465be49fc93b42dc131af574a126ec48aa35bf52986c7`,
  four independent rank-preserving policy slots, and LR `7.5e-6`.
- Frozen development curve: relative to the run's update-zero point, overall
  gameplay changed by `-0.005334`, `+0.011319`, `-0.009312`, and `-0.029865`
  at updates 10, 20, 30, and 40. RL-specific communication changed by
  `-0.054583`, `-0.002504`, `+0.001197`, and `-0.005288`. Update-40 protocol,
  grounded-broadcast, and legal-action rates remained `1.0`; the regression is
  behavioral rather than a parser failure.
- Reward-density audit: across the four ten-update blocks through update 39,
  `90.0%`, `92.5%`, `95.0%`, and `95.0%` of four-replica groups had distinct
  terminal returns. Mean absolute leave-one-out advantage was `0.0754`,
  `0.0704`, `0.0754`, and `0.0822`. The reward is therefore neither mostly
  zero nor incapable of separating sampled joint trajectories.
- Communication/action audit: the designated sender included the privileged
  target fact in `100%` of critical training replicas in every completed
  block. The receiver targeted that node in roughly `58--65%` of replicas but
  captured it in only `14.6--25.0%`; there was no monotonic improvement.
  Intent/action agreement rose only from `47.0%` to `54.8%`. Action decisions
  retained exploration: only about `17--18%` of identical action slots emitted
  the same output in all four replicas. Broadcast decisions were more
  deterministic (`40--47%` identical), consistent with the already-saturated
  canonical sender report.
- Matched step-0/step-40 behavior: on the 120 candidate games, raw mean return
  changed from `-0.02633` to `-0.03373`. CAPTURE frequency rose from `13.23%`
  to `18.71%` while PROBE fell from `24.38%` to `19.02%`; WAIT remained flat
  near `38.3%`. Duplicate-target turns improved from `16.20%` to `13.43%`,
  intent following improved slightly from `69.01%` to `70.05%`, and message
  frequency barely changed. The policy is changing primarily toward more
  aggressive actions, not learning a stronger communication mechanism.
- Interpretation: this is not a corrupt-label or pure capability failure. The
  1.7B policy can sometimes exploit the handoff and produces useful wins, but
  the current estimator gives every independently changing agent the same team
  advantage. Because the critical sender fact is already constant across
  replicas, it supplies no within-group contrast for learning whether to send
  it; receiver-action variation is mixed with the other three agents' action
  variation. Exact terminal reward remains correct, but per-policy credit is
  noisy. The present trajectory is consequently learning an inconsistent
  capture bias rather than a general communication-dependent strategy.
- Next action: inspect the unchanged update-50 and update-60 development pulses
  before spending the remaining trajectory. If neither shows a reproducible
  upward shift, stop and preserve the run. The next prospective run should use
  matched, common-random-number counterfactuals that vary one sender message or
  receiver action at a time and assign the exact terminal-return difference to
  that policy span. This changes credit estimation, not the environment reward,
  and adds no hackable shaping term.
- W&B/artifacts/decommission: offline logging and automatic completion sync are
  still armed; the paid four-L40 host remains active.

### 2026-08-17 — stop shared-team run and design focused-credit restart

- Status: running from update zero.
- Verdict: the 120-update shared-team trajectory was intentionally stopped
  after 43 complete optimizer updates; it is preserved as rejected development
  evidence and will not be resumed or selected as a checkpoint.
- Stop state: trainer/controller/pulse/rescore/health processes for
  `/workspace/runs/rl-v4-staged-120-07edbb90-l40-progress-v2-20260816`
  were stopped cleanly. GPU 0 is free. The three healthy rollout servers on
  GPUs 1--3 remain resident for reuse. The trainer W&B offline run
  `us14yca6` is synced; controller sync remains to be verified.
- Root cause carried forward: terminal return was valid and dense, but all
  four independently changing BLUE policies received the same replica-level
  advantage. The estimator therefore could not localize a return difference
  to a particular action-taking policy. The observed approximately 55% metric
  was intent/action agreement, not task success, so it is not itself an RL
  difficulty target or readiness certificate.
- Prospective estimator: for each four-replica group, designate one trainable
  BLUE agent, vary only that agent's selected ACT sampling stream, and couple
  all other stochastic decisions with common random keys and identical prompt
  permutations. Compute the same verified terminal-control leave-one-out
  advantage, but route non-zero policy-gradient credit only to the designated
  agent. Curriculum cases designate the information receiver; ordinary cases
  rotate `blue-0` through `blue-3`. Non-designated policies retain zero
  advantage in that group so four-policy routing remains atomic.
- Reward decision: no direct speaking bonus, supervised message reward, or
  intermediate action bonus in the first focused run. Exact terminal control
  remains the only reward. This is deliberately stricter than arbitrary reward
  shaping and avoids creating a new incentive to spam messages or captures.
- Prospective schedule: 80 updates with four groups per update and unchanged
  frozen development pulses at updates 0, 10, ..., 80. The curriculum stages
  are 10 updates of focused tactical stabilization, 20 of handoff
  introduction, 30 communication-heavy, and 20 consolidation. Critical and
  matched-decoy counts remain equal within every update; the existing frozen
  OOD evaluation is unchanged.
- Added prospective artifacts:
  `data/rl_v4/staged_curriculum_v2_focused_80.json`,
  `configs/rl_v4_1_7b_focused_80.toml`, a focused-credit supervisor/rollout
  implementation, explicit non-production diagnostic span flags, and a
  parameterized staged launcher. These remain
  prospective until Linux tests, a fresh runtime certificate, and a live
  rollout smoke pass.
- GPU/cost: the four-L40 instance remains active at the user-reported rental
  rate; no new training GPU work had begun at this entry.
- Next action: run the focused-credit unit/integration tests on Linux, build an
  ACT-only immutable 80-update production plan and fresh runtime certificate,
  execute one rollout-only smoke, then launch from update zero only if those
  checks pass.
- Instance decommissioned: no.

#### Focused-credit validation and launch result

- Public training source: `a097bf17594bd5b01158687978b3848b6f94ba79`.
  Linux Ruff passed and the complete Swarm Arena suite passed `112/112`.
- Resolved trainer config SHA-256:
  `936a2ac52935987682cd523323390db6411a5d550cad8d5c1584d4355cca822a`.
  The fresh 32-sample certificate passed with mean absolute log-probability
  error `0.00215958`, p99 absolute error `0.0723795`, mean mismatch-KL
  `0.000108509`, and disjoint optimizer parameter sets; only `run_blue_0`
  changed in the single-policy isolation step. Runtime certificate SHA-256:
  `a7348c2decb36798f489bfbc7270fbafcfa28118b4ebecf8de4e990b13375ad8`.
- The ACT-only production plan passed preflight v2 and contains 320 groups over
  80 updates: 108 ordinary, 106 critical, and 106 matched decoy. Production
  plan SHA-256:
  `fe055740ed9844fd9655f70976140c02489bfd76b5d17dbab93b334a28cf6e5b`.
- A one-group diagnostic passed mechanically but had four zero returns. The
  predeclared four-group follow-up produced return contrast in all four groups.
  Every non-zero advantage was localized to the designated receiver
  (`blue-1` or `blue-2` in those cases), while every non-focused policy
  advantage was exactly zero. This validated the estimator before optimizer
  admission; it is not a learning result.
- Failure/retry: the first certificate-bind command supplied an incorrect
  manually expanded full commit hash and failed before writing a certificate;
  it was rerun with the exact `git rev-parse HEAD` value. A full pytest process
  was briefly scheduled alongside certification, contrary to the repository's
  process-cleanup warning; both jobs had already completed successfully when
  discovered. The workflow skill was updated and future test runs must occur
  before any `torchrun` process begins.
- Live run:
  `/workspace/runs/rl-v4-focused-80-a097bf17-l40-20260817`, tmux prefix
  `swarm-focused80-a097`. Trainer and controller started healthy; GPU 0 loaded
  the trainer and GPUs 1--3 remained healthy rollout servers. The unchanged
  update-zero pulse is the current barrier. Trainer W&B offline run ID is
  `gqv6yifx`; controller run ID is
  `rl-v4-focused-80-a097bf17-l40-20260817-controller-v1`. Both must be synced
  after completion.
- Early live status: the update-zero pulse completed and optimization proceeded.
  The original controller W&B sidecar exited on the first focused-credit record
  because it expected the legacy scalar `replica.advantage` field rather than
  the new per-policy `replica.advantages` mapping. Training, evaluation,
  admission, and trainer W&B remained healthy. The sidecar parser was updated
  to accept both schemas and to log focused-agent advantage density explicitly;
  it can replay every completed update and evaluation when restarted.
- The tested sidecar fix was published at source `c2464ef1` and copied into the
  immutable run's `sidecars/` directory without changing the training checkout.
  Controller W&B restarted successfully as
  `rl-v4-focused-80-a097bf17-l40-20260817-controller-v2` and replayed the
  existing progress/evaluation records; all five live run processes were
  healthy afterward.

#### Update-10 pulse and update-13 live checkpoint

- Status: 14 logical updates completed (`0` through `13`); all trainer,
  controller, rescore, pulse, and W&B-v2 processes remained healthy.
- Credit density: 46/56 groups had distinct replica returns and 179/224
  focused-agent replica opportunities had non-zero advantage. Mean training
  return over the changing scheduled scenarios was `0.04112`; this is a
  density/health statistic, not a learning curve because maps and opponents
  differ across updates.
- Frozen pulse delta from update 0 to update 10: overall gameplay improved by
  `+0.03043`; ordinary legacy by `+0.06427`; ordinary hard by `+0.01569`; and
  critical-handoff capability by `+0.01133`. The update-10 absolute overall
  RL-minus-SFT estimate was `+0.01004` with 95% interval
  `[-0.02184, +0.03955]`, so this is an encouraging direction, not a resolved
  improvement claim.
- Communication result: critical normal-minus-dropped fell from `+0.06750` at
  update 0 to `+0.04403` at update 10 (`-0.02346`); RL-specific communication
  lift likewise moved from `-0.01896` to `-0.04242`. Critical-minus-decoy
  specificity was nearly unchanged (`+0.00154` delta). The first ten updates
  were the predeclared tactical-stabilization stage, so the current evidence is
  "tactical improvement, no communication improvement yet." The run continues
  unchanged into the handoff-introduction stage; update 20 is the next
  informative communication comparison.

### 2026-08-17 — focused run update-16 parity stop and atomic-update diagnosis

- Status: stopped fail-closed after 16 complete logical updates (`0` through
  `15`); no update-20 pulse exists.
- Verdict: valid exploratory update-10 tactical signal; communication
  improvement rejected at that checkpoint; the run is not resumable as one
  coherent four-policy optimizer trajectory after the failed update.
- Hypothesis and decision: determine whether the update-16 failure was a
  corrupt rollout, stale actor policy, durable serving/trainer divergence, or
  transient long-lived-runtime drift before changing the next prospective run.
- Source and immutable inputs: training source
  `a097bf17594bd5b01158687978b3848b6f94ba79`; production-plan SHA-256
  `fe055740ed9844fd9655f70976140c02489bfd76b5d17dbab93b334a28cf6e5b`;
  base revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; initial SFT
  revision `534522a8f3ff3489b1dd8318dc8e533e51264cde` and adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.
- Failure: Prime run index 2, mapped exactly to `blue-3`, failed its complete
  update-16 ACT batch before its optimizer step because mean mismatch-KL was
  `0.0044082049 > 0.002`. Runs 0, 3, and 1 (`blue-0`, `blue-2`, and `blue-1`)
  had already passed and written partial step-17 broadcasts. `blue-3` remained
  at step 16. The controller correctly did not publish a partial logical
  update, but the trainer had already changed three of four policies. Those
  step-17 weights are rejected and will never seed another run.
- Exact failed-batch replay: the immutable `blue-3` step-16 file contained 60
  ACT samples, 480 trainable completion tokens, 360 branching tokens, and no
  broadcast span. Against the coherent `blue-3` step-16 adapter SHA-256
  `5fa8b102a15658ef198b058d0764b87c0765abc96c0b9b5990093d24a9a7023d`,
  two clean process replays were identical and passed the same mean gates:
  mean absolute log-probability error `0.01577488`, p99 `0.31182814`, mean
  mismatch-KL `0.00192902`, max mismatch-KL `0.11828327`, max probability
  error `0.09690857`, p99 probability error `0.05427437`, and 1.25% of tokens
  above probability error 0.05. Replaying against step 15 gave mean
  mismatch-KL `0.00160602`; this does not support a gross stale-policy error.
- Interpretation: the batch and checkpoint are not corrupt, and the exact
  clean replay is deterministic. The live 0.004408 value is a transient
  long-lived-runtime/backend deviation on a vLLM-to-HF constrained action
  batch. It is small enough for bounded off-policy DPPO but above the old
  brittle 0.002 online mean ceiling. This is an infrastructure boundary, not a
  reward, curriculum, or model-collapse result.
- Prospective fix: add opt-in `atomic_multi_run_updates` so all active policy
  batches accumulate, all four complete parity summaries pass, and only then
  do any of the four optimizers/schedulers step or publish weights. For a new
  run only, raise the mean mismatch-KL limit to `0.005` while retaining the
  `0.05` mean absolute log-probability bound, DPPO probability-direction mask,
  exact token/constraint ownership, lag-zero rescore, and all replay/safety
  invariants. This is not a retrospective pass for the rejected update.
- Evaluation result: from update 0 to update 10, overall gameplay improved
  `+0.03043`, legacy `+0.06427`, hard `+0.01569`, and handoff capability
  `+0.01133`. Absolute update-10 overall RL-minus-SFT was `+0.01004`, interval
  `[-0.02184,+0.03955]`. Critical normal-minus-dropped fell `-0.02346`, and
  RL-specific communication lift moved from `-0.01896` to `-0.04242`.
- Preservation: trainer W&B run
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/gqv6yifx` and controller
  runs `rl-v4-focused-80-a097bf17-l40-20260817-controller-v1` and `-v2` were
  synced. The four update-10 adapters and compact provenance/evaluation files
  were uploaded publicly and anonymously checksum-verified at
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-focused-step10-development`,
  revision `a191af71bf622b7a1e4f6bdd47c76d9bf27c838f`. The release is explicitly
  `not-admitted`; no model artifact was copied to the Mac.
- Failures and retries: the first publisher command ran from `/root` and failed
  before upload because the relative script path did not exist. The unchanged
  publication succeeded from `/workspace/blog-rl`. The first controller-v2
  W&B sync attempt occurred while its sidecar still held the offline file open;
  after orderly sidecar shutdown, the sync completed.
- GPU and storage: after preservation, controller, pulse, rescore, W&B, and all
  three inference sessions were stopped. All four L40 GPUs reported 0 MiB.
  Provider price and exact allocation start were not recorded, so no dollar
  cost is invented. Raw run data remains only on the paid host; compact public
  artifacts are durable.
- Next action: Linux-test the atomic trainer path, run a deliberately uneven
  four-policy integration canary that proves no optimizer changes until all
  four are ready, recertify the exact new config, and launch a fresh run from
  the common SFT initializer. Preserve the existing development subset and
  evaluate every ten updates; do not open selection/frozen tiers during the
  curve.
- Instance decommissioned: no; GPUs are idle and the host remains available for
  validation and the prospective fresh launch.

### 2026-08-17 — atomic four-policy canary and fresh focused-80 restart

- Status: atomic canary and update-zero pulse completed; fresh 80-update run is
  running with one complete optimizer update.
- Verdict: the atomic-update implementation passed its deliberately uneven
  live integration test. This is a mechanical safety result, not evidence of
  learning. The new RL run is not yet evaluated.
- Hypothesis and decision: eliminate the partial-joint-policy failure mode,
  retain bounded off-policy DPPO, and rerun the unchanged focused curriculum
  from the common SFT initializer. Do not seed from the rejected partial step
  17 or the non-admitted update-10 development adapters.
- Source commit: `6c5eea739e6d743957a6132dbb0022ea7eab4895` on public branch
  `exp/swarm-arena-4b`. Linux Ruff passed, focused tests passed `33/33`, and the
  complete Swarm plus multi-run selection passed `129/129` before any live
  `torchrun` process started.
- Atomic canary: `/workspace/runs/atomic-canary-6c5eea73`, resolved trainer
  config SHA-256
  `f8b3c78f38e3aeaa25843bafd750bf534b9e21a37b57de066d73eb5e46cd47bf`
  and trainer parity-gate SHA-256
  `671c4631021adcb098941079bca037e4133ca8f34eaef95ef3303ef6b8e0d0b8`.
  Only one recorded SFT batch was initially exposed. At `23:36:07`, one full
  policy was ready while the other three were absent; the trainer logged
  `ready=[2]`, kept learning rate exactly zero, and published no step-1
  broadcast. After the remaining batches arrived, all four parity summaries
  passed at `23:38:36`; the first non-zero optimizer step occurred only at
  `23:38:38`, followed by four step-1 stable broadcasts. There were zero
  trainer errors. The four output adapter SHA-256 values were distinct:
  `adaf1c97...`, `87c94445...`, `df8250a5...`, and `b86fb4e0...` for
  `run_blue_0` through `run_blue_3`. W&B run `x4v19blb` was synced.
- Fresh immutable inputs: base revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; SFT revision
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`; SFT adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.
  Prepared trainer config SHA-256
  `af13b9466c8f120fa81d345dde52748c12c2838e9875ae5c012696c73f02db01`.
- Runtime certificate: three isolated Prime inference servers on GPUs 1--3
  used unique vLLM, Triton, TorchInductor, RPC, and API namespaces. The fresh
  32-decision probe contained 1,367 completion tokens. Numerical certification
  passed with mean absolute log-probability error `0.00185143`, p99
  `0.04923738`, mean mismatch-KL `0.00008623`, and four disjoint optimizer
  parameter sets; only `run_blue_0` changed in the isolation step. The exact
  prospective bounds remain `0.05` mean absolute log-probability error and
  `0.005` mean mismatch-KL. Runtime certificate SHA-256:
  `44bad49c2c0e9fcdf6e07d5de8b0615236cf25a6ab03867558b39f9544afad86`.
- Production plan: ACT-only focused-agent credit, 80 updates, four groups per
  update, 108 ordinary groups, 106 critical groups, and 106 paired decoys.
  Schedule SHA-256
  `a1974e645bf5546a3d940a4fa85a5e218a7ceba351f4013666597660daff2e7b`;
  production-plan SHA-256
  `b1aea6748ae192953a818bf34d3a6cb21b4010890ba098857659eb780c8b8d3f`.
  The unchanged development pulse runs at update 0 and every ten updates;
  selection and final tiers remain unopened during the learning curve.
- Live run:
  `/workspace/runs/rl-v4-focused80-atomic-6c5eea73-l40-20260817`, tmux prefix
  `swarm-focused80-atomic-6c5`, launched at `23:50` on four NVIDIA L40 GPUs.
  Preflight v2 passed with GPU 0 idle and all three rollout servers healthy.
  Trainer, controller, rescore worker, pulse worker, and W&B sidecar survived
  launcher health checks. Controller W&B run ID is
  `rl-v4-focused80-atomic-6c5eea73-l40-20260817-controller-v1`; trainer W&B is
  offline run `220pn93o`. Both must be synced after completion.
- Initial live verification: the update-zero pulse completed all 192 rows and
  all four candidate adapter hashes exactly equalled the pinned SFT adapter
  hash before the barrier released training. Candidate-minus-SFT overall return
  was `-0.009648` with interval `[-0.026706,+0.005923]`; protocol and grounding
  were all `1.0`. The absolute same-weight RL-specific communication-lift
  estimate was nevertheless `+0.018417` with interval
  `[+0.000790,+0.037868]`, while critical-minus-decoy specificity did not pass
  and the evaluator correctly set `communication_claim_passed=false`. This
  demonstrates why absolute pulse deltas cannot establish learning under
  independently sampled rollouts: subsequent interpretation must use change
  from this update-zero anchor and require critical-over-decoy specificity.
- First live optimizer update: four groups and 16 replicas produced return
  contrast in all four groups, mean scheduled return `0.024446`, and 16 non-zero
  focused-agent advantages out of 64 policy slots. Every non-designated policy
  advantage remained exactly zero. All four complete parity summaries passed
  before the optimizer step; the worst run-level mean absolute log-probability
  error was `0.007460` and worst mean mismatch-KL was `0.000700`. Only then did
  one optimizer step occur and all four distinct step-1 adapters publish
  together. All eight long-lived sessions remained healthy with zero logged
  errors.
- Failures and retries: the first prepare command omitted the experiment-local
  package and failed before creating a run directory with
  `ModuleNotFoundError: swarm_ctf_eval`; the retry used
  `uv run --with ./experiments/swarm_arena`. The first certificate invocation
  used the CLI's diagnostic default `0.0005` instead of the immutable trainer's
  declared `0.005` threshold and failed before model loading or output. Its
  work directory was preserved as `parity-work.failed-default-threshold`; the
  retry explicitly supplied all trainer gate values and passed. Neither failure
  admitted data or changed policy weights.
- GPU and storage: no model or rollout artifact was copied to the Mac. Large
  checkpoints and raw evidence remain on the paid host; compact results will
  be committed and selected checkpoints published publicly. Provider price and
  exact allocation start are not recorded here, so no dollar cost is invented.
- Next action: complete the update-zero invariance pulse, then train. Treat
  update 10 as an early direction check and update 20 as the first substantive
  tactical-plus-communication comparison. Continue to update 80 unless a real
  process failure, non-finite optimization, extreme KL, or clear collapse is
  observed; do not stop merely because an early confidence interval overlaps
  zero.
- Instance decommissioned: no; the substantive run is active.

#### Update-40 live checkpoint

- Status: 40 logical optimizer updates completed (`0` through `39`); frozen
  development pulses at updates 0, 10, 20, 30, and 40 all completed and
  released their barriers. Trainer, controller, rescore, pulse, W&B, and all
  three inference sessions remained healthy with zero logged failures.
- Capability curve: overall candidate-minus-SFT return was `-0.009648` at
  update 0, `-0.035842` at 10, `-0.034174` at 20, `+0.005944` at 30, and
  `+0.022621` at 40. Update 40 therefore improved `+0.032270` from its exact
  update-zero anchor, although its absolute 95% interval
  `[-0.030138,+0.073127]` still includes zero. From update 0 to 40, legacy,
  hard, and critical-handoff deltas improved by approximately `+0.048296`,
  `+0.025338`, and `+0.023175`, respectively. The delayed reversal after the
  update-10/20 dip is evidence against selecting runs from very early pulses.
- Communication curve: critical normal-minus-dropped moved from `+0.058958`
  at update 0 to `+0.035081` at update 40, and RL-specific communication lift
  moved from `+0.018417` to `-0.005460`; both are a `-0.023877` change from the
  same-weight anchor. Critical-minus-decoy specificity changed only
  `-0.003937`, from `+0.022899` to `+0.018961`, and its update-40 interval
  `[-0.001014,+0.038937]` crosses zero. The evaluator correctly retains
  `communication_claim_passed=false`. Current evidence is capability learning,
  not communication learning.
- Stability and credit: action protocol, broadcast protocol, and grounded
  broadcast rates remained exactly `1.0` at every pulse. Across successive
  ten-update training windows, 30, 33, 31, and 32 of 40 groups had replica
  return contrast; non-zero focused-agent advantages were 117, 132, 123, and
  128 of 640 policy-replica slots. Non-designated slots remained zero by
  construction. Recent run-level parity means remained below the prospective
  bounds; no partial joint-policy update, non-finite value, or process failure
  occurred.
- Decision: continue unchanged. Updates 40--59 are the remaining
  communication-heavy stage and updates 50/60 are the decisive tests for
  whether the emerging capability gain becomes message-specific. Do not call
  the update-40 result swarm cooperation, and do not stop solely because the
  communication metric has not yet turned.
- Subsequent runtime decision: the user capped this exploratory run at the
  update-60 checkpoint to avoid spending through the originally declared 80
  updates. A one-shot `swarm-focused80-atomic-6c5-stop60` watcher waits for the
  content-bound `step_60.ready.json`, then stops only the controller while the
  independent pulse worker finishes the complete update-60 evaluation. This is
  a prospective stop made at update 41, before seeing update-50 or update-60
  results. The original 80-update production plan remains immutable; the
  truncated run must not be reported as completing that full horizon.
- Instance decommissioned: no; the live run is active on the four-L40 host.

### 2026-08-17 — focused atomic RL capped at update 60

- Status: completed at the prospectively requested cap. The one-shot watcher
  observed the content-bound `step_60.ready.json` at
  `2026-08-17T03:51:12Z`, stopped the controller before update 61, and allowed
  the complete update-60 frozen pulse to finish. The original immutable plan
  still declares 80 updates; this result must be described as truncated at 60,
  not as completing the planned horizon.
- Verdict: exploratory capability signal; not admitted for communication.
- Source and pins: commit
  `6c5eea739e6d743957a6132dbb0022ea7eab4895`; base
  `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
  initializer
  `CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible@534522a8f3ff3489b1dd8318dc8e533e51264cde`.
- Immutable evidence hashes: production plan
  `d67718f2685006916e43de32353f2877c3f7dff0bcbeed799454295a9a09fb7b`;
  plan audit
  `797358d2837131086c80b1f16db5baeb5920823e1b10addea4c1e734ec8f3453`;
  runtime certificate
  `6c9cde513beee986bf427e59f1e62f25096c5deafc0c286c320dac448b1e53be`.
- Capability result: overall RL-minus-SFT moved from `-0.009648` at the
  exact same-weight update-zero anchor to `+0.019663` at update 60, a
  `+0.029311` change. Legacy changed `+0.081167`, hard `-0.009878`, and
  critical-handoff capability `+0.016645`. Update 50 had the highest observed
  overall development mean, `+0.025233` with 95% interval
  `[-0.022926,+0.080502]`; it is retained retrospectively as a capability-only
  development checkpoint, not a confirmatory result.
- Communication result: RL-specific message lift moved from `+0.018417` at
  update zero to `-0.045288` at update 60, a `-0.063705` change.
  Critical-minus-decoy specificity moved from `+0.022899` to `-0.015273`, a
  `-0.038171` change. The communication claim failed at all seven pulses. The
  apparent positive update-zero lift came from stochastic sampling between
  same-weight policies and is only a baseline anchor; only change from it is
  interpretable.
- Optimization and stability: updates 40--59 had mean scheduled return
  `0.032386`; 64 of 80 rollout groups had return contrast and 255 of 1280
  policy-replica slots received nonzero focused advantages. Action protocol,
  broadcast protocol, and grounded-broadcast rates remained `1.0` at every
  pulse. No partial four-policy update, non-finite value, collapse flag, or
  process error occurred.
- Public preservation: the four update-50 adapters and compact reports are at
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-focused-step50-development`
  revision `049e95062903501a8a50efac09d1b2caab393364`. The four update-60
  adapters and reports are at
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-focused-step60-truncated-development`
  revision `a64eb9278f88cd1a31528be1460e22a7fd311945`. Both repositories are
  public and were anonymously downloaded and checksum-verified after upload.
  Trainer W&B run: `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/220pn93o`.
  Controller/evaluation W&B run:
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v4-focused80-atomic-6c5eea73-l40-20260817-controller-v1`.
- Storage and cost: no checkpoint or raw rollout was copied to the Mac. Compact
  evidence is in `results/rl_v4_focused_atomic_60/`. Provider price and exact
  allocation start were not recorded, so no dollar cost is invented.
- Interpretation: the reward/curriculum can teach game tactics, but this run
  did not teach agents to rely on private teammate information. A subsequent
  run should change the training distribution or objective so successful play
  actually requires message-conditioned action; merely training longer on this
  mixture is not supported by these results.
- Instance decommissioned: all trainer, controller, evaluator, rescore,
  inference, and W&B processes were stopped; all four L40 GPUs reported
  `0 MiB` and `0%` utilization. The instance is safe to decommission.

### 2026-08-17 — joint sender-receiver curriculum implementation

- Status: CPU implementation and audit completed; no GPU run started.
- Verdict: implementation ready for Linux validation, not an RL result.
- Triggering evidence: the focused update-60 run improved overall gameplay by
  `+0.029311` from update zero but changed RL-specific communication lift by
  `-0.063705`. Inspection of the immutable training contract found that every
  handoff focused the receiver and `SharedReturnSpec` rejected every focused
  phase except `ACT`. The optimizer therefore had no direct sender-broadcast
  credit path. Shared LoRA weights also allowed action-only gradients to alter
  broadcast behavior indirectly. This is the concrete design defect addressed
  here; the game reward itself was not changed.
- Implementation: `ScenarioAssignment` now binds each handoff pair to either a
  `sender` or `receiver` focus role. Sender groups vary only the designated
  sender's turn-zero `BROADCAST`; receiver groups vary only the receiver's
  turn-zero `ACT`; ordinary groups retain rotating focused `ACT` updates. The
  supervisor permits exactly one causal phase for focused credit, hashes that
  per-group spec into the run lock, and continues routing four distinct policy
  envelopes atomically. The only scalar remains verified terminal control
  delta with a leave-one-out replica baseline.
- Curriculum: `staged_curriculum_v3_joint_80.json` declares 80 updates / 320
  groups: 60 ordinary, 130 critical, and 130 matched decoy. Critical focus is
  exactly 65 sender and 65 receiver groups; every critical pair has a decoy
  with the same role. Communication-heavy groups occupy the middle stage, and
  ordinary games return during consolidation. The next run starts from the
  pinned SFT initializer rather than the tactically stronger but communication-
  degraded update-50 adapters.
- Evaluation: the frozen 192-game ten-update pulse keeps the same cases and
  causal return endpoints. It now additionally records active-target fact
  coverage by the certified sender, normal-minus-dropped critical capture
  rate, and the RL-minus-SFT change in capture dependence. W&B separates
  focused `BROADCAST` and `ACT` advantage density. These are diagnostics, not
  shaped rewards or new success shortcuts; the selection/final OOD data remain
  unchanged and unopened.
- Validation: `py_compile` passed for all changed modules and tests; Ruff
  passed; 17 curriculum/evaluation tests passed in `0.51s`; direct validation
  confirmed that sender-`BROADCAST` and receiver-`ACT` focused specs are valid
  and hash-distinct. The full shared-return rollout test imports the Linux GPU
  dependency stack (`torch`, xgrammar, transformers, vLLM) and is intentionally
  deferred to the fresh host rather than downloading that stack to the Mac.
- Local tooling note: `/opt/homebrew/bin/uv` was version `0.9.2` and rejected
  the repository's declared `uv>=0.11.1` plus relative `exclude-newer`. The
  repository-compliant `/Users/chinmay/.local/bin/uv` version `0.12.5` was used
  instead. Its isolated test environment initially lacked `pytest`, `httpx`,
  and then the GPU-only `torch` import; lightweight dependencies were installed
  in `/private/tmp`, and no model/checkpoint artifact was stored on the Mac.
- GPU time and cost: none; CPU-only preparation.
- Next action: commit/push, then provision four L40-class GPUs. On Linux, run
  the complete focused-return and controller tests, build and audit a fresh
  immutable production plan/runtime certificate, execute update-zero, and only
  then launch the declared 80 updates with ten-update pulses.
- Instance decommissioned: previous four-L40 host was safe to decommission and
  is not required for this implementation.

### 2026-08-17 — training-only pass@k curriculum screen

- Status: completed; compact results preserved locally and GPU stopped.
- Verdict: receiver/tactical RL has a useful stochastic learning band; sender
  learning and terminal-return communication specificity are rejected for this
  slice.
- Verdict scope: training-data selection diagnostic only. It cannot promote a
  checkpoint or establish communication on development/frozen evaluation.
- Hypothesis: stochastic pass@k separates handoff cases that are already
  solved, impossible for the receiver, communication-unnecessary, or useful
  sender-learning examples more reliably than choosing a curriculum band from
  single greedy rollouts.
- Data: exactly 12 role-balanced bundles from the existing
  `handoff_train.json`; both latent worlds and both critical/matched-decoy
  scenarios. Development and frozen OOD manifests remain unopened.
- Models: the pinned 1.7B SFT initializer plays BLUE against a model-controlled
  copy of itself. BLUE and RED each make four independent requests per phase.
- Sampling: temperature 0.7 with request seeds shared across corresponding
  generated, dropped, and reference-message repetitions. Critical generated
  play uses K=8; critical controls and all decoy conditions use K=4. Total:
  672 complete model-vs-model games.
- Reference condition: the certified sender's grounded active-target fact is
  required at turn zero while preserving its generated intent, resource
  request, and other facts up to the existing three-fact budget. Every action
  and every later decision remains model-generated. This distinguishes sender
  omission from receiver/game incapability without supervising an action.
- Measurements: pass@1/2/4/8 target capture, turn-zero capture, expected
  best-return@k, return contrast@4, target-fact emission, receiver target
  action, protocol validity, generated-minus-dropped capture, reference-minus-
  generated sender gap, and reference-minus-dropped communication headroom.
- GPU: one NVIDIA L40S 46,068 MiB on a fresh user-provided host; 554 GB free at
  inspection and no stale process. Provider rate is unknown, so no cost is
  inferred. Large trajectories/checkpoints will not be copied to the Mac.
- Source: rollout commit `69fe74ff5ca6954310dd22848de77ba5293cb538`;
  pinned base revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
  pinned SFT revision `534522a8f3ff3489b1dd8318dc8e533e51264cde`;
  adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`.
- Validation and throughput: the complete Linux suite passed 114/114 before
  inference; Ruff and the reference-merge focused assertion passed. The
  corrected run completed 672 games, 26,880 independent agent requests, and
  1,008,519 completion tokens in 1,952.76 seconds. Protocol validity was 1.0.
- Pass@k result: critical target capture was 0.1510 at pass@1, 0.4161 at
  pass@4, and 0.5833 at pass@8. Critical return contrast@4 was 0.7655. Of 24
  critical worlds, the exploratory selector found 12 primary receiver-band,
  three hard-reserve, one easy-stabilizer, and eight low-signal worlds.
- Sender result: generated target-fact coverage was exactly 1.0. The reference
  insertion was therefore semantically an identity intervention, not a source
  of missing information. These pairs should not receive sender-BROADCAST
  updates; later training pairs require a cheap sender-only omission screen if
  sender learning remains in scope.
- Paired causal result, bootstrapped over 12 complete bundles: critical
  generated-minus-dropped terminal return was `+0.00696`, 95% interval
  `[-0.01190,+0.02428]`; critical-minus-decoy terminal specificity was
  `-0.00192`, interval `[-0.02354,+0.01652]`. Turn-zero capture specificity was
  directionally `+0.07292`, interval `[0,+0.16667]`, but only four of 12 bundle
  effects were positive. The immediate hint does not survive into terminal
  reward and is not an admitted communication result.
- Interpretation: terminal shared return is dense enough for tactical RL, but
  the current turn-zero handoff is washed out over later turns. The next
  communication curriculum should start with handoffs near the terminal
  boundary (or a short-horizon stage), emphasize receiver ACT learning, retain
  the matched decoy for every selected critical world, and only later extend
  to 4/5-turn play. This changes the curriculum rather than the game, reward,
  or frozen evaluation.
- Compact artifact hashes: manifest `a4b2c896...ae0`; rows
  `c7e4268f...d36`; summary `780a3f5e...b1b`; analysis
  `874dec87...129`. The complete directory is 552 KB under
  `results/rl_v4_passk_screen_1_7b/`; no model artifact or raw trajectory was
  copied to the Mac.
- Final public validation: exact commit `f2691ec1` passed Ruff and the complete
  Linux Swarm suite, **115 tests** with only the two known third-party SWIG
  deprecation warnings. The remote checkout was clean and the L40S remained at
  0 MiB / 0% utilization.
- Next action: revise the joint curriculum from sender/receiver 50/50 to a
  terminal-proximal receiver-first stage, add a cheap sender-omission screen
  for later pairs, then launch a small RL pilot before a long run.
- Instance decommissioned: no; inference is stopped and the L40S reports 0 MiB
  and 0% utilization. The host is safe to decommission after Git publication.
- Rejected engineering partial: the first reference implementation replaced
  the sender's entire generated message with a one-fact message. Inspection of
  the first completed bundle showed this could delete a useful intent or
  unrelated grounded facts, so reference-minus-generated would not isolate the
  missing target fact. The run was stopped after 96/672 games; its compact rows
  remain on the host under `sweep.rejected-replace-reference` and will never be
  combined with the corrected result. No model weights were changed. The
  corrected prospective run inserts the target fact only when absent and
  otherwise leaves the generated message unchanged.

### 2026-08-17 — receiver-first terminal-proximal curriculum screen

- Status: planned; implementation and live training-split screen in progress.
- Verdict scope: curriculum-selection evidence only; no optimizer, development,
  selection, or frozen OOD evaluation is authorized by this screen.
- Hypothesis: the pass@k-selected receiver worlds contain useful action
  variation at turn zero, but their message effect is washed out by later
  transitions. Ending the handoff after two turns should preserve more of that
  information-specific effect in verified terminal return without changing the
  reward or mechanically unlocking an action.
- Proposed curriculum: 40 updates, four groups per update, receiver `ACT`
  focus only, 40 ordinary / 60 critical / 60 matched-decoy groups. The exact
  12 primary pass@k worlds are bound by pair and latent-world label. Handoff
  horizons retain two remaining turns for updates 0--19 and use the original
  scenario horizon for updates 20--39. Sender `BROADCAST` updates are excluded
  because target-fact emission was 100% on this screened slice.
- Reward and model: unchanged verified terminal control delta, no shaping;
  pinned Qwen3-1.7B SFT initializer against a model-controlled copy of itself.
- Live screen: rerun the same 12 selected worlds, their matched decoys,
  generated/dropped communication, and four common-seed repetitions with one
  and two remaining turns (192 complete 4v4 games per horizon). Compare with the same repetitions from the
  already-completed original-horizon screen, clustering uncertainty by handoff
  pair rather than treating games or agents as independent.
- Decision rule: adopt the terminal-proximal first stage only if protocol stays
  exact, terminal critical-minus-decoy communication specificity improves over
  the original horizon, and capture/action behavior remains non-degenerate.
  A negative screen changes the curriculum before any RL compute; it does not
  trigger reward shaping or access to frozen evaluation.
- GPU: one user-provided NVIDIA L40S 46 GB host. Provider price is unknown, so
  cost will not be invented. Models and raw rollouts remain remote; only compact
  rows, summaries, hashes, code, and documentation may return to the Mac.
- Instance decommissioned: no; the user is keeping the host active for this
  bounded screen.
- First launch failure: the draft encoded absolute `horizon=2`, but handoff
  states already begin at simulator turn two. The environment rejected the
  state before any model request. The contract was renamed to explicit
  `handoff_remaining_turns`, the empty failed directory was preserved, and no
  result from that invocation will be scored.
- Results: both corrected screens completed with 192/192 games and exact
  protocol validity. One remaining turn took 202.44 seconds but reduced
  critical generated return-contrast to `0.50`, so it is rejected as too
  sparse for the first RL stage. Two remaining turns took 501.33 seconds and
  retained critical return contrast `0.8333`, receiver-action contrast
  `0.8333`, and nonzero-return cells `0.9167`.
- Communication diagnostic: at two remaining turns, critical-minus-decoy
  generated/dropped terminal-return specificity was `+0.01213`, clustered
  bootstrap interval `[-0.00410,+0.02962]`. The change from the original mixed
  horizon was `+0.03411`, interval `[-0.01696,+0.07902]`. This is modest,
  unresolved communication sensitivity—not a learned communication result—but
  it combines the correct sign with useful stochastic reward density.
- Decision: adopt two remaining turns for receiver warmup/density, then transfer
  directly to original scenario horizons. Do not use the one-turn stage. Keep
  sender updates excluded and keep every selected critical world paired with
  its exact matched decoy.

### 2026-08-17 — receiver-first terminal curriculum live launch

- Status: stopped atomically after completing updates 0--14. The trainer
  rejected the next four-policy update on its pre-optimizer mean mismatch-KL
  check; trainer, controller, and pulse processes then exited. Inference,
  rescore, and offline-W&B sidecars remain alive pending preservation.
- Verdict: useful but incomplete capability signal; no communication-learning
  result. Update 10 modestly improved gameplay, while both RL-specific
  communication lift and critical-minus-decoy specificity remained unresolved.
- Hypothesis: concentrating exact terminal control-delta credit on receiver
  `ACT` spans, first at two remaining turns and then at the original horizon,
  will improve information-handoff play without a communication bonus or
  supervised action target.
- Decision unlocked: if development pulses improve handoff and overall return
  while generated messages outperform dropped messages more in critical than
  matched-decoy cases, retain this initializer/curriculum for the next
  end-to-end communication experiment. If only generic return improves, treat
  the result as tactical capability learning. If neither improves, revise
  case diversity or credit localization rather than simply extending steps.
- Source commit: `4bf3fcb32a36ed7aaab2fde31ec2770469609946`.
- Base / adapter / opponent revisions: pinned Qwen3-1.7B base
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; public SFT step 320
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`, adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`;
  public historical RL-v1 `ad51ef261f3e7b7b2d3c6433106bd667ba1da81c`,
  blue-0 adapter SHA-256
  `1004e012cd96a6377006c334d997825e3ebb25828b482a4644b7149a823d873a`.
- Data: `data/rl_v4/staged_curriculum_v4_receiver_terminal_40.json`;
  40 logical updates and four groups per update: 40 ordinary, 60 critical,
  and 60 exactly paired decoys. Updates 0--19 retain two handoff turns;
  updates 20--39 restore the original handoff horizon, with ordinary games
  reintroduced for the final ten updates. The twelve selected pair/world
  cases are fixed from the training-only pass@k screen. Development pulses
  occur at updates 0, 10, 20, 30, and 40; selection and frozen OOD remain
  unopened.
- Reward and optimization: unchanged verified terminal control delta only;
  focused receiver `ACT` credit; four separate rank-16 LoRA policies and
  optimizers; constant learning rate `7.5e-6`; atomic four-policy updates;
  lag-zero bounded asynchronous admission.
- GPU, wall time, and estimated cost: fresh user-provided host
  `64.247.196.196:40301`, four NVIDIA L40S 46,068 MiB, initially idle. Price
  was not supplied for this allocation, so no dollar cost is inferred.
- Exact launcher/config: `scripts/launch_staged_rl.sh` with
  `configs/rl_v4_1_7b_receiver_terminal_40.toml`, expected updates 40,
  checkpoint/evaluation interval 10, curriculum artifact v4, and
  `SWARM_SHARED_RETURN_CREDIT_ASSIGNMENT=focused_agent`. Resolved trainer
  config SHA-256 is
  `d9ba25cc9de38f9a6bdd8012e559d362a45a5d2b44920ae9c2a9b9f83df424e3`;
  runtime certificate SHA-256 is
  `57ad8f3920bb7b833a6c25ad4c3341b6eb64f8b90eae0948090b14e1d12e24b0`;
  ACT-only production-plan SHA-256 is
  `93112d85c9fd03014de251dd61610dcc2199009c0327f4105f6bee45e2f221cc`;
  exact schedule SHA-256 is
  `ef7fea19c908cbc6de6ff410c285377cc96850513d5a67fd6198501b58658e05`.
- Predeclared measurements: real development return and handoff return;
  generated-minus-dropped effect in critical cases; critical-minus-decoy
  specificity; protocol validity; action/speaking/target collapse; policy KL;
  and opponent cross-play. Structural preflight, fresh runtime parity, and
  update-zero are integrity checks, not scientific outcomes.
- Validation: all 116 Linux Swarm tests passed before any live GPU process,
  with only the two known third-party SWIG deprecation warnings. Anonymous
  downloads reproduced the exact SFT and historical-adapter hashes. The fresh
  32-decision, three-server certificate passed with mean absolute log-probability
  error `0.00197840`, p99 `0.0633736`, mean mismatch KL `0.0000834760`, and
  maximum mismatch KL `0.0135564`; the configured mean gates passed, optimizer
  parameter sets were disjoint, and a test update changed only `run_blue_0`.
  Public-input preflight v2 passed with exactly 40 updates, 160 groups, four
  L40S GPUs, all three serving registries, and 2.30 TiB free disk.
- Results: update 10 completed the full 192-game development pulse with exact
  action, broadcast, and grounding protocol rates of `1.0`. Overall gameplay
  RL-minus-SFT was `+0.018045`, 95% interval
  `[-0.024934,+0.058953]`; legacy capability was `+0.048567`, hard capability
  `-0.007099`, and handoff capability `+0.012667`, all with intervals crossing
  zero. Critical normal-minus-dropped return was positive at `+0.058750`, but
  the SFT baseline already had `+0.058819`, leaving RL-specific communication
  lift `-0.000069`, interval `[-0.032760,+0.030823]`. Matched-decoy
  normal-minus-dropped was also `+0.056392`, so critical-minus-decoy
  specificity was only `+0.002358`, interval `[-0.026435,+0.030159]`.
  Therefore the ten-update checkpoint shows a tentative generic gameplay gain,
  not learned information-specific coordination.
- Failures and retries: the first certifier invocation used its narrower CLI
  defaults and rejected the already-declared trainer gate before model loading.
  Its directory is preserved. The unchanged probe was rerun with the exact
  config-bound `0.05` mean log-probability and `0.005` mean mismatch-KL values;
  no threshold changed and that binding passed. A repository-wide Ruff check
  reported ten pre-existing import/unused-variable findings in untouched
  legacy files; the complete behavioral suite passed. After update 14, the
  next atomic update reached per-policy parity checks. Runs 0--2 passed, but
  one policy batch had mean mismatch KL `0.0053858049`, slightly above the
  declared `0.005`; Prime aborted before optimizer application. This was not
  an OOM, NaN, NCCL, simulator, protocol, or partial-update failure. The
  controller subsequently timed out waiting for all four rejected policy
  publications, and the pulse sidecar later timed out waiting for update 20.
- Artifacts: complete retained per-policy checkpoints and evaluation evidence
  exist at update 10. Fifteen progress entries cover updates 0--14. Broadcast
  adapters labelled step 15 exist, but update 15 was not atomically admitted
  and those files are invalid as a scientific checkpoint. The remote run is
  1.2 GB; no model/checkpoint was copied to the Mac.
- Next action: preserve/sync the update-10 compact evidence and W&B runs before
  decommissioning. Any continuation must be a newly certified run, either from
  the valid update-10 four-policy checkpoint or from the original initializer.
  Do not silently relabel step-15 broadcasts or restart the failed run in
  place. Decide separately whether to retain the `0.005` mean-KL bound and
  diagnose the outlier or predeclare a slightly wider bounded-off-policy
  envelope; no threshold may be altered retroactively for this evidence.
- Instance decommissioned: no.

### 2026-08-17 — pair-7 two-world communication overfit run

- Status: running on the live four-L40S allocation; no learning result is
  claimed before the scheduled development pulses complete.
- Question: can terminal-return RL teach even one small-model receiver to use
  a teammate's private fact? This is intentionally a learnability/overfit
  experiment, not a generalization or emergent-swarm claim.
- Curriculum: 60 updates of exactly two critical and two matched-decoy groups.
  Every update repeats both latent worlds of handoff pair 7 with two remaining
  turns. In the critical pair, blue-1 has the same private observation and the
  same legal actions in both worlds; blue-2's broadcast is the only input that
  identifies whether `V13` or `V19` is exposed. Training only one world would
  permit pair-identity memorization, so both worlds are mandatory in every
  update. The decoys retain world information in blue-1's own observation and
  expose generic tactical learning as a control.
- Credit and reward: train receiver `ACT` spans only with the unchanged,
  verified terminal control-delta return and common-random-number leave-one-out
  credit. There is no message, target-action, capture, or intermediate shaping
  reward. Sender target-fact emission was already 100% in the training-only
  screen, so this run does not update sender broadcasts.
- Initialization and optimization: restart from the pinned public SFT step-320
  adapter, not the invalid post-failure step-15 files and not the prior RL
  update-10 checkpoint. Retain four separate role-policy slots, rank-16 LoRA,
  constant LR `7.5e-6`, atomic updates, and model-controlled opponents. Only
  blue-1 is expected to receive nonzero focused credit in this deliberately
  single-pair experiment.
- Measurement: development pulses at updates 0, 10, 20, 30, 40, 50, and 60.
  The primary result is critical generated-minus-dropped receiver target-action
  and terminal-return lift; critical-minus-matched-decoy intervention lift is
  the information-specific control. Raw training return alone is not success.
  The frozen selection and OOD suites remain unopened.
- Runtime policy: retain the `0.05` mean absolute log-ratio check and predeclare
  a `0.01` mean mismatch-KL ceiling. The prior run's sole failure was
  `0.0053858` against `0.005`, before optimizer application, with no NaN, OOM,
  NCCL, protocol, or partial-update fault. The wider ceiling applies only to
  this new run and is not a reinterpretation of the rejected update. All other
  dynamic-constraint and atomicity checks remain intact.
- Inputs: curriculum
  `data/rl_v4/staged_curriculum_v5_communication_overfit_60.json` and trainer
  `configs/rl_v4_1_7b_communication_overfit_60.toml`. Exact source, plan,
  schedule, runtime-certificate, W&B, wall-time, and outcome identities will be
  appended after launch/completion.
- Live launch identity: source
  `42448837317e6dbe7896c8583b169b1fe1aa2703`; run
  `rl-v4-communication-overfit-60-42448837`; remote directory
  `/workspace/runs/rl-v4-communication-overfit-60-42448837-l40s-20260817`;
  production-plan SHA-256
  `7d66c6a4a69808ea56abf1e43790cdb0030f14610d6715e367f9a3a49bf63065`;
  schedule SHA-256
  `dcd05f0f6786688c05ff0ffd1664f43276e39956f517b17bc09e7de09584ddb3`;
  runtime-certificate SHA-256
  `171a81daf45c94d9dfe4449ed4d5f7ec6de03ba91ddea9dbd50a3f0997173a87`.
  The plan contains 60 updates, 120 critical groups, 120 matched decoys, and no
  ordinary groups.
- Validation and startup: 117 Linux tests passed in 40.31 seconds. The single
  32-decision runtime probe contained 1,367 completion tokens and passed with
  mean absolute log-probability error `0.00173572`, mean mismatch KL
  `0.0000663506`, and isolated four-policy optimizers. The three live endpoints
  then passed one structured broadcast and one six-choice legal action. A
  direct certifier invocation failed before model loading because PyTorch's
  distributed environment variables were absent; the unchanged command was
  rerun under one-process `torchrun` and passed. The GitHub HTML commit route
  briefly returned 404 while the public branch and exact GitHub API commit
  route returned 200, so immutable public-input preparation used the exact API
  commit URL. Neither startup retry changed data, reward, model, or thresholds.
- Logging and first health check: the trainer and controller use the W&B group
  `qwen3-1.7b-communication-overfit-60-42448837` in offline/failure-isolated
  mode for later sync. All five training sidecars and all three inference
  servers were live. The update-0 evaluator had begun writing its fixed
  192-game baseline (`raw.jsonl` and `rows.jsonl`) while the controller remained
  blocked at the pre-update barrier, as intended.

### 2026-08-18 — lost-node incident and unattended-run redesign

- Status: failed infrastructure run; no learning result and no recoverable
  checkpoint are claimed.
- Cost: the operator reports approximately `$50` lost. This was preventable:
  all durable state, the watcher, compact status, and offline W&B files lived on
  the same ephemeral pod. The monitoring process could detect a training fault,
  but it could not preserve evidence after loss of the host itself.
- Last confirmed observation: at `2026-08-17T16:38:02Z`, update 0 evaluation was
  active, all trainer/controller/evaluator/serving sessions existed, and GPUs
  1--3 were serving at 64%, 46%, and 74% utilization with 38,249 MiB allocated
  each. GPU 0 had 8,921 MiB allocated and 0% utilization while the controller
  waited at the evaluation barrier. Only the step-0 barrier and partial
  update-0 evaluation files were confirmed. No optimizer update, later
  checkpoint, W&B sync, HF upload, or clean completion was observed.
- Failure evidence: repeated SSH probes to `64.247.196.196:40301` timed out.
  This proves only endpoint loss; it cannot distinguish pod termination,
  endpoint reassignment, provider/network failure, or host failure. There was
  no confirmed in-run OOM, NCCL, parity, protocol, or optimizer failure before
  connectivity disappeared.
- Process failure: the launch contract incorrectly treated an on-pod watcher
  and offline logging as adequate unattended protection. It also blocked the
  optimizer at step zero on a 192-game development evaluation. Both choices
  increased exposure to host loss without producing a checkpoint.
- CPU-side correction: `run_live_artifact_mirror.py` now requires and
  anonymously verifies a public HF recovery repository before optimizer launch.
  It mirrors compact progress and eval records every five completed updates and
  uploads only the four complete LoRA adapters at every checkpoint barrier. A
  checkpoint is uploaded only after both `STABLE` markers exist and its four
  adapter hashes equal the controller's signed ready record; public downloads
  are re-hashed before the step is marked mirrored. Optimizer state and full
  model weights are deliberately excluded.
- Deadline correction: unattended launch now requires the provider's pod
  termination epoch. Forty-five minutes before it, the mirror performs and
  records an additional compact sync. Training may continue afterward, but a
  previously verified recovery checkpoint is already off-node.
- Telemetry correction: controller/evaluation W&B is online by default and
  remains failure-isolated from training. The Prime trainer itself stays
  offline so a W&B outage cannot kill an optimizer step. HF mirroring is an
  independent tmux process with retry-on-network-error behavior.
- Evaluation correction: the online barrier for this deliberate learnability
  run is reduced from the 192-game development suite to a 24-game matched
  pair-7 probe: two latent worlds, critical and decoy controls, normal/dropped/
  sender-shuffled messages, and two stochastic repetitions. It reports return,
  receiver target choice, sender target-fact use, protocol validity, and
  critical-minus-decoy specificity. The large held-out development evaluation
  is reserved for checkpoint selection; the frozen OOD tier remains unopened.
- Claim boundary: improvement on the pair-7 probe demonstrates training-pair
  message-conditioned learnability only. Generalization still requires the
  byte-identical development intervention suite, and neither result establishes
  broad emergent swarm intelligence.
- New unattended-run rule: a future GPU run is not described as safely launched
  until (1) the public HF heartbeat was anonymously downloaded, (2) the online
  W&B run is visible, and (3) the mirror session is healthy. At update 10, the
  public adapter hashes must also be verified before the run is left alone.

### 2026-08-19 — completed pair-7 overfit run and larger diagnosis

- Completed run: `rl-v4-pair7-overfit60-2243447c-5090-20260819`, source
  `2243447cc891e9432309fc79dfa99aed7a99038d`, reached all 60 atomic updates.
  Checkpoints 10--60 and compact pulse evidence were publicly mirrored and
  anonymously hash-verified. The update-60 ready-record SHA-256 is
  `d71fd405bc82838d9ead4bd9d555f5d1dfe0444726bdd8983e333cbeef7bc181`;
  the four adapter SHA-256 values are `c8268185...6764d`,
  `1594af86...25c8c`, `8dc5de28...50646`, and `86056aef...60a0f8`.
- Original pulse verdict: the 24-game update-50 pulse briefly reported normal
  return `0.09259`, normal-minus-dropped `+0.01852`, and receiver target choice
  `0.75` versus `0.50`. Update 60 returned to `0.05556`, `0`, and
  `0.50`/`0.50`. Specificity was negative at update 50 and zero at update 60.
- Exact row diagnosis: every world/condition cell in the pulse had only two
  repetitions, so critical normal contained four samples. The apparent
  update-50 gain was one right-world sample flipping. The same target behavior
  appeared in the matched decoy and shuffled-message controls. It was not
  information-specific communication learning.
- Larger replay: the SFT initializer, update 50, and update 60 were each run on
  192 matched games (16 repetitions, both pair-7 worlds, critical and decoy,
  normal/dropped/sender-shuffled). Their normal returns were `-0.009259`,
  `+0.039352`, and `+0.037037`, respectively. Normal-minus-dropped return was
  `-0.002315`, `-0.002315`, and `0`. Critical-minus-decoy specificity was
  `+0.002315`, `0`, and `-0.002315`. Every paired bootstrap 95% interval for a
  communication return effect or specificity crossed zero. The three sweeps
  consumed 788.5 seconds of measured evaluation wall time, about `$0.44` at
  the operator's `$2/hour` node rate.
- Training audit: all 960 retained blue-1 decisions were decoded. Critical
  correct-target action rate changed only from `48.75%` in updates 0--9 to
  `52.50%` in updates 50--59. Correct-target capture rose from `15.00%` to
  `33.75%`; the matched decoy rose almost identically from `17.50%` to
  `32.50%`. Across the full run, critical target actions averaged `+0.05563`
  terminal return versus `-0.03688` for other actions, so reward direction was
  sound. However, decoys received at least as much usable gradient: 127
  positive and 182 negative receiver advantages versus 117 and 165 on
  critical cases. The receiver retained a left/V13 prior instead of reliably
  switching from the sender fact.
- Scientific verdict: the optimizer, LoRA path, protocol constraints, and
  terminal reward were capable of learning generic tactics. The run did not
  learn useful or information-specific communication. Update 50 must not be
  selected as a communication checkpoint, and simply extending this
  curriculum is not justified.
- Next design: keep the unchanged terminal reward but expose more independent
  communication-critical contexts, shorten the receiver prompt, calculate a
  paired receiver advantage across matched latent worlds/conditions, and move
  most decoys out of the optimizer stream while retaining them as a causal
  evaluation control. Require a CPU batch audit showing critical-specific
  advantage density before renting GPUs again.
- Compact evidence: `results/rl_v4_pair7_overfit60_1_7b/`; diagnosis SHA-256
  `dfd77496d2b3661776f091cf3465537ec3f2f1197be8b3980675505b9ce748f0`.
  No model files or raw checkpoints were copied to the Mac. Core run artifacts
  remain public at `CK0607/swarm-arena-live-runs`. Instance decommission status:
  safe to decommission after this compact evidence is pushed.

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
- Training-only pass@k curriculum screen:
  `results/rl_v4_passk_screen_1_7b/`
- Receiver terminal-proximal curriculum screen:
  `results/rl_v4_terminal_proximal_screen_1_7b/`
- Completed pair-7 overfit diagnosis:
  `results/rl_v4_pair7_overfit60_1_7b/`
- RL v4 30-update development result:
  `results/rl_v4_1_7b_long/` and
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-long-development`
- RL v4 stronger-learning-rate development result:
  `results/rl_v4_1_7b_lr_ablation/` and
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-lr1e5-step8-development`
- Focused atomic RL run through the prospective update-60 cap:
  `results/rl_v4_focused_atomic_60/`,
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-focused-step50-development`,
  and
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-focused-step60-truncated-development`
- Joint sender-receiver RL launch contract:
  `JOINT_COMMUNICATION_RL_PLAN.md`,
  `data/rl_v4/staged_curriculum_v3_joint_80.json`, and
  `configs/rl_v4_1_7b_joint_80.toml`
- Compact multi-pair communication learnability contract:
  `COMMUNICATION_LEARNABILITY_V6_RUNBOOK.md`,
  `data/rl_v4/staged_curriculum_v6_compact_multipair_40.json`, and
  `configs/rl_v4_1_7b_compact_multipair_40.toml`
- Frozen message-credit admission plan:
  `MESSAGE_CREDIT_AUDIT_PLAN.md`
- Public, non-admitted mechanical RL artifact:
  `https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v1`

### 2026-08-19 — RTX 5090 communication-overfit backend-bound failure

- Status: failed at the declared runtime gate after 11 completed controller
  records; update 10 is the last accepted, publicly mirrored checkpoint.
- Runtime: four RTX 5090 GPUs, driver `595.45.04`; source
  `8fd1a918ab6831d3900cd7bd708b3d66bb49d5ca`. All 121 Linux tests passed.
  The fresh 32-decision probe passed with mean absolute log-probability error
  `0.00172195` and mean mismatch KL `0.000058302` under the original
  `0.05`/`0.01` limits.
- Startup failures before optimizer work were preserved separately: missing
  declared `flash-attn`, a tmux uv-cache inheritance miss, HTTP 403 from W&B's
  GraphQL endpoint on the provider IP, and a stale step-0 barrier left by the
  aborted controller. W&B moved to offline logging; public HF mirroring stayed
  active after every completed update.
- Run identity:
  `rl-v4-pair7-overfit60-8fd1a918-5090-20260819`. The complete four-policy
  update-10 checkpoint was uploaded, anonymously re-downloaded, and SHA-256
  verified. Update 0 had critical normal return `0.05556`,
  generated-minus-dropped return `0.03704`, and receiver target-action rate
  `0.5` under both conditions. Update 10 had critical normal return `0.05556`,
  zero generated-minus-dropped return, receiver target-action rate `0.5` under
  both conditions, and generated-minus-shuffled return `-0.01852`. Therefore no
  communication learning was observed by update 10.
- Rejected update 11: before applying its atomic optimizer update, one role
  measured mean absolute log-probability error `0.083509512` and mean mismatch
  KL `0.025269199`, above `0.05` and `0.01`. Exact adapter SHA-256 values matched
  across the trainer broadcast, controller progress, lag-zero rescore manifest,
  and all three vLLM registries. The rescorer reuses same-backend rollout
  log-probabilities rather than running an independent HF forward pass. This
  isolates the failure to the vLLM-FA2 versus HF-FA2 numerical envelope on a
  peaked constrained choice, not stale-policy lag, OOM, partial optimization,
  or a launcher failure.
- Decision: preserve this run as rejected at its declared backend bound and
  restart from the SFT initializer with a new identity. The newly declared hard
  compatibility ceilings are `0.10` mean absolute log-probability error and
  `0.05` mean mismatch KL. KL regularization, probability-tail checks, atomic
  four-policy updates, task, terminal reward, curriculum, and evaluation remain
  unchanged. This is not a retroactive pass for the rejected update.
- Operator correction before the next material run: a first replacement under
  `0.10`/`0.05` was intentionally stopped after its first completed update.
  For this narrow overfit feasibility test, the final predeclared backend
  envelope is `0.25` mean absolute log-probability error and `0.15` mean
  same-policy mismatch KL. These bounds do not change the separate `kl_tau =
  0.001` policy regularizer, reward, samples, or evaluation. The stopped
  one-update replacement is not treated as evidence.

### 2026-08-19 — compact multi-pair v6 CPU preparation

- Status: CPU implementation and focused audits complete; fresh Linux/GPU
  integration run pending. No pod or GPU was active during this work.
- Hypothesis: the pair-7 run learned generic capture behavior because only four
  receiver samples per world were available and matched decoys occupied half
  of every optimizer batch. A small model should receive a clearer conditional
  signal when several ambiguous communication worlds co-occur, irrelevant
  receiver context is removed, and decoys remain causal evaluation controls
  rather than optimizer examples.
- Curriculum: `data/rl_v4/staged_curriculum_v6_compact_multipair_40.json`.
  Every one of 40 updates contains pair 7 left/right and pair 9 left/right.
  Pair 7 trains receiver `blue-1`; pair 9 trains receiver `blue-0`. Each group
  has eight common-random replicas, for 160 critical groups and 1,280 focused
  receiver decisions. No ordinary or decoy group is optimized in this narrow
  learnability run. File SHA-256:
  `0cffa3458f945bad62dcae8734fc6e5fd3ab1dacc2a21e96e3d495951659bb4f`.
- Reward and credit: unchanged replay-verified terminal control delta with
  focused-agent within-world leave-one-out credit. No message/action/capture/
  target reward was introduced. A proposed cross-world scalar baseline was not
  added: it would not solve contextual credit and could mix different return
  distributions. Instead, both worlds are balanced inside every update and
  each world's counterfactual variation remains isolated by common randomness.
- Prompt: new `focused_handoff_compact` profile applies only to the focused
  receiver's ACT request. It retains the inbox, complete legal action list,
  self state, local events, unknown neighbors, and known nodes referenced by a
  legal action; other agents and all evaluation defaults remain unchanged.
  The profile and replica count are bound into the shared-return spec and
  immutable production-plan digest. Legacy full-prompt spec hashes are
  preserved.
- Evaluation: the old pair-7 online probe now supports a multipair mode. The
  48-game checkpoint pulse reports aggregate and per-pair normal-minus-dropped,
  normal-minus-shuffled, receiver target choice, critical-minus-decoy
  specificity, and protocol validity. Matched decoys are therefore still
  tested at every checkpoint despite receiving no optimizer gradient. The
  unchanged development and frozen OOD suites remain the selection and final
  claim surfaces.
- Observability: compact controller progress and W&B now include focused
  non-zero-advantage rate and within-group focused-action diversity on every
  update. This makes a low-diversity first batch visible without another
  multi-stage launch gate. Public HF mirroring remains per-update, with full
  adapter snapshots every ten updates.
- CPU audit: `scripts/audit_communication_learnability_plan.py` passed. Both
  pairs have indistinguishable critical receiver observations across latent
  worlds, identical receiver legal actions, no message-unlocked action, and
  minimum certified terminal advantages `0.074074` (pair 7) and `0.066667`
  (pair 9). Script SHA-256:
  `798dfe15f576236ee88dee91d228d68e41e190756b537226cc79823ddf8c0a9c`.
- Tests: all 14 production/curriculum tests pass; focused prompt, immutable
  spec, multipair summary, W&B metric, shell syntax, compile, JSON, Ruff, and
  diff-whitespace checks pass. Full PyTorch-dependent test collection was not
  run on the Mac because the repository submodules are intentionally absent
  and installing PyTorch would violate the local-storage policy. A fresh Linux
  host must run the complete suite once before launch.
- Trainer: `configs/rl_v4_1_7b_compact_multipair_40.toml`, SHA-256
  `2daa6202a7a4e71e33bae6b1ac9742dd8717042f720d38ab0bcf32ac8819b4c9`.
  It retains LoRA rank 16, learning rate `7.5e-6`, terminal-only loss, policy
  KL regularization, and the measured `0.25`/`0.15` backend compatibility
  alarms.
- Runbook: `COMMUNICATION_LEARNABILITY_V6_RUNBOOK.md`. The remaining empirical
  risk is insufficient on-policy action diversity even with eight replicas.
  The first GPU update will measure, not assume, this quantity. The run's
  success still requires intervention lift on both pairs and held-out
  development improvement; training return alone remains insufficient.

### 2026-08-19 — compact multi-pair v6 GPU launch

- Status: running. The definitive 40-update Qwen3-1.7B communication-
  learnability run launched at `2026-08-19T12:10Z` as
  `rl-v4-compact-multipair40-3ca20933-l40s-a`.
- Question: can terminal-return RL teach two separately optimized receiver
  adapters to choose between already-legal targets using a teammate's private
  fact, after the valid pair-7 60-update run learned only a generic capture
  tactic?
- Source commit: `3ca20933409f2e02f9dce60a3f295f10d15a0806` on
  `exp/swarm-arena-4b`. The preceding Linux integration run exposed one test-
  only import error: the compact-prompt test called the legacy communication
  reconstructor on a v4 handoff manifest. Commit `3ca20933` switched that test
  to the v4 handoff reconstructor; runtime code already used the correct
  function. Focused Ruff and the complete Linux suite then passed 130 tests
  with two known SWIG deprecation warnings.
- Construction audit: passed at curriculum SHA-256
  `0cffa3458f945bad62dcae8734fc6e5fd3ab1dacc2a21e96e3d495951659bb4f`.
  Every update contains pair 7 and pair 9 in both latent worlds, eight common-
  random replicas per group, and receiver ACT-only focused credit. Certified
  minimum terminal advantages remain `0.074074` and `0.066667` respectively.
- Hardware/runtime: four NVIDIA L40S GPUs with 46,068 MiB each, driver
  `580.126.09`; vLLM `0.22.0+cu129`. GPUs 1--3 run independent rollout servers
  and GPU 0 runs the Prime trainer. The fresh image omitted Prime's optional
  `flash_attn` import, so the repository-pinned prebuilt
  `flash-attn==2.8.3+cu128torch2.11` wheel was installed before calibration.
  No config, sample, or threshold was changed in response.
- Public inputs: Qwen3-1.7B revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; SFT revision
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`, adapter SHA-256
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`;
  historical opponent revision
  `ad51ef261f3e7b7b2d3c6433106bd667ba1da81c`.
- Fresh 32-decision host calibration: parity and four-policy optimizer
  isolation passed. Mean absolute log-probability error was `0.0018759182`,
  p99 absolute error `0.0581101403`, mean mismatch KL `0.0000817391`, and max
  mismatch KL `0.0146048069`. The trainer's predeclared backend alarms remain
  `0.25` mean absolute error and `0.15` mean mismatch KL; policy regularization
  remains `kl_tau = 0.001`.
- Runtime certificate SHA-256:
  `a3da30e9d65b5ef457058444ac6ccf3dbd448758063c26fc7a9e86c5173a8115`.
  Production-plan SHA-256:
  `93ae311d15bf57a40d7d4a16298bdb8fe35e28997eb68726bcc69e251ef6d64a`.
  Schedule SHA-256:
  `acbe23d77277293e71b614a2024ec3e28c5ed474e3a3bad8187b730a6d179821`.
- Recovery/logging: compact run state is mirrored publicly to
  `CK0607/swarm-arena-live-runs` after every completed update; complete atomic
  four-adapter checkpoints are mirrored and anonymously hash-verified every
  ten updates. Trainer W&B remains offline by design so telemetry failure
  cannot stop optimization. Controller/evaluation telemetry is online at
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v4-compact-multipair40-3ca20933-l40s-a-controller-v1`.
  A detached lightweight watcher checks process/GPU/log/mirror health and is
  authorized to preserve evidence and apply narrow infrastructure recovery,
  but not to weaken or relabel scientific gates.
- Result interpretation remains predeclared: gameplay return alone is a
  capability result. Communication learning requires normal messages to beat
  dropped and shuffled messages in aggregate and on both pairs, with a larger
  critical than matched-decoy effect. No result is claimed at launch.

### 2026-08-19 — compact multi-pair v6 completed result

- Status: completed all 40 atomic updates and five checkpoint evaluations;
  mechanically valid, scientifically negative for communication learning.
- Run: `rl-v4-compact-multipair40-3ca20933-l40s-a`, source
  `3ca20933409f2e02f9dce60a3f295f10d15a0806`. Launch-to-final-evaluation wall
  time was approximately 72 minutes. The provider price for this allocation
  was not supplied, so no dollar cost is inferred.
- Integrity: all 130 Linux tests passed before launch. No OOM, NaN, NCCL,
  launcher, protocol, or parity rejection occurred. The final atomic update's
  largest per-policy mean absolute log-probability error was `0.0305019` and
  largest mean mismatch KL was `0.00809766`, inside the unchanged `0.25` and
  `0.15` compatibility alarms. Policy KL regularization remained
  `kl_tau = 0.001`.
- Durable artifacts: complete four-adapter sets at updates 10, 20, 30, and 40
  were uploaded to `CK0607/swarm-arena-live-runs`, anonymously downloaded, and
  SHA-256 verified. Final ready-record SHA-256 is
  `640f17dba7003298900ae8712acfc13d16ff4780eb5bbcd0ecbf5c4fae074eef`;
  final adapter hashes are `4b50b681...a85fb`, `2124b5bb...9ed0f`,
  `9108b0da...cbeac`, and `c0dfb9b9...d04d1`. Controller and trainer W&B runs
  both synced successfully.
- Checkpoint curve, normal-minus-dropped return at updates 0/10/20/30/40:
  `-0.01667`, `-0.00926`, `-0.00833`, `+0.00093`, `+0.00833`.
  Normal-minus-shuffled was `+0.02778`, `-0.00926`, `0`, `-0.05093`, and
  `-0.01759`. Receiver target choice stayed exactly `4/8` under both normal
  and dropped messages at every checkpoint. Protocol and grounding stayed
  `1.0`.
- Final paired uncertainty: update-40 normal-minus-dropped was `+0.00833`,
  four-unit bootstrap 95% interval `[-0.05370,+0.07037]`, with two pair/world
  units positive and two negative. Normal-minus-shuffled was `-0.01759`,
  interval `[-0.08333,+0.04722]`. Critical-minus-decoy specificity was
  `+0.00833`, interval `[0,+0.025]`, from one positive unit and three zero.
- Per-pair failure: pair 7 normal-minus-dropped was exactly `0`; pair 9 was
  `+0.01667`. Neither changed receiver target accuracy, and both had negative
  normal-minus-shuffled return. The preregistered aggregate, both-pair,
  receiver-choice, and shuffled-message requirements therefore did not pass.
- Training diagnosis: signal density was not the bottleneck. Non-zero focused
  advantage rose to `95%` for both receiver slots in the last ten updates.
  Pair-9 target captures rose from `6.25%` in the first ten updates to `20.63%`
  in the last ten; pair 7 rose from `20.63%` to `26.25%`. Pair 9 improved far
  more in the left world (`51.25%` target actions) than the right (`35%`), so
  the policy strengthened a target prior and generic capture behavior rather
  than switching from the teammate fact.
- Decision: no checkpoint is eligible for development selection; the held-out
  development and frozen OOD suites remain unopened. This is a clean negative
  result for within-world leave-one-out receiver credit on Qwen3-1.7B, not a
  failed RL pipeline. Full compact evidence and interpretation live in
  `results/rl_v4_compact_multipair40_1_7b/RESULT.md`.
- Next action: prospectively implement and audit a paired terminal-return
  receiver estimator centered on normal-versus-message-intervention outcomes,
  with matched decoys as a null control. Require balanced non-zero
  critical-over-decoy signal for both receiver slots before another optimizer
  run; if the 1.7B model cannot meet that gate, move to the 4B instruct model.
- Instance status: trainer, inference, rescore, and watcher sessions were
  stopped after publication. All four GPUs report `0 MiB` and no compute
  processes. Safe to decommission.

### 2026-08-19 — literature note: RLSVR / SpyRL

- Reference: Wang et al., *From RLVR to RLSVR: Task Transformation Induces
  Self-Verifiable Rewards for Open-Ended LLM Self-Improvement*, arXiv
  `2607.23802v2`, COLM 2026. The authors release SpyRL and train
  Qwen3-4B-Instruct-2507 with GRPO for 100 iterations on one eight-GPU node.
- Conceptual overlap: SpyRL also constructs an information-asymmetric
  multi-agent game around a deterministic environment-assigned latent fact.
  Four players see complete task information, one spy sees degraded
  information, all produce outputs, and the group votes for the spy. This is
  strong external evidence that proxy games with private information can make
  otherwise difficult objectives trainable at the 4B scale.
- Important difference: SpyRL is competitive detection, not constrained
  communication or team cooperation. Its five players sample from a shared
  performer policy and a shared detector policy; it does not maintain five
  independently adapting policies. It uses one public-output/voting round and
  does not evaluate a causal message channel. Swarm Arena instead maintains
  independently updated agent-slot adapters in a multi-round 4v4 environment
  and directly intervenes on teammate messages.
- Reward caveat: the detector's spy-identification reward is exactly verifiable
  because the environment assigned the spy identity. The performer's reward is
  derived from learned detector votes, so it can inherit detector weaknesses or
  superficial shortcuts. SpyRL validates correlation with human and model
  quality rankings, but this remains less direct than Swarm Arena's deterministic
  terminal outcome combined with matched message interventions.
- Credit-assignment lesson: SpyRL's role-advantage estimation is essential.
  Their raw-role-reward ablation underperforms the untrained model, while the
  role-calibrated version improves substantially. This supports Swarm Arena's
  paired receiver estimator: compare the same receiver under normal and dropped
  messages, center within the receiver role, and avoid broadcasting an
  uncalibrated team return to every token span.
- Curriculum lesson: SpyRL alternates performer and detector optimization based
  on detector saturation, and its joint-update ablation collapses. Swarm Arena's
  current receiver-ACT-only phase already follows the same stability principle.
  If receiver learning succeeds, a prospective extension should alternate
  frozen-sender/receiver-learning and frozen-receiver/sender-learning phases
  rather than updating both sides simultaneously.
- Scale lesson: SpyRL reports the largest marginal benefit when increasing from
  three to five players, with diminishing returns at six and eight. This argues
  against increasing Swarm Arena's agent count before the four-agent team learns
  a reliable information handoff.
- Evaluation lesson: a rising training return establishes capability learning,
  not its mechanism. An agent can raise return by learning a generic capture
  prior, exploiting one opponent, or using a state cue that bypasses the
  teammate message. Evidence for communication requires a controlled causal
  contrast on the same state distribution: normal messages must beat dropped,
  shuffled, and irrelevant/reference messages; the receiver must change toward
  the message-conditioned target; the effect must be larger on critical cases
  than matched decoys; and it must generalize to held-out pairs/opponents without
  reducing absolute task performance or protocol validity. These interventions
  make the claim more rigorous than selecting the checkpoint with the highest
  reward curve.
- Blog framing: SpyRL asks whether information-asymmetric self-play can create a
  verifiable proxy reward for open-ended quality. Swarm Arena asks the
  complementary causal question: can independently adapting small agents learn
  to use a constrained teammate message containing private information, without
  an explicit communication reward?

### 2026-08-19 — Qwen3-4B paired receiver RL completed result

- Status: completed all 30 atomic updates and four checkpoint evaluations.
  Mechanically valid; positive evidence of message sensitivity, but not a
  demonstrated improvement in receiver target selection or absolute task
  performance.
- Run: `rl-v7-paired4b30-memfix-5f3ddf7e`, source
  `5f3ddf7ef9c38f8f37954a216a0604eeeef18500`. The production run started at
  `2026-08-19T16:23Z`; final evaluation completed at `18:42Z`, approximately
  2 hours 19 minutes later.
- Model: public `Qwen/Qwen3-4B-Instruct-2507` revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`; public SFT adapter
  `CK0607/Qwen3-4B-Swarm-Arena-SFT-v2` revision
  `d1a55d5594c8b544121e546e14229268c8c26bae`, adapter SHA-256
  `168c9f9cdd0537660b664e9863ec9e351faf5e84d85ffbc77e95501fe1d903d2`.
  Four agent-slot LoRA policies were optimized independently.
- Plan and calibration: paired terminal-return receiver credit over pair 7 and
  pair 9 in both latent worlds, eight common-random replicas, receiver ACT-only
  spans, and rotating base/SFT/historical/current opponents. Production-plan
  SHA-256 was
  `41cb423f1e527725c754ecbd75af87ad956caed12211e1563958920874342b51`.
  The valid preflight calibration had mean absolute log-probability error
  `0.0036866383` and mean mismatch KL `0.0004692887`.
- Infrastructure recovery: the first run identity,
  `rl-v7-paired4b30-5f3ddf7e`, completed three updates and then OOMed during
  backward on a 2,526-token slice. PyTorch held 5.24 GiB reserved but
  unallocated and recommended expandable segments. No atomic update-10
  checkpoint existed, so this partial state was preserved as rejected failure
  evidence rather than treated as resumable. The replacement changed only the
  trainer allocator environment to
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Its first trainer launch
  selected the inference UV cache and failed before loading because that
  environment lacked `flash_attn`; the log was archived and the trainer was
  relaunched with the previously validated default UV environment before any
  rollout, barrier, or update. It then completed without OOM, NaN, NCCL, parity,
  or protocol failure. Peak trainer memory was approximately 42.8 GiB.
- Integrity and publication: complete four-adapter checkpoints at updates 10,
  20, and 30 were uploaded to public
  `CK0607/swarm-arena-live-runs`, anonymously downloaded, and SHA-256 verified.
  Final ready-record SHA-256 begins `99c50e0c`. Controller W&B is
  `rl-v7-paired4b30-memfix-5f3ddf7e-controller-v1`; the trainer's offline run
  `vzd45lk6` was explicitly synced before decommission approval.
- Training-pair evaluation curve at updates 0/10/20/30:
  normal-minus-dropped return was `+0.05556`, `+0.07037`, `+0.05370`, and
  `+0.07870`; normal-minus-shuffled was `+0.03889`, `+0.03704`, `+0.03704`,
  and `+0.04537`; critical-minus-decoy specificity was `+0.07037`, `+0.08889`,
  `+0.10741`, and `+0.10741`. Protocol action validity, broadcast validity,
  and broadcast grounding remained `1.0` throughout.
- Final pair detail: pair 7 normal-minus-dropped was `+0.07407`,
  normal-minus-shuffled `+0.07407`, and specificity `+0.14815`. Pair 9 was
  `+0.08333`, `+0.01667`, and `+0.06667`, respectively. Thus both training
  pairs ended with positive intervention lift, unlike the 1.7B run.
- Scientific caveat: normal receiver target-action accuracy was exactly `0.5`
  under both normal and dropped messages at updates 10, 20, and 30. Aggregate
  final normal return remained negative at `-0.06204`, compared with
  `-0.05185` at baseline. The main normal-minus-dropped statistic improved by
  only `+0.02315` over the 4B baseline and was non-monotonic. Most of the large
  improvement over the prior 1.7B experiment therefore comes from the 4B
  starting policy, not an established RL effect.
- Verdict: the experiment establishes that the 4B policy has reproducible,
  pair-specific causal message sensitivity and that the paired estimator can
  train stably. It does not establish that RL taught the receiver to decode the
  private fact into the correct target action, improved game performance, or
  generalized beyond the two training pairs. The held-out development and
  frozen OOD suites remain unopened because the stronger behavioral claim was
  not met.
- Next action: diagnose which action transitions create positive terminal
  intervention lift despite unchanged target accuracy. The next prospective
  experiment should optimize an alternating frozen-sender/receiver phase only
  after confirming that its terminal counterfactual reward ranks the intended
  target-switch behavior, then use a larger training-pair set and select on a
  development suite without touching frozen OOD.
- Instance status: final compact artifacts, checkpoints, controller telemetry,
  and trainer telemetry are off-node. Remaining GPU processes are idle; safe to
  decommission.

### 2026-08-20 — Qwen3-4B raw rollout attribution diagnosis

- Status: completed over every paired training replica from
  `rl-v7-paired4b30-memfix-5f3ddf7e` (`960` replicas: 30 updates, four
  pair/world groups, eight common-random replicas). Analysis script:
  `scripts/analyze_paired_rollout_attribution.py`.
- Storage: the complete run remains node-local at approximately `17 GiB`, with
  `1,279` Prime `.bin` rollout shards. The semantic paired-return audit is
  `576 MiB`. Neither was copied onto the Mac. The compact `24 KiB` attribution
  report was uploaded to public `CK0607/swarm-arena-live-runs` at Hub commit
  `9d30cb39147601005c0eeef768d7ebb8a875a52a`; local and anonymous-download
  SHA-256 both equal
  `80bb1ddc705afad9168bbee4c15e7a9002ce4a7b944d1a1924d9388d77c48809`.
  The full semantic audit was also gzip-compressed to `106,345,893` bytes and
  preserved at Hub commit `af5e099fa3469bef58e7ae3ce3cde1f3c602e41f`;
  local and anonymous-download SHA-256 both equal
  `3e6b1f177742bbd279ccc4ca02a06d7990e1472c6c641ddbe4c739e4dbe37569`.
  This retains reproducible semantic evidence without publishing the disposable
  `17 GiB` trainer shards.
- Direct attribution result: mean normal-minus-dropped terminal return across
  training replicas was `+0.04951`; `54.38%` had a non-zero effect and `44.48%`
  a positive effect. In all `522/522` non-zero-effect replicas, the trained
  receiver's first action changed. In `414/427` positive-effect replicas
  (`96.96%`), normal selected the active target while dropped did not. This
  rejects the hypothesis that the positive paired reward was created only by
  an unrelated teammate transition while the credited receiver was unchanged.
- On-policy learning curve: normal receiver target selection increased from
  `70.31%` in updates 0--9 to `91.25%` in 10--19 and `97.81%` in 20--29;
  dropped-message selection stayed approximately chance (`50.31%`, `50.00%`,
  `49.69%`). Mean paired return effect rose `+0.02329 -> +0.05917 -> +0.06606`.
  The run therefore did learn the intended target switch under its rollout
  context, despite the small checkpoint evaluation reporting `50%/50%`.
- World decomposition: in updates 20--29 normal target selection reached
  `100%` in all four pair/world cells. Dropped behavior retained a strong
  left-target default: pair 7 left/right target accuracy was `97.5%/1.25%`,
  and pair 9 was `88.75%/11.25%`. Consequently almost all measured reward lift
  appears in right-exposed worlds, while left-exposed worlds have little lift
  because the no-message default is already correct. This is asymmetric
  fallback behavior, not evidence that normal messages are ignored.
- Train/eval mismatch found: the credited receiver is trained with the
  `focused_handoff_compact` action prompt and temperature-1 constrained
  multinomial sampling. `run_pair7_communication_eval.py` calls
  `evaluate_final_case -> evaluate_crossplay`, whose action prompt is always
  the full profile, at temperature `0.4`; its final receiver statistic has only
  eight critical rows. Thus the published checkpoint eval currently mixes
  prompt transfer, sampling change, and a high-variance behavioral estimate
  with the within-training-distribution learnability question.
- Interpretation: this run is more successful than the original headline
  suggested. It demonstrates receiver-specific, message-conditioned learning
  on the compact training interface. It still does not demonstrate robust
  communication under the normal full game interface or held-out
  generalization. The failure is now localized to transfer/evaluation rather
  than credit attribution or optimizer stability.
- Prospective fix: add a sufficiently powered matched compact-prompt
  development evaluation at the same checkpoint, then separately measure
  compact-to-full-prompt transfer. Balance the dropped baseline or add
  content-swapped messages so the no-message left-target prior cannot dominate
  one world. Keep the frozen OOD suite untouched until the full-prompt
  development effect is positive.

### 2026-08-20 — powered 4B prompt-transfer matrix

- Status: completed (`3,072/3,072` games); design fixed before any cell summary
  was available. All eight cells retained 100% action validity, broadcast
  validity, and broadcast grounding.
- Scope: training pairs 7 and 9 only, both latent worlds, critical and matched
  decoy cases, normal/dropped/sender-shuffled messages, and 16 independent
  sampling repetitions. The frozen development and OOD suites remain unopened.
- Matrix: final step-30 four-policy checkpoint and its SFT initializer, each
  evaluated with the receiver's exact `focused_handoff_compact` training prompt
  and the normal full-game prompt, at temperatures `1.0` and `0.4`. Each cell
  contains 384 complete games and 64 critical matched intervention units; the
  full eight-cell matrix contains 3,072 games.
- Purpose: compact/T=1 tests in-distribution learnability; compact/T=0.4
  isolates the sampling change; full/T=1 isolates prompt transfer; full/T=0.4
  measures the existing operational evaluation interface. Identically seeded
  SFT cells distinguish RL learning from starting-model capability.
- Primary readout: paired normal-minus-dropped terminal return and receiver
  target accuracy, final-minus-SFT difference-in-differences, and
  critical-minus-decoy specificity. Normal-minus-shuffled return, sender target
  fact rate, protocol validity, pair/world decomposition, Wilson intervals,
  paired bootstrap intervals, and exact McNemar tests are supporting readouts.
- Interpretation fixed prospectively: a positive compact result alone is an
  in-interface learning result. A full-prompt result is required for transfer.
  Neither licenses a held-out or broad swarm claim. The no-message left-target
  prior is reported per world rather than hidden by aggregate averaging.
- Implementation source: `340ca74d` adds an explicit per-agent action-prompt
  profile to cross-play and the training-pair evaluator. Focused Linux test
  passed 1/1 and changed-file Ruff passed before launch. The first pytest
  invocation omitted the experiment `PYTHONPATH` and collected no test; the
  corrected command passed and no model request was made by the failed command.
- Results at the trained checkpoint: compact/T=0.4 target accuracy was
  `100%` normal versus `50%` dropped, paired return lift `+0.07569` with
  sampling-bootstrap 95% CI `[+0.05810,+0.09340]`. Compact/T=1.0 was
  `89.06%` versus `50%`, lift `+0.05255` `[+0.03542,+0.07037]`. Under the full
  prompt, target accuracy fell to `51.56%` versus `50%` at T=0.4 and `48.44%`
  versus `51.56%` at T=1.0, although return lift stayed positive (`+0.07037`
  and `+0.05764`). This is direct evidence of compact-interface learning with
  poor transfer of the intended target-selection rule to the real interface.
- RL-minus-SFT: compact return-lift difference was `+0.03102`
  `[+0.01505,+0.04769]` at T=0.4 and `+0.04028`
  `[+0.02037,+0.06019]` at T=1.0; normal target-accuracy improvements were
  `+25.0` and `+26.56` percentage points. Full-prompt return-lift difference
  was smaller: `+0.01285` `[-0.00868,+0.03426]` at T=0.4 and `+0.02164`
  `[+0.00255,+0.04109]` at T=1.0. Full-prompt target-accuracy differences were
  `+17.19` points `[+3.13,+29.69]` and `+9.38` points
  `[-3.13,+21.88]`, respectively. Thus RL added clear compact-prompt capability
  but at best weak/inconsistent full-prompt behavioral transfer.
- Specificity: trained-checkpoint critical-minus-decoy drop lift was `+0.04016`
  compact/T=0.4, `+0.03299` compact/T=1.0, `+0.10428` full/T=0.4, and
  `+0.08287` full/T=1.0. Positive terminal sensitivity in the full cells cannot
  substitute for target accuracy; rollout attribution remains necessary.
- Runtime/recovery: the four final cells used GPUs 0--3. The four SFT controls
  were initially all routed through GPU 0; after final cells completed, three
  were deliberately stopped and resumed by immutable evaluation ID on GPUs
  1--3. Two resume attempts failed before generation because the evaluator
  correctly rejected source-commit mismatches (current head, then `847ecf31`)
  against manifests bound to `0fe6949b`. Checkout at exact `0fe6949b` resumed
  successfully; existing rows were retained and no cell was duplicated or
  relabeled.
- Artifacts: complete rows, manifests, summaries, and analysis are public under
  `CK0607/swarm-arena-live-runs/rl-v7-paired4b30-memfix-5f3ddf7e/diagnostics/prompt-transfer`.
  Final upload commit is `dc4db51b38f30a75a1e4177d42dc856f35c149f5`;
  canonical report SHA is
  `6b0bb1d781cd8ed916c30cf4f3cca4980b73d3b088346e656b8ff7f48d5595ee`
  and file SHA-256 is
  `7266c6b090dbb3943a569f04d77cdc1c321566f436d5ecac5b751e07fbf2fc9f`.
  Anonymous repository listing confirmed all eight completed cell summaries.
  Two anonymous re-download/hash invocations for the final analysis were
  aborted by the local approval-review timeout rather than a Hub error; public
  upload is confirmed but byte re-verification is not claimed.
- Verdict: positive and useful. The old run did learn causal message-conditioned
  receiver behavior, but mostly under the compact training prompt. This
  justifies the next preregistered run's full-prompt-only curriculum; it does
  not by itself justify a held-out swarm-intelligence claim.

### 2026-08-20 — representative 4B full-interface curriculum (prospective)

- Status: implementation complete; launch waits only for the powered
  prompt-transfer matrix and one exact-runtime calibration on the freed trainer
  GPU. The frozen OOD evaluation remains unopened.
- Hypothesis: the Qwen3-4B SFT initializer can learn a message-conditioned
  receiver action under the same full action prompt used at evaluation when the
  curriculum covers all directed roles and increases the remaining game horizon
  gradually. This is a stricter target than the preceding compact-prompt run.
- Curriculum: `staged_curriculum_v8_4b_representative_full_60.json` schedules
  60 updates and 240 groups over training pairs 12--23. Every six-update cycle
  covers all 12 directed sender-to-receiver roles, both latent worlds, and all
  four receiver policy slots. The first 12 updates use two remaining turns,
  followed by 24 updates each with three and four remaining turns.
- Credit and optimization: four common-random normal/dropped replicas provide
  paired terminal-return advantage. Only the normal receiver's first `ACT`
  span is trainable; there is no message, target, capture, protocol, or other
  additive shaping reward. Four LoRA rank-32 agent policies remain independently
  optimized from the same pinned 4B SFT initializer at learning rate `5e-6`.
- Development measurement: checkpoints at updates 0/12/24/36/48/60 are measured
  on unseen training-manifest pairs 24, 28, 32, and 33, both worlds, with the full action
  prompt and a fixed four-turn horizon. Primary endpoints are
  normal-minus-dropped terminal return and receiver target accuracy;
  shuffled-message lift and matched critical/decoy specificity distinguish
  useful message content from generic perturbation sensitivity.
  Those four pairs form the directed cycle `0->1, 1->2, 2->3, 3->0`, so every
  sender and every receiver policy slot appears exactly once in each pulse;
  the initially drafted contiguous range 24--27 covered all receivers but
  overrepresented sender 0 and was corrected before any training launch.
- Implementation correction discovered before launch: the generic preflight
  still hard-coded the older 1.7B run's learning rate and LoRA rank even though
  PREPARE, parity, and the runtime certificate already hash-bind the exact
  trainer config. It now validates safe numeric/rank invariants and the inference
  rank limit without duplicating obsolete experiment values. The launcher now
  passes the curriculum's declared online pair indices to the pulse evaluator
  and accepts an explicit W&B model tag. Trainer W&B is offline/failure-isolated;
  the compact controller sidecar remains the online telemetry source.
- Validation: focused production-plan tests passed `14/14`; focused staged-pulse
  tests passed `15/15`; changed Python Ruff and shell syntax checks passed. One
  validation command accidentally included the Bash launcher in Ruff's Python
  inputs and produced syntax noise without executing code or changing state;
  the corrected language-specific checks passed.
- Memory setup: the trainer launcher now explicitly carries
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (overridable by a scoped
  Swarm environment variable), preserving the allocator-only fix that let the
  prior rank-32 4B run complete at roughly 42.8 GiB peak memory.
- Interpretation boundary fixed prospectively: success on pairs 24--27 is
  within-manifest role/pair transfer. Only after development selection will one
  checkpoint be evaluated once on the unchanged frozen OOD suite. A rising
  training return without dropped/shuffled intervention lift is not a positive
  communication result.

### 2026-08-20 — representative 4B full-interface RL v8

- Status: completed, 60/60 optimizer updates. Verdict: useful development
  signal, rejected as a semantic held-out communication result.
- Source/run: commit `3c6b8f7998062349d34f4a6f0e9796549032a876`, run
  `rl-v8-representative4b60-3c6b8f79`; Qwen3-4B SFT initialization, four
  separate rank-32 LoRA policies, learning rate `5e-6`, KL tau `0.001`.
- Exact-runtime calibration passed: mean log-probability error
  `0.0006304588`, p99 `0.004082242`, mean mismatch KL `0.00007731738`, and
  maximum mismatch KL `0.0934515`.
- Runtime result: no OOM, NaN, parity rejection, or fatal training failure.
  Checkpoints 12/24/36/48/60 were mirrored to public
  `CK0607/swarm-arena-live-runs` and anonymously re-downloaded/hash-verified.
  Controller telemetry is public at
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v8-representative4b60-3c6b8f79-controller-v1`.
- Development trajectory (`normal return`, `normal-dropped`,
  `normal-shuffled`, `critical-decoy specificity`, `target accuracy`): update
  0 `(+.00828,-.01892,+.00774,-.02305,.625)`; 12
  `(+.02205,+.02105,+.00327,+.04619,.625)`; 24
  `(+.05328,+.03360,+.04827,+.04726,.625)`; 36
  `(+.02092,+.03778,+.01224,+.08001,.625)`; 48
  `(+.01936,+.01054,+.01903,+.06300,.625)`; 60
  `(+.03601,+.03058,-.00620,-.00069,.625)`. Update 36 was selected for
  communication specificity; update 24 had the highest normal return.
- GPU/time/cost: approximately 8.5 hours of a 4xL40S node for the live run and
  final small screen, roughly `$17` at the user-reported `$2/hour`; provider
  billing was not independently available. The node was reported safe to
  decommission after public preservation; a confirmed provider-side
  decommission timestamp was not observed.

### 2026-08-20 — v8 small frozen exploratory screen and rollout diagnosis

- Status: completed `252/252` rows after the user stopped the full frozen run
  at 73 rows to minimize GPU cost. This is a small feasibility screen with only
  two independent handoff units, not the preregistered confirmatory final.
- Public artifact: Hub revision
  `be6722128f6ded06481830f4051904f4e74b38db` under
  `runs/rl-v8-representative4b60-3c6b8f79-final-heldout-small-u36/`.
- Critical RL normal-minus-dropped return was `+.014261`, 95% interval
  `[-.007937,+.036458]`; shuffled `+.013145`
  `[+.010417,+.015873]`; delayed `-.000124`
  `[-.015873,+.015625]`; zero-budget `-.008185`
  `[-.047619,+.031250]`.
- The matched decoy normal-minus-dropped effect was identically `+.014261`
  `[+.005952,+.022569]`, leaving critical-minus-decoy specificity approximately
  zero `[-.013889,+.013889]`. RL-specific drop lift over SFT was `+.020337`,
  but critical capture lift was `-.041667`, ordinary-hard RL-minus-SFT was
  `-.053968`, legacy was `-.080808` (`n=1`), and handoff-normal RL-minus-SFT
  was `-.008433`. All confirmatory claim checks were false.
- Sender target-fact, defined broadcast protocol/grounding, and action validity
  were `1.0`. The original summarizer failed only after all rows completed
  because zero-budget rows correctly stored undefined broadcast protocol fields
  as `null`. Raw evidence was unchanged; a defined-row aggregation recovery
  produced summary SHA-256
  `a1cf1afbd8de772c8d21f03d37bb6da7a2bca7beb9dae4f47df58b4b221f66b3`
  and recovery-record SHA-256
  `4e7a48bb6c83a480b86fd34fb08e244debf50c2d05fbfd29058bfd5a9c9a34d4`.
- Qualitative diagnosis over 186 candidate games found zero invalid actions or
  broadcasts and near-maximal communication spend in 161/162 defined games,
  but mean duplicate-target-turn rate was `0.215`. A positive critical handoff
  existed (`bundle-000`, historical opponent, BLUE: `+.2083` normal versus
  `0` dropped), yet a decoy cell gained the same `+.2083`; bundle-level normal
  returns also reversed from `+.0538` to `-.1032`. The learned effect is best
  described as fragile generic coordination/de-duplication, not reliable use of
  the certified private fact.
- Decision: v9 must replace message presence/silence as the primary training
  contrast with a well-formed target-swapped fact counterfactual, preserve
  terminal-only reward, train receiver actions first, and select checkpoints on
  semantic critical-minus-decoy specificity plus ordinary-game preservation.
  The prospective design is `REPRESENTATIVE_RL_V9_PLAN.md`.

### 2026-08-20 — held-out summary robustness fix

- Status: implemented locally. `progress_eval_v4` now excludes intentionally
  undefined protocol values from means and reports defined/undefined
  denominators. Compact final-eval rows now retain communication spend, invalid
  action/broadcast counts, and duplicate-target-turn rate so future diagnosis
  does not require downloading the full raw trace.
- Failure/retry: the first local `uv run` could not initialize the default uv
  cache under the desktop sandbox; an isolated `/private/tmp` cache then found
  uninitialized repository submodules. A minimal `uv --no-project` test
  environment was used instead. Its first collection lacked
  `huggingface_hub`; the dependency-complete retry exposed an incorrect expected
  test denominator (`8` rather than `6`), which was corrected without changing
  production behavior. The focused summary suite then passed `15/15` and
  changed-file Ruff passed. A focused final-runner test could not collect in the
  lightweight CPU environment because that broad test module imports Torch;
  Torch was deliberately not downloaded solely for this test. The assertion is
  included for the next existing Linux/GPU environment test pass.

### 2026-08-20 — semantic target-swap v9 CPU implementation

- Status: CPU implementation and focused validation completed; no GPU used.
  Source begins from `969fff0197bd712d38cf559e36dfbbae6e7e7c91`.
- Intervention: added a first-class `message_swap` rollout branch. The trusted
  transform requires the sender's generated message to contain the certified
  active candidate, swaps the two candidate node identities in all candidate
  facts and candidate-target intents, and preserves sender, turn, schema, fact
  count, non-candidate content, state, opponent, prompt ordering, and the
  common-random schedule. Messages containing one or both candidate facts are
  supported; omission of the active fact fails closed.
- Credit/safety: actual and swapped trajectories are independently replayed and
  delivery-verified. Immutable evidence records swap sender, turn, candidates,
  active target, decisions, and replay. Advantages are centered verified
  actual-minus-swapped terminal-return effects, routed only to actual receiver
  ACT tokens. The four LoRA policies remain separate. Ordinary preservation
  groups automatically use ordinary leave-one-out return rather than an
  inapplicable message intervention.
- Compact evaluation: the training-pair evaluator is versioned as
  `pair7-semantic-communication-eval-v2` /
  `multipair-semantic-communication-eval-v3`. It adds target-swapped return,
  receiver target-action rate, critical-minus-decoy semantic specificity,
  per-pair decomposition, pulse completeness validation, and W&B metrics. The
  frozen OOD design and rows remain unchanged.
- Curriculum: `staged_curriculum_v9_4b_semantic_target_swap_60.json`, SHA-256
  `15afa7bd6b9516fb61ade1eb6779ce413946e7533051732a879327729177937b`.
  It schedules 60 updates: 216 semantic critical groups and 24 ordinary
  preservation groups, four replicas each. Stages contain 10/20/30 updates at
  two/three/four remaining turns. Each sender and receiver policy slot appears
  exactly 54 times; left/right worlds appear exactly 108 times each. All 12
  directed roles and both worlds are covered. Critical certified opportunities
  are positive and matched-decoy certified advantage is zero.
- Failure/retry: the first prospective schedule used 210 semantic groups. The
  audit rejected it because sender counts were `54/54/53/49` and world counts
  `106/104`. No GPU work or model request occurred. A five-update pattern and
  10/20/30 stage lengths produce 216 groups, exact role/world balance, and a
  passing audit. A first end-to-end test fixture used a seed that did not yield
  a v4 handoff and was replaced with checked-in manifest pair 12. The next test
  exposed that valid sender output may contain both candidates; the transform
  was generalized to swap both identities while still requiring the active
  fact.
- Validation: focused CPU matrix passed `42/42`: compact summaries/pulses
  `15`, production scheduling `14`, shared-return replay/supervisor `10`,
  structural swap `1`, and final-evaluator normal/swap smoke tests `2`.
  The complete Swarm Arena CPU directory then passed `135/135`. Changed-file
  Ruff and Python compilation passed. Temporary import stubs under
  `/private/tmp` avoided installing Torch solely for CPU tests and are not
  repository artifacts.
- Interpretation: the CPU audit proves construction, balance, non-leakage, and
  exact replay/admission behavior. It cannot prove sampled advantages are
  non-zero for the SFT model. A short first-update GPU diagnostic remains the
  honest final precondition before the 60-update run.
- GPU time/cost: none. Instance decommissioned: not applicable.

### 2026-08-21 — privileged-value-function follow-up design

- Reference: Venkatraman, Dinot, and Aitchison, *Le Critique: Privileged Value
  Functions for LLM Reinforcement Learning*, arXiv:2608.16739v1
  (`https://arxiv.org/abs/2608.16739`). The paper is unusually well matched to
  this project: its policy experiments use Qwen3-4B-Instruct-2507 and its
  strongest qualitative result is on a long-horizon, multi-turn environment.
- Main opportunity: retain decentralized policy inputs and separate blue-0…3
  LoRA policies, but train an auxiliary shared critic that sees the acting
  agent's token prefix plus privileged simulator context. Candidate context is
  the pre-action global graph state, agent identity/policy slot, the other
  agents' contemporaneous private observations, and broadcasts already fixed
  before the current action token. This can provide token-level advantages
  without exposing privileged state to the deployed policies.
- Admissibility boundary: the privileged context must be conditionally
  independent of the agent's current sampled token given its policy history.
  The critic must not see future teammate actions influenced by that token,
  later transitions, realized terminal reward, counterfactual outcomes, or
  future messages from the same trajectory. Use Monte-Carlo advantages
  (`lambda_GAE = lambda_target = 1`) for the initial experiment; lower lambda
  introduces critic-dependent bias unless the critic is exact.
- Recommended estimator: do not replace the verified paired target-swap
  estimator immediately. First run v9 unchanged. For a subsequent v10
  ablation, compare (a) paired target-swap/RLOO, (b) an ordinary token critic,
  and (c) the privileged critic while keeping policy data, reward, and
  optimizer fixed. A TETHER-style lagged mixture can start at the group
  baseline and move toward the critic only as held-out critic explained
  variance improves; the mixture coefficient for batch k must be fitted only
  after batch k's policy advantages are frozen and used on later batches.
- Measurements: policy sample efficiency, wall-clock throughput, advantage and
  gradient variance, critic explained variance, communication specificity,
  receiver target use, ordinary-game regression, and compute overhead. The
  paper's own experiments are small-scale and its value setup adds a dedicated
  evaluator and trainer, so this is a meaningful follow-up rather than a free
  change to the current 4xL40S run.
- Current decision: log the design now, but keep the frozen v9 semantic
  target-swap diagnostic and run unchanged. Otherwise a new critic would
  confound whether the counterfactual reward estimator itself is learnable.
- Implementation reference: the authors' public Prime-RL fork is
  `https://github.com/HyperPotatoNeo/prime-values`. It already separates an
  asynchronous value evaluator and value trainer from the policy trainer,
  carries monotonic value-weight versions, and uses bounded queues/replay for
  non-blocking rollouts. A v10 prototype should extend this implementation
  instead of recreating critic infrastructure. It is not currently a drop-in
  path for this project: the documented scope rejects LoRA, and the task-owned
  privileged prompt is fixed for an episode rather than accepting structured
  per-turn online state. Our v10 therefore needs an explicit compatibility
  extension for LoRA policies and causally frozen pre-action snapshots. Swarm
  returns also cross zero, so the default categorical `[0,1]` critic support is
  invalid without a widened support; MSE is the simpler initial choice. The
  first controlled comparison remains paired target-swap/RLOO versus ordinary
  critic versus privileged critic, with identical policy data and optimizer
  settings.

### 2026-08-21 — v9 4B semantic-signal diagnostic and production launch

- Status: the no-optimizer diagnostic completed; the 60-update production run
  is running. Source commit
  `802cc869cd6a794124ea685d94af939e01053e55` on 4xL40S host
  `64.247.196.177:40299`. Base revision is
  `cdbee75f17c01a7cc42f958dc650907174af0554`; SFT revision is
  `d1a55d5594c8b544121e546e14229268c8c26bae`; adapter SHA-256 is
  `168c9f9cdd0537660b664e9863ec9e351faf5e84d85ffbc77e95501fe1d903d2`.
- Host validation: all 135 Linux tests passed. Three vLLM 0.22.0 servers passed
  the structured serving probe. The exact resolved trainer config passed the
  32-decision numerical calibration: mean absolute log-probability error
  `0.0009186207`, p99 `0.0019978492`, and mean mismatch KL `0.0001179355`.
  Bound runtime-certificate SHA-256 is
  `2873b8ac4c518cc4f8797c41f5f34318d430354f2b50d5f3e814612f57d6091c`.
- Learnability diagnostic: one balanced update with four critical groups, all
  four receiver policy slots, two left/two right worlds, and eight common-random
  replicas per group (32 actual/swapped pairs). Every receiver sample received
  a non-zero centered advantage; 21/32 raw semantic effects were non-zero.
  Actual target accuracy was `19/32`. Correct-target actions were `16/17`
  among positive-advantage samples but only `3/15` among negative-advantage
  samples. This is strong evidence that the estimator reinforces the certified
  target and suppresses the wrong candidate rather than merely tracking return.
  The overall mean actual-minus-swapped effect was `+.00822`; its small size is
  expected because the initializer is directionally biased: receiver means
  were blue-0 `-.0500`, blue-1 `+.09677`, blue-2 `-.09722`, blue-3 `+.08333`.
  This failure symmetry is the curriculum's intended learning problem.
- Production plan: SHA-256
  `0f78bb77a3de8e095287f75171118fc45f2a32d2e9c99a16e4b612373f071076`;
  60 updates / 240 groups (`216` critical, `24` ordinary), 12 directed
  handoff pairs, four replicas, receiver ACT-only credit, horizons 2→3→4,
  checkpoints/evaluations every ten updates. Full preflight passed. Public
  recovery heartbeat was anonymously verified at Hub revision
  `6e7b5fbce88f86998d85b0272adfbc03a6146e38`. Controller W&B:
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v9-semantic4b60-802cc869-controller-v1`.
- Startup failure/recovery: the generic launcher changed `.venv` back to the
  default `/root/.cache/uv` environment, whose trainer lacked `flash_attn`.
  It exited before optimizer step 1. The exact log/config and hashes were
  preserved under the run's `audit/failures/20260821T061723Z-trainer-uv-cache/`.
  The narrow retry bypassed `uv run` and launched Torchrun plus Trainer from
  the already-calibrated `/workspace/.uv-cache/...` environment. The recovered
  trainer loaded the model/optimizer and entered the training loop; the frozen
  plan, data, reward, gates, and run identity were unchanged.
- GPU/time/cost: user supplied the node and did not provide an independently
  verifiable hourly rate in this launch message. Software deadline was set
  eight hours after launch with a 45-minute final-sync margin. Instance
  decommissioned: no; production run active.

### 2026-08-21 — v9 evaluator rejection and corrected v9b restart

- Status: running; no optimizer update had completed at the time of this log
  entry. Corrected run identity is `rl-v9b-semantic4b60-34444777`, source
  commit `344447776986af42a16c1344f57fda6706684113`.
- Rejection: the first v9 run stopped during its update-0 baseline after six
  rows, before any optimizer update. The target-swap evaluator assumed every
  generated sender message contained the certified active candidate and raised
  when the model omitted it. This was an evaluation eligibility bug, not a
  training or model failure. Evidence was preserved under
  `audit/failures/20260821T062148Z-update0-swap-ineligible/`; the earlier
  default-UV-cache startup failure remains separately preserved under
  `audit/failures/20260821T061723Z-trainer-uv-cache/`.
- Fix: target swap is now attempted only on the initial handoff turn. A narrowly
  typed ineligibility exception leaves the original message unchanged, records
  eligibility per team, and excludes undefined paired units from semantic
  effects rather than fabricating a fact or treating missing intervention as
  zero. Coverage counts/rates are reported. The supervised training path still
  fails closed if a sampled training group lacks its certified active fact.
- Validation: changed-file Ruff passed and the complete Linux Swarm Arena suite
  passed `137/137`. The corrected run re-used the exact validated interpreter
  and passed numerical calibration: mean absolute log-probability error
  `0.0009186207`, p99 `0.0019978492`, mean mismatch KL `0.0001179355`, and max
  mismatch KL `0.0934324`. Resolved trainer config SHA-256 is
  `2e18aa5ef7d1c576ae390a4a8f28070d157f608bef96b62d93695f16316a30aa`;
  runtime certificate SHA-256 is
  `d8ebab0163b82135b8691633331320f2bab89c348f5ce75cbbc08cc60276f248`;
  frozen production-plan SHA-256 is
  `ea2f4a628d0fd65d910caac2fa3c6ae2cb9dcc4ad90f045f9f6b9b94f52b5d29`.
- Preservation: public recovery heartbeat was anonymously verified at Hub
  revision `5643a8889083147f5d6d39879611be18fe575af2`; live-mirror heartbeat at
  `d79c15a5cd022ae4652ca3100b12bb0915f35f32`. W&B controller run:
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v9b-semantic4b60-34444777-controller-v1`.
- Interpretation: v9b remains the same preregistered 60-update scientific test;
  only evaluator handling of a mathematically undefined intervention changed.
  The corrected update-0 baseline completed all 128 rows with action,
  broadcast, and grounding validity `1.0`. Baseline critical normal return was
  `+.02652`; normal-minus-dropped `+.00713`; normal-minus-shuffled `+.02597`;
  normal-minus-target-swapped `-.009375`; receiver target-action was `.625`
  normally versus `.500` with dropped messages. Target-swap eligibility was
  `10/16` (`.625`) for both critical and decoy units; critical-minus-decoy
  target-swap specificity was `+.03254`. Pairwise heterogeneity was material,
  so these are starting coordinates rather than evidence of learned semantic
  communication.
- First optimizer update: durable controller progress step `0` was written at
  `06:52 UTC`, corresponding to real optimizer update 1 and policy revision
  `f4ae14ec172784b7c6de944d6d8acda609c0ebc20206739c89a424ecb89cf925`.
  Four separate adapter hashes were recorded. All policy parity checks passed;
  the largest mean log-probability error was `.006107` and largest mean
  mismatch KL `.001169`. The update used LR `5e-6` and had grad norm `3.3625`.
  Signal was sparse in this first batch: one receiver group had non-zero
  centered counterfactual advantages, while three groups' replicas were
  identical and centered to zero. The balanced 60-update schedule must show
  that useful signal reaches all receiver slots by later milestones.
- Update-10 milestone: checkpoint 10 was complete and anonymously verified by
  the public mirror before evaluation. All four policies had received non-zero
  centered semantic advantages by update 4. The 128-row paired evaluation was
  protocol-valid (`1.0` action, broadcast, and grounding rates) and retained
  identical target-swap eligibility to baseline (`10/16` critical and `10/16`
  decoy units). On the paired eligible critical units, normal and swapped
  messages both selected the original target at `.70` at update 0 (gap `0`);
  at update 10, normal remained `.70` while target-swapped fell to `.30` (gap
  `+.40`). This is an early causal semantic-response signal: changing only the
  communicated candidate changed receiver behavior in the intended direction.
  It is not yet a coordination win. Critical normal-minus-target-swapped return
  worsened from `-.00938` to `-.05814`, normal return fell from `+.02652` to
  `+.00633`, drop lift fell from `+.00713` to `+.00142`, shuffle lift fell from
  `+.02597` to approximately zero, and critical-minus-decoy target-swap
  specificity changed from `+.03254` to `-.02739`. Interpretation: the policy
  has begun to follow message identity, but the receiver choice is not yet
  reliably converted into team return or confined to information-critical
  states. Continue the frozen run; do not tune on this development checkpoint.
- Update-20 milestone: checkpoint 20 and its four adapters were publicly
  mirrored before evaluation. All 128 paired rows completed with protocol
  validity `1.0` and unchanged target-swap coverage (`10/16` eligible critical
  units). On the same eligible units, normal target choice remained `.70` and
  target-swapped choice fell further to `.20`, increasing the paired semantic
  action gap from `0` at update 0 and `+.40` at update 10 to `+.50` at update
  20. This strengthens the evidence that the receiver is conditioning its node
  choice on the message identity. Outcome alignment is still absent: normal
  return was `+.00438`, normal-minus-drop `-.00812`, normal-minus-shuffle
  `-.00720`, normal-minus-target-swap `-.04577`, and critical-minus-decoy
  target-swap specificity `-.03818`. Thus v9 has learned a causal behavioral
  response before learning to turn that response into higher team return or
  information-specific coordination. The frozen run continues to later
  milestones without checkpoint-driven tuning.
- Update-30 milestone: checkpoint 30 was complete, publicly mirrored, and
  followed by a 128-row protocol-valid evaluation. This is the first balanced
  promising checkpoint. On the same 10 eligible paired critical units, normal
  target choice remained `.70`, target-swapped choice was `.40`, and the causal
  semantic action gap was `+.30` versus `0` at baseline. Normal return recovered
  to `+.02089` (baseline `+.02652`, update 20 `+.00438`); normal-minus-drop was
  `+.00840`; normal-minus-shuffle `+.01299`. Critical-minus-decoy specificity
  was positive for both drop (`+.04089`) and target swap (`+.02731`). The
  remaining caveat is outcome alignment under target swap: normal-minus-swap
  remained negative at `-.03051`, although less negative than updates 10 and
  20. Interpretation: message-conditioned receiver behavior, information
  specificity, and ordinary task capability now coexist, but the semantic
  intervention has not yet produced a positive terminal-return effect. The
  frozen run continues; update 30 is currently the best balanced development
  checkpoint, not a final claim.
- Update-40 milestone: checkpoint 40 was complete and publicly mirrored before
  its 128-row evaluation. This is the first checkpoint where the intended
  communication effect is simultaneously positive in receiver behavior, team
  return, and matched-decoy specificity. On the same 10 eligible paired
  critical units, normal target choice was `.70`, target-swapped choice `.20`,
  and the semantic action gap `+.50` versus `0` at baseline. Normal return was
  `+.02824` (slightly above the `+.02652` baseline); normal-minus-drop
  `+.01965`; normal-minus-shuffle `+.01667`; and, critically,
  normal-minus-target-swap became positive at `+.01907` after remaining
  negative through update 30. Critical-minus-decoy specificity was positive
  for both drop (`+.02487`) and target swap (`+.06549`). Action, broadcast, and
  grounding validity remained `1.0`; all four policy slots retained non-zero
  advantage coverage. Training behavior was capture-heavy as intended by the
  receiver curriculum (711/800 focused actions), but retained four action types
  and 37 target nodes, so there was no single-action or single-target collapse.
  Interpretation: update 40 is the first strong development result that causal
  message dependence improves action selection and terminal outcome more on
  critical than matched-decoy states. Later frozen milestones remain necessary
  to establish stability rather than selecting a transient peak.
- Update-50 milestone: checkpoint 50 was complete and publicly mirrored before
  its 128-row evaluation. The semantic action response persisted on the same
  eligible paired units: normal target choice `.70`, target-swapped choice
  `.20`, gap `+.50`. Outcome and specificity regressed from update 40, however:
  normal return `-.00005`, normal-minus-drop `-.01254`, normal-minus-shuffle
  `-.00060`, normal-minus-target-swap `-.02836`, drop specificity `+.00524`,
  and target-swap specificity `-.01513`; all protocol rates remained `1.0`.
  Interpretation: the learned message-to-action dependency is stable through
  update 50, but its conversion into beneficial team behavior is not monotonic.
  Update 40 remains the best preregistered development checkpoint so far; do
  not prefer the latest checkpoint merely because it trained longer. Continue
  the unchanged run to update 60 to measure endpoint stability.
- Final update-60 milestone: all 60 optimizer updates completed; checkpoint 60
  and all four policy adapters were publicly mirrored. The final 128-row
  evaluation completed in `933.19s` with all action, broadcast, and grounding
  validity rates `1.0`. On the same 10 eligible paired critical units, normal
  target choice was `.70`, target-swapped choice `.20`, and the semantic action
  gap remained `+.50`. Final normal return was `+.02824`;
  normal-minus-drop `+.01575`; normal-minus-shuffle `+.02034`; and
  normal-minus-target-swap `+.00693`. Critical-minus-decoy target-swap
  specificity was positive at `+.02731`; drop specificity was approximately
  neutral/slightly negative at `-.00387`. Thus the endpoint retained the core
  causal message-to-action behavior and a positive target-swap return effect,
  while the matched-decoy drop effect was not stable. Update 40 remains the
  strongest balanced development checkpoint (`+.01907` target-swap return and
  `+.06549` swap specificity), while update 60 provides a positive endpoint
  replication rather than a monotonic-learning claim.
- Preservation: public Hugging Face inspection without authentication found
  all four `264,308,896`-byte step-60 adapter files, the step-60 manifest, all
  evaluation summaries from updates 0 through 60, and final compact progress.
  W&B reported a successful sync of 4 files and 11 artifact files. No fatal,
  OOM, NaN, parity-rejection, or protocol fault occurred after launch.
- Qualitative rollout review: before decommissioning, 12 immutable training
  evidence records at steps 39, 49, and 59 were reviewed (9 critical groups,
  36 actual/swap replica pairs, and 3 ordinary groups). Receivers selected the
  certified active target in 32/36 actual branches and the swapped alternate in
  36/36 counterfactual branches; 32/36 decisions therefore changed when only
  the sender's candidate fact changed. This confirms the semantic action effect
  is visible in traces rather than only aggregate metrics. However, only 4/36
  reviewed pairs changed terminal return, and those four belonged to a group
  where the focused receiver's action itself did not change, implying a
  teammate-mediated effect and noisy focused-agent attribution. At steps 49
  and 59, all 24 critical pairs switched targets but had equal actual/swapped
  returns. Ordinary preservation traces retained non-capture behavior and
  positive returns. The compact review and record hashes are preserved in
  `results/rl_v9b_rollout_review/final_rollout_review.json`; the full 848 MiB
  token-level evidence file is intentionally not mirrored.
- Instance decommissioned: ready after `2026-08-21T13:16:36Z`; training,
  evaluation, public checkpoint preservation, and W&B sync complete.

### 2026-08-22 — v10 receiver-isolated semantic curriculum

- Status: CPU implementation and construction audit passed; GPU learnability
  smoke test not yet run.
- Diagnosis addressed: v9b changed the sender's delivered fact for the entire
  team but routed the resulting terminal contrast only to one focused receiver.
  The rollout review showed why that was insufficient: 32/36 receiver choices
  reacted to the fact, yet only 4/36 terminal returns changed, and those four
  changes were mediated by teammates while the focused action stayed fixed.
- Intervention fix: added the versioned
  `paired_receiver_target_swap` baseline. The accepted broadcast remains
  factual for the simulator and every other teammate; only the certified
  receiver's inbox receives the well-formed alternate-target fact. At the
  intervention turn, the supervisor now requires every non-receiver private
  context and decoded output to be byte-identical across branches, while the
  receiver ACT context must differ. The receiver uses the same sampling key in
  both branches, removing sampling noise from the paired contrast. Replay
  verification reconstructs the receiver-specific delivery rather than
  trusting rollout metadata.
- Reward contract: unchanged verified terminal team return only. There is no
  message reward, target-action reward, heuristic shaping, or supervised
  broadcast target. Only the receiver's factual-branch ACT tokens are trained;
  the target-swapped branch supplies a paired terminal-return baseline.
- Terminal distinguishability: added an exhaustive one-turn certificate over
  every legal joint action of the other three BLUE agents and the frozen
  balanced/aggressive/defensive opponent policies. All 12 selected directed
  sender/receiver pairs passed all 492 joint-action/style evaluations. The
  factual-target action beats the alternate-target action in every cell, with
  minimum advantage `0.1025641` and maximum `0.16`.
- Curriculum: v10 uses 10 one-turn warmup updates, 20 two-turn transfer
  updates, and 30 three-turn transfer updates. Across 60 updates it contains
  228 critical groups and 12 ordinary preservation groups, four replicas per
  group, exact left/right world balance (`114/114`), and exact sender and
  receiver policy-slot balance (`57` critical groups per slot). Matched decoys
  remain evaluation-only, preventing the optimizer from learning a decoy
  classification shortcut.
- Evaluation: the development runner now supports an explicitly bound
  receiver-only target-swap scope. The old global target-swap behavior remains
  available for reproducing v9b, and the frozen OOD data was not edited.
- Verification: all 139 Linux-independent Swarm Arena tests passed locally,
  including receiver-only inbox delivery, replay reconstruction, supervisor
  isolation checks, final-eval routing, exhaustive terminal separation, and
  all legacy behavior. The disposable macOS test environment lived under
  `/private/tmp`; no model/checkpoint artifacts were added to the Mac.
- Frozen preparation artifacts:
  `data/rl_v4/staged_curriculum_v10_4b_receiver_isolated_60.json` and its
  compact audit. Curriculum SHA-256:
  `ed6a7e1074403233c6ef78dc57137df61197f8b25ec278082bb8655f19922bfb`.
- Next action: run a short 4B rollout-only/update-0 screen and 10-update
  learnability test. Proceed to the full 60-update schedule only if the
  receiver-isolated contrast yields nonzero policy advantages and improves
  unseen-pair receiver action and terminal-return endpoints without ordinary
  regression.

### 2026-08-22 — v10 receiver-isolated 4B live launch and update-10 result

- Live run: `rl-v10-receiver-isolated4b60-d25505dc` on four NVIDIA L40S
  GPUs, source `d25505dc41485e547f06a8c217eddce6ec051897`. The trainer
  uses the checked-in rank-32 LoRA config at LR `5e-6`; three independent vLLM
  workers serve rollouts and GPU 0 trains four separate receiver-policy
  adapters. The immutable production-plan SHA-256 is
  `654f672c6c19240cb0be6c836287de4659e2400916aaf6817b479ddb664df007`.
- Exact inputs: public Qwen3-4B-Instruct-2507 revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`; public SFT revision
  `d1a55d5594c8b544121e546e14229268c8c26bae`; adapter SHA-256
  `168c9f9cdd0537660b664e9863ec9e351faf5e84d85ffbc77e95501fe1d903d2`.
  All `139/139` Linux Swarm Arena tests passed. The single 32-decision
  calibration passed with mean absolute log-probability error `0.00105693`,
  p99 `0.00138824`, mean mismatch KL `0.000148966`, and four-policy optimizer
  isolation; runtime-certificate SHA-256 is
  `59598e86fe8e3b7341c37de4eda58fe36d31335a46cb7d2cdd2ff6cd71981af3`.
- Integration fix before launch: the staged pulse process now forwards
  `--receiver-isolated-target-swap` whenever the bound production plan uses
  `paired_receiver_target_swap`. Its focused test passed `17/17`; the complete
  Linux suite had already passed. Thus training and every checkpoint pulse use
  the same receiver-only intervention rather than silently evaluating the old
  team-wide v9 intervention.
- Startup recovery: the generic tmux launcher inherited the existing tmux
  server environment and initially selected `/root/.cache/uv` for the trainer.
  This was detected while GPU 0 was still unused and before any rollout or
  optimizer update. The exact log was preserved; only the trainer session was
  replaced using the already calibrated interpreter under
  `/workspace/.uv-cache/environments-v2/prime-rl-cp3.12.3-3b50a7203453f896`.
  The plan, data, weights, reward, gates, and run identity did not change.
- Update-0 receiver-only baseline: 128/128 rows, all action/broadcast/grounding
  validity `1.0`. Critical normal return `+.01759`; normal-minus-dropped
  `+.00142`; normal-minus-shuffled `+.02230`; normal-minus-target-swapped
  `-.01526`; receiver target action `.625` normally versus `.700` under the
  swapped fact; target-swap specificity `-.00313`. This was unsaturated and
  left room for semantic receiver learning.
- Five-update learnability screen: all four independent policy slots received
  nonzero receiver-attributable advantages. Counts over 80 slot samples each
  were blue-0 `8`, blue-1 `4`, blue-2 `8`, and blue-3 `4`. The first update's
  one-slot sparsity was therefore transient sampling/within-group-centering
  sparsity, not a dead estimator. The verified factual-minus-swapped terminal
  contrast was nonzero for every receiver role.
- Update-10 checkpoint: all four adapters were uploaded to the public live-run
  mirror before evaluation. The 128-row receiver-only pulse remained fully
  protocol-valid. Normal target choice stayed `.625`, while swapped-fact target
  choice fell from the baseline `.700` to `.300`, moving the causal action gap
  from `-.075` to `+.325`. Normal return rose to `+.02322`, and
  normal-minus-dropped rose to `+.01073`. Terminal outcome alignment is not yet
  achieved: normal-minus-target-swapped was `-.02090`; swap specificity was
  only slightly positive at `+.00373`, with material per-pair heterogeneity.
  Interpretation: receiver-only RL has learned a clean content-conditioned
  behavioral response, but has not yet converted it into a reliable team-return
  gain. Continue the unchanged preregistered run to update 20; do not retune or
  claim success from update 10.
- Preservation: compact progress is mirrored every update, complete adapters
  every ten updates, and the controller logs online to W&B run
  `rl-v10-receiver-isolated4b60-d25505dc-controller-v1`. A detached recurring
  watcher checks process/GPU/log/mirror health without modifying the run.
- Update-24 controller failure: 24 atomic optimizer updates completed and the
  four stable trainer broadcasts at step 24 remained coherent, but the next
  scheduled group failed before trainer admission because its sampled sender
  message omitted the certified active candidate fact. The supervised
  `paired_receiver_target_swap` path correctly raised
  `TargetSwapIneligibleError`; no partial update 25 was applied. The pulse
  worker later timed out waiting for step 30 as a secondary consequence.
  Failure evidence was preserved under
  `audit/failures/20260822T074348Z-step24-target-swap-ineligible/`; the preserved
  progress SHA-256 is
  `5f298a0a68a33c764904cb74db1bd92556916ca24e1788d33cb6534ff8a0a226`.
  Public checkpoints 10 and 20 remain valid. The detached watcher detected the
  missing controller but only logged it, leaving resident GPU allocations idle;
  that is an operational failure, not a scientific result.
- Prospective recovery: add an explicit controller resume mode that accepts
  only contiguous durable progress, verifies all four recorded policy hashes
  against the trainer's matching stable broadcast, rebinds live serving aliases
  to those exact adapters, and resumes at the next curriculum ordinal. Archive
  incomplete step-24 rescore files and retain the failed group records. The
  recovery changes no model, optimizer, reward, curriculum, intervention, or
  evaluation setting, but its new source commit must be recorded for updates
  after the recovery boundary rather than pretending the original source alone
  executed the complete trajectory.
- Recovery retry diagnosis: the hash-verified resume reproduced the same
  ineligible sender output at the same scheduled group, proving the failure was
  deterministic. The next recovery therefore enables a bounded sender-only
  eligibility retry: only the frozen sender's BROADCAST sampling namespace is
  changed; state, prompt permutation, receiver sampling key, other agents,
  opponent, terminal reward, and curriculum ordinal remain fixed. Retries are
  hash-chained in `audit/target_swap_sender_retries.jsonl` and never condition
  on receiver action or return. This changes the estimand to receiver learning
  conditional on a defined factual handoff and cannot support a claim that the
  sender itself learned to communicate.
- Recovery completion: source `952bd50e80049e4cef9667e616790c0239c6494b`
  resumed from the four hash-verified step-24 adapters. The failed scheduled
  sender needed three bounded broadcast-only retries before producing the
  certified fact. Update 25 then committed atomically; all four rollout/trainer
  parity checks passed (maximum mean mismatch KL `0.000713372`). No further
  eligibility retry, OOM, NaN, parity rejection, or controller failure occurred.
- Final v10 result: all 60 optimizer updates and the 128-row update-60 pulse
  completed. All checkpoints `[10, 20, 30, 40, 50, 60]` are present in the
  healthy public mirror, and the controller synced four W&B files plus eleven
  artifact files. The step-60 ready-record SHA-256 is
  `afc862ba67241f5f30b6a9ddc5efd858c4d0d61fdab09bc0b4c64e60fa4d1547`.
  Final adapter SHA-256 values are blue-0 `7d26d4bf27af3fd5791bd8fa20fab50fa5cecadc63d2b6ba588eb57902150382`,
  blue-1 `aa36c3f6ed1de5d1303eefaf1433a38be38ed2860fa0c8f2ca23c780dbd592b9`,
  blue-2 `f6e4da9e76c0b55a423c6c7af1a944c71cc286837acd8b6912dea243c929be4e`,
  and blue-3 `6c8be5ba5dde74221f616cf3284be8dc8b6ce58d28c01caf881ac7042f136b8e`.
- Checkpoint trajectory (`update | normal return | normal-drop | normal-shuffle |
  normal-target-swap | normal target rate | swapped target rate | drop
  specificity | swap specificity`):

  ```text
   0 | +.01759 | +.00142 | +.02230 | -.01526 | .6250 | .7000 | +.00348 | -.00313
  10 | +.02322 | +.01073 | -.00697 | -.02090 | .6250 | .3000 | -.01240 | +.00373
  20 | +.01353 | +.01365 | -.02377 | -.05983 | .6250 | .4545 | +.00896 | -.03446
  30 | +.02903 | +.02546 | -.01642 | -.04853 | .6875 | .2000 | -.00153 | -.01176
  40 | +.04202 | +.03189 | +.02504 | -.02925 | .6875 | .2000 | +.04716 | +.04280
  50 | +.01799 | +.00419 | -.01610 | -.02650 | .6875 | .2000 | -.01644 | -.00312
  60 | -.00359 | +.00660 | -.01618 | -.01212 | .6875 | .4545 | -.01518 | -.04286
  ```
- Selection and interpretation: update 40 is the preregistered-development
  winner, not the final checkpoint. It is the only checkpoint where normal
  return, drop lift, shuffle lift, and both matched-decoy specificity measures
  are simultaneously positive; receiver target selection is `.6875` normally
  versus `.2000` under the receiver-only swapped fact. This is promising
  evidence of a transient learned message-conditioned coordination policy, but
  it is not yet confirmatory: normal-minus-target-swapped terminal return stays
  negative (`-.02925`), the development screen has only 16 critical units, and
  performance regresses after update 40. The held-out frozen evaluation must
  compare the selected update-40 checkpoint against update 0/SFT without
  selecting again on held-out outcomes. The late collapse also makes early
  stopping and broader pair coverage necessary in the next run.
- Public logging: W&B controller run
  `https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v10-receiver-isolated4b60-d25505dc-controller-v1`;
  public artifact repo `CK0607/swarm-arena-live-runs`.
- Update-40 rollout inspection before decommission: the evaluation retained
  compact matched rows rather than full text traces, while the training
  rescore requests retained token-exact prompts and completions. Decoding those
  requests in place showed no formatting or legality failure. Messages were
  grounded, actions used legal IDs, and apparent WAIT behavior was usually
  forced by a singleton legal-action set rather than unconditional action
  collapse.
- Concrete semantic behavior exists. In training pair 22, blue-3 reported that
  critical node V33 was exposed; receiver blue-1 selected `CAPTURE V33` in all
  four replicas. Replaying only the receiver with the alternative certified
  target fact reduced terminal return from `.16` to `.08`, a `+.08` factual
  semantic effect. In pair 20, blue-2 reported V30 exposed and receiver blue-3
  sometimes captured V30, but other replicas captured V32 and all had equal
  return. Pair 21 followed the communicated V68 in only two of four replicas,
  and either capture produced the same return. Pair 23 always captured the
  reported exposed V20, but swapping the fact again left return unchanged.
  Thus content-conditioned action selection is visible, while many examples
  still lack outcome-discriminating credit.
- The 16 critical development units are heterogeneous. Update 40's positive
  aggregate drop/shuffle signal is materially concentrated: pair 32
  `right_exposed`, repeat 1 has normal-minus-drop and normal-minus-shuffle
  `+.40`; pair 28's two `left_exposed` repeats each have drop lift `+.14286`.
  Counterexamples include pair 24 `left_exposed`, repeat 1 with drop lift
  `-.11765`, and both pair 33 `left_exposed` repeats with `-.125`. Several
  normal-versus-target-swapped terminal differences are zero or negative even
  when the receiver action changes. This supports selecting update 40 for a
  larger frozen held-out test, but rules out calling the development result a
  robust communication win by itself. The main bottleneck is no longer syntax
  or basic message following; it is producing broadly outcome-relevant private
  facts and stable receiver credit across maps/opponents.

### 2026-08-23 — v10 clean held-out package frozen

- Status: CPU preparation completed; GPU evaluation not yet run.
- Verdict: mechanically ready. No held-out outcome from the clean remainder has
  been observed.
- Hypothesis: the development-selected v10 update-40 four-policy checkpoint
  improves ordinary gameplay and learns outcome-relevant use of a teammate's
  private fact, beyond both the SFT initializer and generic coordination on a
  matched decoy.
- Scientific correction: the earlier v8 exploratory screen opened frozen
  handoff source pairs 4 and 17, hard cases `ordinary-hard-003` and
  `ordinary-hard-018`, and legacy seed `3000003`. The original complete suite
  is therefore not untouched. The new lock excludes all of those independent
  units, including every option order of the legacy seed, and labels the
  analysis accurately as the **clean unexposed frozen remainder**.
- Selection: update 40 was selected once from the repeated v10 development
  trajectory in source `3168b59db45bdfdb6ec3c9ece5c62a026e81b4bf`.
  Held-out results may confirm or reject it but may not select another
  checkpoint.
- Frozen inputs: 22 two-world handoff bundles, 22 hard ordinary maps, and 23
  legacy seeds under three option orders; both focal sides and base, SFT, and
  historical-league labels. The exact matrix is 4,260 deterministic games.
- Lock: `data/rl_v4/v10_clean_holdout_lock.json`, canonical body SHA-256
  `1a5bb75f165cbf320e9f9761064d2baa9d000533c537ad367e45a7518a9ffb32`.
  It binds both frozen file and manifest-body hashes, the immutable Qwen base
  revision, SFT revision/adapter hash, all four update-40 adapter hashes, the
  public artifact revision, and every opponent revision.
- Matrix hashes: config
  `dbcbaa632cfc4cbf87921e96c4efbab83c1823d30e307eaf23d18de1daf6bbe2`;
  ordinary case IDs
  `fbd539e8fe1e527073881978ca375e265f39f931ee013be08c377577872dc95e`;
  handoff case IDs
  `6fd4623074c811fe5233e8e015cc7966b551fabefe8236031be0a2f861faccca`.
- Semantic estimand: intention-to-treat. A target-swap row whose sender omits
  the active fact remains an eligible evaluation observation with a
  zero-strength intervention; it is never retried or discarded. The swap is
  applied only to the designated receiver.
- Predeclared semantic endpoints: candidate critical normal-minus-swapped
  return; candidate-minus-SFT semantic lift; critical-minus-matched-decoy
  specificity; receiver target-action gap; eligibility; and per-opponent
  effects. Existing dropped, shuffled, delayed, zero-budget, capability,
  protocol, regression, and collapse endpoints remain in the final summary.
- Reproducibility: the evaluator is resumable and writes raw traces before
  compact rows with `fsync`. Its mirror uploads a stable compact-row prefix and
  immutable compressed raw shards every 100 rows. It matches raw records by
  evaluation ID, so a crash-created orphan or duplicate trace cannot misalign
  resumed shards. Each public commit is anonymously downloaded and SHA-256
  checked before mirror state advances. Logs, credentials, caches, and model
  weights are excluded from this mirror.
- CPU verification: Ruff passed; all affected modules compiled; focused tests
  passed `25/25` in `6.97s`; the fail-closed audit reproduced the exact counts
  and hashes above. A dedicated test simulates an orphan raw record and verifies
  correct resumed sharding.
- Post-evaluation diagnostics are also frozen: a balanced 32-decision legal
  constraint probe, per-policy candidate-to-SFT KL, and a raw-trajectory
  collapse audit. Limits are speaking extremes `.02/.98`, action/message-target
  concentration `.95` after at least 20 observations, constrained KL mean
  `.08`, and p99 `.30`. The public mirror will not declare completion until
  these three artifacts are present and anonymously verified.
- GPU estimate: one 4xL40S inference node, approximately 9 hours plus setup and
  final publication; provider auto-termination at 11 hours. At `$2/hour`,
  expected cost is `$18–22`. No GPU is currently active or accruing cost for
  this preparation.
- Operational runbook: `V10_CLEAN_HOLDOUT.md`. The GPU host prepares artifacts
  by anonymous immutable downloads and verifies the SFT plus all four candidate
  adapter hashes before any request. Completion requires 4,260 rows, summary
  and `COMPLETE`, a healthy fully caught-up public mirror, and an anonymously
  verified final revision.
- Limitation fixed in advance: the `historical_league` label intentionally uses
  the same frozen SFT adapter revision as the v10 preregistration. It is not an
  independent historical policy family, so results must not claim three unique
  opponent policies.
- Next action: commit and publish this frozen preparation, then request one
  fresh 4xL40S SSH endpoint and execute the once-only evaluation without
  changing the lock or selecting on held-out outcomes.

### 2026-08-24 — v10 clean held-out update-40 result and post-audits

- Status: completed.
- Verdict: promising causal message-use signal; the strict confirmatory
  RL-specific communication claim is rejected narrowly. Protocol, constrained
  KL, and collapse diagnostics pass.
- Hypothesis: the development-selected v10 update-40 policies use a teammate's
  private target fact to improve receiver behavior and terminal return, beyond
  SFT and beyond generic coordination on matched decoys.
- Decision unlocked: preserve update 40 as positive but non-confirmatory
  evidence. The next run should improve critical-versus-decoy specificity and
  ordinary capability rather than extend the same curriculum blindly.
- Source commit: `54f325284f944237031de218db077c0615984f8f` for the
  clean held-out runner; the selected policies were trained from v10 source
  `3168b59db45bdfdb6ec3c9ece5c62a026e81b4bf`.
- Base / adapter / opponent revisions: Qwen3-4B base `cdbee75f`; SFT revision
  `d1a55d55` with weight SHA-256 `168c9f9cdd0537660b664e9863ec9e351faf5e84d85ffbc77e95501fe1d903d2`;
  four update-40 policy hashes are frozen in the lock and result manifest. The
  historical-league label intentionally resolves to the preregistered SFT
  revision and is not claimed as an independent policy family.
- Data split and manifest SHA-256: clean held-out lock body
  `1a5bb75f165cbf320e9f9761064d2baa9d000533c537ad367e45a7518a9ffb32`;
  4,260 compact rows SHA-256
  `d599e0ad15d532df8f00f78f1dc278566c5ae9563f0f9252e29aef87d1b20c6f`;
  reconstructed raw rows SHA-256 begins `dd05d07c` and contains exactly 4,260
  evaluation IDs.
- GPU, wall time, and estimated cost: the once-only evaluation used four L40S
  GPUs from 2026-08-23 until provider auto-termination; the exact billed amount
  is not available in the artifacts. The independent post-audit reproduction
  used one RTX 3090 from approximately 04:55–05:16 UTC on 2026-08-24; its exact
  provider rate is unknown. No GPU is required to inspect the compact result.
- Exact launcher/config: `V10_CLEAN_HOLDOUT.md`; config SHA-256
  `dbcbaa632cfc4cbf87921e96c4efbab83c1823d30e307eaf23d18de1daf6bbe2`.
- Predeclared gates: semantic normal-minus-receiver-target-swap; RL-minus-SFT
  semantic lift; critical-minus-decoy specificity; receiver target-action gap;
  per-opponent direction; ordinary/capability regression; action, broadcast,
  and grounding validity; speaking/action/target collapse; candidate-to-SFT
  constrained KL mean at most `.08` and p99 at most `.30`.
- Results: candidate semantic normal-minus-target-swapped terminal return
  `+.0320811`, 95% CI `[+.0157342,+.0506818]`, `n=22`. Receiver target action
  was `.7803` normally and `.2121` when only the receiver's fact was swapped;
  paired gap `+.5682`, 95% CI `[+.3939,+.7348]`. The direction was positive
  against every labelled opponent. Sender target-fact and intervention
  eligibility were both `.8826`. RL-minus-SFT semantic sensitivity was
  `+.0194063`, 95% CI `[-.0003998,+.0386262]`; critical-minus-matched-decoy
  specificity was `+.0131639`, 95% CI `[-.0022569,+.0290604]`. Delayed and
  shuffled messages caused significant degradation; dropped and zero-budget
  intervals crossed zero. Overall RL-minus-SFT gameplay was `+.005213`, 95% CI
  `[-.005165,+.015588]`. Every defined protocol metric was `1.0`.
- Post-audit results: the frozen 32-decision probe produced 1,339 token rows
  with 154 branching tokens. Candidate-to-SFT KL mean was `.0013960` and p99
  `.0027070`, passing both frozen limits. One isolated token reached `1.5688`
  and remains reported. The full 4,260-trajectory collapse audit passed: policy
  speaking rates `.2947–.3132`, action concentration `.2549–.2856`, message
  target concentration `.0394–.0484`, zero orphan rows, and every registered
  collapse flag false.
- Failures and retries: the first clean audit environment lacked PEFT. A generic
  temporary resolver attempted to pull a separate CUDA 13/Torch stack and was
  stopped before model loading; the successful audit used a no-dependency
  `peft==0.20.0` / `accelerate==1.13.0` overlay on the frozen Torch environment.
  The next load exposed a real export bug: public candidate configs declared
  rank 16 / alpha 32 although all immutable tensors and the frozen trainer
  config establish rank 32 / alpha 64. The repair utility created
  metadata-corrected views, symlinked the unchanged weights, verified every
  tensor rank/target module/hash, and retained a manifest. No scientific
  threshold, case, checkpoint, weight byte, or held-out row changed.
- Artifact paths and hashes: compact bundle
  `results/v10_clean_holdout_u40/`; summary `13c7f787e3930d0893f52b1f26290ccd63ca49b00b722aedbd7f0ecd7c39aee2`;
  probe `db525f1201775febc4fb8933758c16e4821fd7e1d2e9a524db0a650cc7142edb`;
  KL `09673097fcef7b8e30106f58aafcef77020bdab51bf1ab7bbb4e2c15bd3008ae`;
  collapse `54f72bd078f2371e0be52f4f57aa151b7876a86fc5b03cc4d397c7af67025be6`;
  metadata repair `91243fadb80cb7441e23a9353610f34a0f8a2aeb5b687676e35aa49c0847b3b4`.
  Public held-out rows and compressed traces:
  `https://huggingface.co/CK0607/swarm-arena-live-runs/tree/main/runs/rl-v10-clean-holdout-u40`.
- Interpretation: this is real evidence that the learned policies condition
  receiver behavior and return on teammate message content. It is not yet
  evidence that RL created a broadly communication-specific capability: both
  decisive RL-over-SFT and critical-over-decoy confidence intervals narrowly
  include zero, and overall gameplay improvement is not significant.
- Next action: train on counterfactual units where the critical receiver action
  changes terminal outcome but the matched decoy does not; maintain ordinary
  anchors and select on a joint semantic-specificity/capability criterion.
- Instance decommissioned: the original 4xL40S evaluation node auto-terminated;
  the RTX 3090 audit node was idle and decommission-ready after these compact
  artifacts were copied. No model weights or raw traces were copied to the Mac.

### 2026-08-24 — v11 diverse receiver scale-up preparation

- Status: CPU preparation completed; no GPU run started.
- Verdict: mechanical and construction pass. This is a launch candidate, not a
  learning result.
- Hypothesis: the positive v10 message-use effect can become RL-specific and
  critical-specific when training covers many independent topologies and
  retains ordinary capability, instead of repeating twelve small maps.
- Decision unlocked: request a four-GPU node for runtime binding and the
  180-update scale-up after this preparation commit is public.
- Source commit: `e15b6388`.
- Base / adapter / opponent revisions: Qwen3-4B base `cdbee75f`; SFT initializer
  `d1a55d55` / weight SHA-256 `168c9f9c...`; historical opponent is the
  development-selected v10 update-40 blue-0 adapter, weight SHA-256
  `4afed7f7...`. V11 still initializes from SFT, never from v10 RL.
- Data split and manifest SHA-256: combined train/development manifest body
  `41c6caa00b7d2854c3667058feb52256dbeff67515c7976636c5636e38b5f364`;
  frozen manifest body
  `6a6ad15bd7619e390587bff26b0b2ad1cf54d1065a53884e3616a98f7023c50c`;
  curriculum file SHA-256
  `99f3efa8bad862d661a1ae9c9fae84336473d700768afb4866465577c0bc0864`;
  schedule SHA-256
  `1f4a518bae9551572b3c3ec81d322b06cb437163f20cfcc33b539cfdeac5a458`.
- GPU, wall time, and estimated cost: no GPU used. Local deterministic manifest
  generation took about 16 seconds and the exhaustive audit about 11 seconds.
  Based on v10 throughput, the eventual 180-update 4xL40S run plus scheduled
  evaluation is estimated at 18–22 hours; refine after the first 20 updates.
- Exact launcher/config: trainer
  `configs/rl_v11_4b_diverse_receiver_180.toml`; base plan
  `configs/rl_v11_4b_base_plan.json`; curriculum and evaluation design under
  `data/rl_v11/`; operational notes in `V11_DIVERSE_SCALEUP.md`.
- Predeclared gates: the headline evaluation is limited to semantic message use,
  RL-over-SFT semantic lift, critical-over-decoy specificity, receiver target
  switching, and ordinary/overall capability. Drop/delay/shuffle/zero-budget
  remain reported diagnostics rather than four extra promotion gates. Protocol,
  constrained KL, collapse, and artifact completeness remain mechanical
  requirements.
- Results: 96 independent training bundles, 24 independent development bundles,
  and 36 independent frozen bundles. Every directed sender/receiver role has
  eight training bundles and covers sizes 12/14/16/18/20 plus all seven source
  horizons. The 180-update schedule contains 540 critical and 180 ordinary
  groups, producing 2,160 focused receiver replicas and 720 ordinary replicas.
  Every policy is receiver and sender in exactly 135 critical groups; both
  latent worlds appear exactly 270 times.
- Construction audit: passed 3,870 exhaustive legal teammate-joint-action and
  three-opponent-style comparisons over all 96 training bundles. The global
  minimum factual-target terminal advantage is `.08`, all critical comparisons
  are strictly positive, and every matched decoy advantage is exactly zero.
  Private worlds are indistinguishable without the message, legal actions do
  not change, and train/development/frozen state hashes are disjoint.
- Failures and retries: the first generated schedule failed its receiver balance
  check because pair-major left/right ordering skewed stage boundaries. The
  second schedule balanced receiver slots but the exact audit caught a 240/300
  left/right skew. The final ordering uses complete four-receiver blocks with
  complementary latent worlds; it passes exact 135-per-policy and 270/270 world
  balance. Failed generated schedules were never admitted or committed.
- CPU verification: deterministic regeneration matched every builder-owned
  artifact; Ruff passed; focused tests passed `18/18` in `7.36s`; both the
  structural data audit and exact communication-learnability audit pass.
- Artifact paths and hashes: `data/rl_v11/`,
  `V11_DIVERSE_SCALEUP.md`, and the trainer/base-plan configs. Total new data is
  about 1.6 MB; no model, checkpoint, or rollout artifact was stored on the Mac.
- Interpretation: this removes the clearest data-design confound and scales the
  number of training topologies eightfold while quadrupling ordinary anchors.
  It guarantees clean learning opportunity, not that sampled 4B rollouts will
  exploit it. Development and frozen outcomes remain unseen.
- Next action: commit/publish, then on a fresh 4xL40S node download and
  hash-check the base/SFT/v10 opponent, repair only the historical adapter's
  stale metadata view, create a fresh runtime certificate, bind the production
  plan, start public mirroring at update zero, and launch.
- Instance decommissioned: not applicable; no GPU instance was used.

### 2026-08-24 — v11 independent launch-route recheck

- Status: completed; no GPU run started.
- Verdict: two CPU integration defects were found and fixed before they could
  consume paid compute. The regenerated curriculum and exact learnability
  result remain unchanged.
- Hypothesis: a deterministic data package is not launch-ready unless the paid
  preflight and the semantic selection/frozen evaluators can consume that exact
  package without falling back to v4 filenames or endpoints.
- Decision unlocked: v11 is eligible for host-specific runtime calibration and
  production-plan binding after this fix is committed and published.
- Source commit: `cf98c342` for the launch/evaluator fix; the following
  documentation-only commit records that immutable identifier.
- Data split and manifest SHA-256: v11 task index file
  `544bb787ac496a393ae442ce6621e47dc201ed29c789e39deaeba75728eaee2f`;
  separate development handoff file
  `52591c48c6a36ad01643720fcb5d7935aea11cea9a36f3902aed37ef7458189f`;
  progress-eval design file
  `0001bee19eba3c3ffc322fd31cf7e1126a6ff9d8bc85af577d3304547e7eff81`.
- GPU, wall time, and estimated cost: no GPU used and no paid node active. The
  CPU recheck took minutes. Retain the prior 18–22 hour 4xL40S estimate and set
  a 22-hour initial termination deadline, then revise it from measured update-20
  throughput.
- Exact launcher/config: unchanged v11 trainer/base plan and
  `scripts/launch_staged_rl.sh`; it must receive `SWARM_DATA_DIR=data/rl_v11`.
- Results: independent regeneration matched every builder-owned byte. The v4
  production binding now resolves distinct train/development/final hashes; the
  development evaluator expands global bundle IDs 96–107; the frozen evaluator
  expands all 36 independent handoff bundles; both selection and frozen routes
  include the receiver-only `target_swapped` endpoint and semantic summary.
  Exact curriculum audit still passes 3,870 legal joint-action/opponent-style
  comparisons with minimum advantage `.08`, balanced 135-per-policy receiver
  allocation, and 270/270 latent worlds.
- Failures and retries: the first recheck found that v11 omitted the v4-compatible
  `index.json`, `curriculum.json`, and `handoff_development.json`, so paid
  preflight would have failed before update zero. It also found the generic
  selection runner omitted the primary target-swap endpoint and defaulted the
  frozen route to 24 rather than the declared 36 handoff bundles. These were
  integration bugs, not model/data failures. A sandboxed test-tool download
  initially failed DNS and was repeated with the approved pinned-tool network
  path; the first test import then exposed a missing test-only
  `huggingface-hub` dependency, which was added only to the ephemeral uv test
  environment.
- CPU verification: deterministic byte comparison passed; exact audit matched
  the committed result apart from its expected path string; Ruff passed; 40
  focused production-binding, schedule, evaluator, mirror, and semantic tests
  passed in 11.18 seconds.
- Artifact paths: `data/rl_v11/index.json`,
  `data/rl_v11/handoff_development.json`, `data/rl_v11/curriculum.json`, and the
  updated builder/evaluator/tests. No weights, checkpoints, raw traces, or uv
  caches were added to the repository or retained as Mac project data.
- Interpretation: this second pass was necessary and useful: without it the
  GPU launch would have stopped at preflight, and a later evaluation could not
  have tested the stated causal claim. It does not add evidence that RL learns;
  that remains the purpose of the 180-update run.
- Next action: publish the fix, then request one 4xL40S/4xL40 node for exact
  runtime calibration and the v11 training run.
- Instance decommissioned: not applicable; no GPU instance was used.

### 2026-08-26 — V11 conclusion and V12 counterfactual-robustness CPU freeze

- Status: V11 completed/rejected; V12 CPU preparation completed; no V12 GPU run
  started.
- Verdict: V11 is promising mechanistic evidence but failed its exact
  development selector. V12 is a mechanical/construction pass, not a learning
  result.
- Hypothesis: V11 learned real message-conditioned receiver behavior but became
  too literal and slightly regressed on legacy ordinary play. Directly training
  the misleading-message receiver branch on privately observable matched decoys,
  while front-loading ordinary terminal-return groups, can repair both defects
  without an additive communication reward.
- Decision unlocked: request a fresh four-GPU Linux node for the update-0/1
  smoke and, only if distinct-adapter loading plus nonzero challenge credit pass,
  continue the 160-update V12 run.
- Source commit: V12 implementation `80d67a047b64de371c8aab728f39467480ff7f9a`.
- V11 result being acted on: update 60 passed semantic return `+.03755064`,
  specificity `+.01456961`, and hard ordinary `+.00562644`, but failed legacy
  ordinary at `-.04781285`; update 120 failed specificity `-.00936699` and
  legacy `-.01713771`; update 180 passed semantic `+.04534265`, specificity
  `+.02927313`, and hard ordinary `+.03410836`, but failed legacy ordinary at
  `-.00938840`. No V11 checkpoint was selected and frozen evaluation was never
  opened. Protocol validity remained `1.0`; rollout diagnosis found
  over-literal response to swapped teammate targets rather than formatting or
  action collapse.
- Initializer decision: by explicit user direction, V12 continues from all four
  separate public V11 update-180 policy adapters rather than repeating SFT
  learning. This is a warm-start choice, not retroactive V11 admission. New
  trainer/controller support binds one path/hash to each `run_blue_0..3` and
  fails closed if any policy is missing, cloned, corrupt, or mismatched.
- Reward and credit design: reward remains independently replay-verified terminal
  team control delta only. Critical handoffs train the factual receiver ACT
  branch on centered `factual - target-swapped` return. Matched decoy challenges
  train only the misleading-message receiver ACT branch on centered
  `target-swapped - factual` return. There is no message, truthfulness, target,
  action, capture, validity, or oracle bonus.
- Curriculum: 160 updates / 640 groups: 260 ordinary retention, 220 factual
  critical handoffs, and 160 matched challenge decoys. Critical receiver counts
  are exactly 55 per policy; challenge counts are exactly 40 per policy. The
  first 40 updates use two ordinary groups per update; later stages alternate
  retention-heavy and transfer-heavy batches at one-, two-, and four-turn
  handoff horizons.
- Opponents: every update contains base, SFT, V10 historical, and current-policy
  opponents once each. Scenario-kind/family counts are decorrelated to maximum
  deviation two from exact balance: ordinary 64/65/66/65, critical
  54/55/55/56, challenge 42/40/39/39 across the four families.
- Data split and hashes: fresh 36-bundle development file
  `46046fdeb4e4d121fdc235b812044d2ced5d4e8d5f8b36a57a692ff323817931`;
  fresh 36-case ordinary development
  `7041ccc936dcae100695ec43c054657392f42a6540a77d9f7ac62321ae292320`.
  The never-opened V11 frozen handoff and ordinary files are reused byte-for-byte
  at `c2ae30275fb626688c971a9e807dabea4047b6faf6a94edb0ac82fb2ba1714b5`
  and `544998349b3f7b24ce5ef5ec44cd8c0f28ee5f6e5a072b68d8380d79c2e830b9`.
  Curriculum file SHA-256 is
  `b0d7b69b9b7413ce85c13d3edecf1ee24954b809b8a5cef9ef8ab31af6701c2f`;
  schedule SHA-256 is
  `3bd30f26d3ba448fc2eb86ae714fed84bd8103cc3fe9501d34be608f83c4d756`.
- Selection and fail-fast rules: candidates are 20/40/80/120/160. The earliest
  candidate requires positive semantic mean, positive critical-minus-decoy
  specificity mean, and clustered 95% lower bounds at least `-0.02` for both
  legacy and hard ordinary return. The non-inferiority margin is fixed before
  V12 observations and is not retroactive to V11. Stop after update 40 only if
  updates 20 and 40 both show no ordinary-retention improvement, no specificity
  improvement, and non-positive semantic return. Frozen runs once after formal
  selection.
- Exact configs: trainer
  `configs/rl_v12_4b_robust_communication_160.toml` (rank 32, LR `3e-6`,
  checkpoint interval 20); plan `configs/rl_v12_4b_base_plan.json`; data and
  evaluation design under `data/rl_v12/`; operational rationale in
  `V12_COUNTERFACTUAL_ROBUSTNESS.md`.
- CPU verification: deterministic regeneration matched every builder-owned byte;
  task binding and 640-group plan load passed; exhaustive learnability audit
  passed over all 96 training pairs; all train/development/frozen state hashes
  are disjoint; every challenge is a matched critical case; legal actions are
  unchanged; decoy worlds are privately distinguishable; critical worlds are
  privately indistinguishable; 24 lightweight selector/plan/warm-start/fail-fast
  tests passed in `0.33s`; Ruff and compileall passed. Local TrainerConfig
  validation of the new per-run mapping passed. Full Torch/xgrammar/vLLM tests
  are deliberately deferred to the next Linux node because installing that
  stack on the Mac would waste local storage.
- Failures and retries: the first challenge schedule covered all policies but
  gave decoy counts 40/48/33/39; rotating the final-stage case order produced
  exact 40/40/40/40. The original shuffle seed correlated scenario type with
  opponent family by as many as 14 groups; a pre-result seed search reduced the
  maximum deviation to two. The local checkout lacked the pinned
  `pydantic-config` submodule; the exact 460-KB submodule was initialized instead
  of installing the full CUDA stack. Failed schedules were never admitted.
- GPU, wall time, cost, and storage: no GPU used and no pod cost incurred for
  this preparation. CPU generation/audits/tests took minutes. Expected training
  is roughly 12–16 four-L40S hours, with a paid stop decision at update 40;
  development/frozen duration is measured separately after checkpoint
  selection. Only about 2 MB of manifests/code were added to the Mac; no model,
  checkpoint, rollout trace, or CUDA environment was copied locally.
- Interpretation: V12 now supplies the missing negative credit on the exact
  misleading-message receiver action while retaining V11's learned communication
  behavior and separate policy identities. Construction guarantees a clean,
  balanced opportunity to learn; it cannot guarantee the 4B policy samples a
  nonzero advantage. That is the only substantive GPU-side unknown before the
  long run.
- Next action: on Linux, anonymously download/hash all four public V11-u180
  adapters, run `prepare_v12_distinct_warmstart.py`, execute full tests and one
  update-0/1 challenge smoke, verify every policy slot receives nonzero challenge
  coverage plus trainer/serving parity, then start public mirroring before any
  long optimizer run.
- Instance decommissioned: not applicable; no V12 instance was rented.

### 2026-08-26 — V12 absolute paired-contrast correction

- Status: CPU implementation and audit completed; no V12 GPU run started.
- Verdict: mechanical pass. This supersedes only the V12 credit-centering
  choice recorded above; the V11 evidence, V12 data split, schedule, reward,
  policy warm start, selection rule, and unopened frozen suite are unchanged.
- Hypothesis: replica-mean centering can erase precisely the uniform behavior
  V12 must repair. If all four factual receivers succeed equally, or all four
  misleading-message receivers fail equally, their paired effects are equal
  and the centered advantages become zero. Preserving the raw paired terminal
  return difference supplies the missing positive/negative signal without an
  additive reward.
- Decision unlocked: V12 may proceed to its Linux update-0/1 smoke using
  absolute paired contrasts. A long GPU run remains prohibited until the smoke
  proves nonzero challenge credit for every policy slot and passes the existing
  distinct-adapter, replay, legality, parity, and artifact checks.
- Source commit:
  `68fdc1835d43d2e0bc703777ba6874eb88ab13a6`.
- Implementation: `SharedReturnSpec` and `ProductionPlan` now bind
  `paired_contrast_centering` as either legacy `replica_mean` or `none`.
  Defaults are omitted from immutable hashes, preserving every completed-run
  identity. V12 alone sets `none`; critical groups route
  `factual_return - swapped_return` to factual receiver ACT spans, and challenge
  groups route `swapped_return - factual_return` to misleading receiver ACT
  spans. Ordinary groups explicitly revert to leave-one-out return credit and
  legacy centering metadata.
- Uniform-signal audit: factual returns `(0.2, 0.2, 0.2, 0.2)` versus swapped
  returns `(-0.2, -0.2, -0.2, -0.2)` retain four `+0.4` critical advantages;
  the reverse retains four `-0.4` challenge advantages. Under the old default,
  the same uniform effects remain centered to numerical zero, proving backward
  compatibility and the repaired V12 distinction.
- Data/config identity: the 640-group schedule SHA-256 remains
  `3bd30f26d3ba448fc2eb86ae714fed84bd8103cc3fe9501d34be608f83c4d756`.
  Updated curriculum file SHA-256 is
  `3898680505b61f60591167d3d6b41776433846fd2d97848399a01e2e40bf27c7`;
  production-plan file SHA-256 is
  `725862d1ed4d9d8f0857868241aa641415241e13aab739277f3a6e7364702556`;
  semantic production-plan SHA-256 is
  `5abab026b99fb5ee6f0a1554438441ed7f2309a7645ed9745dbbae756ffb8062`.
  Builder audit SHA-256 is
  `05d41f6955084f91562e46873f1a71374b1d49012a74e9e807efe3125e68059c`;
  exhaustive learnability-audit SHA-256 is
  `d734200ac9d9a42d711624a61aa0b040b7e452224a1436d5f237a71e7ab6c34f`.
- CPU verification: deterministic regeneration matched every builder-owned
  artifact byte-for-byte; 26 dependency-light production, task-binding,
  selector, and fail-fast tests passed in `0.16s`; the pure estimator/hash
  smoke passed; Ruff, compileall, and `git diff --check` passed. The full
  receiver/challenge integration module remains a mandatory Linux gate because
  importing it requires the intentionally Mac-absent Torch/live rollout stack.
- Reward/safety interpretation: this is a paired counterfactual control
  variate over independently replay-verified terminal team return, not a
  truthfulness, obedience, target, capture, or communication bonus. The
  counterfactual branch is never relabelled as oracle supervision. Uniform
  negative challenge outcomes can now reduce the sampled misleading ACT
  behavior rather than disappearing from the optimizer batch.
- GPU, wall time, cost, and storage: no GPU allocated and zero GPU cost. Only
  compact code/config/audit changes were added; no model, checkpoint, rollout,
  CUDA environment, or large cache was stored on the Mac.
- Next action: on a fresh four-GPU Linux node, anonymously hash all four V11
  update-180 adapters, generate the distinct warm-start binding, run the full
  suite, and execute one critical plus challenge rollout/update smoke. Inspect
  signed per-policy advantages before authorizing the 160-update trajectory.
- Instance decommissioned: not applicable; no V12 instance was rented.

### 2026-08-26 — V12 paid Linux gate started and caught rollout integration failure

- Status: setup and Linux validation in progress; no inference server,
  `torchrun`, optimizer update, or V12 model result yet.
- Verdict: operational recovery plus genuine code-gate failure. The paid gate
  prevented a live run whose paired/focused rollout construction would have
  crashed before inference.
- Hypothesis: unchanged from the V12 absolute paired-contrast plan. This entry
  records runtime readiness only and makes no learning claim.
- Decision unlocked: fix the single branch-classification ordering defect,
  publish the corrected source, and require the entire Linux suite to pass
  before model download, serving, parity certification, or optimizer smoke.
- Source at provisioning: `ee3f8323e7a0c3bf29848e7e81c35b694191e567`.
  Narrow source correction
  `ff14648a5bb752a09fab1fa9900cfcab9a8f73b2` moves `branch_kind`
  construction before its `collect_training_samples` validation; scientific
  configs, data, hashes, reward, gates, and curriculum are unchanged.
- GPU allocation: Lium pod `lunar-fox-bc`, immutable pod ID
  `5d5f4e96-8d14-4832-b5dc-3f6df8f5d5a4`, 4x L40S with 49,140 MiB per GPU,
  283 GiB RAM, and about 2.3 TiB free disk. Price is `$1.52/hour`; a 39-hour
  hard TTL caps scheduled spend at `$59.28`, below the user-authorized `$60`
  maximum. Removal is scheduled for `2026-08-28T04:34:07.509466` as reported
  by the provider. All four GPUs were idle during setup/testing.
- Setup evidence: the first public clone paused because two submodules use
  `git@github.com:` URLs on a credential-free host. The blocked setup pane and
  log were preserved, then only setup was restarted with an HTTPS URL rewrite;
  no GitHub credential was copied. The frozen environment installed under
  `/workspace/.uv-cache` with uv `0.12.6`, Torch `2.11.0+cu128`, CUDA visible on
  four GPUs, and FlashAttention `2.8.3`.
- Setup retry: one verification command omitted the dedicated `UV_CACHE_DIR`,
  started a redundant uv resolution, and repointed `.venv` to the secondary
  cache. That exact resolver was terminated, the frozen sync was rerun with
  `/workspace/.uv-cache`, and `.venv` again points at the canonical
  environment. The training skill now records both this cache requirement and
  the public-submodule HTTPS rule.
- Test evidence: the first pytest command omitted the required experiment-local
  `PYTHONPATH` and failed collection with 21 import errors; its log is retained
  as invocation evidence. The unchanged corrected command collected the suite
  and produced **155 passed, 8 failed** in 44.48 seconds. All eight failures are
  the same `UnboundLocalError`: `rollout_branch` checked `branch_kind` before
  assigning it. Existing tests cover ordinary focused credit, paired message
  drop, factual target swap, receiver-only target swap, and the V12 challenge
  branch, so this is a real live-path blocker rather than a missing test.
- Post-fix validation: Ruff passed on the changed module and the entire same
  Linux command passed **163/163 tests** with two import deprecation warnings in
  40.48 seconds. No test, threshold, data file, or scientific config was removed
  or relaxed to obtain the pass.
- Launch-route recheck then found that the generic staged launcher did not pass
  V12's `swarm-distinct-policy-warmstart-v1` manifest to the controller. The
  trainer had the correct four per-run paths, but the controller would otherwise
  have initialized serving aliases from the common SFT path. Correction commit
  `9779f3976ce77ff59c15a85e8e50c51f6a68e6e6` makes preflight validate the
  controller manifest against all four trainer paths, hashes, revisions, and
  unique adapter bytes; passes the same file to the controller; includes it in
  the safe mirror allowlist; and preserves the dedicated uv cache in all tmux
  children. The actual generated V12 manifest passed this check at SHA-256
  `67ae25d29e5b8b2be0d8536e93e0e1fabc765f77df060b0e4f26028fd0595190`.
  Bash syntax and Ruff passed, and the complete Linux suite again passed
  **163/163 tests** in 40.82 seconds. No inference or optimizer process had
  started when this gap was found.
- Public input preparation: the base Qwen3-4B, SFT baseline, V10 historical
  opponent, and all four V11 update-180 policy adapters were downloaded
  anonymously and hash-checked on the pod. The public V11 manifest SHA-256 is
  `b4e1559d23fac93869b0ed47d2c6c11b029e95b4b734bdf7e97959c9113e2c75`;
  the distinct V11 policy hashes are `2b6078b3...b304`, `2b1338f2...53b3`,
  `2aae96b4...4a3f`, and `a03e6ce8...e29d`. Generated V12 trainer TOML SHA-256
  is `aba301c2c60bdba34c3a01dc9acf861df45c5bac0b5ad3c5166fe3e028bd51b9`.
- Artifact/storage status: only source, test logs, and the CUDA environment are
  on the paid pod. No base model, SFT/V10 opponent, V11 warm-start adapter,
  rollout, checkpoint, or model artifact has been copied to the Mac. Public HF
  mirroring is not yet enabled because optimizer launch remains prohibited.
- Next action: publish the narrow source correction, update the pod to that
  immutable launch-route commit, then proceed to inference/runtime parity and
  the update-0/1 signed-credit smoke only because all 163 Linux tests passed and
  all four distinct public adapters matched.
- Instance decommissioned: no; pod is active under the bounded TTL.

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
