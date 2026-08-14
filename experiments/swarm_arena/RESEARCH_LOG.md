# Swarm Arena research log

Last updated: 2026-08-14 17:54 IST  
Branch: `exp/swarm-arena-4b`  
Message-estimator implementation checkpoint:
`567bc1393d101ca9f4a9613cabececece09a2399`  
Status: environment and pre-RL infrastructure are mechanically ready; the new
message-edge credit estimator is awaiting live GPU validation. No RL optimizer
step has been admitted from the counterfactual audits.

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

## Current decision gates

Before the first real RL step, all of the following remain required:

- full Linux and live vLLM smoke passes at the exact public commit;
- actual and four message-drop branches replay exactly;
- intervention changes only the named first-turn sender delivery;
- owned optimizer batches contain exactly four first-turn broadcast spans and
  no action spans;
- empty-message invariance is exact;
- the 24-scenario critical/decoy audit shows intended-sender localization rather
  than broad off-role effects;
- constrained rollout/trainer parity is recertified for the new token set;
- model-controlled opponent identities and revisions are immutable per epoch;
- collapse monitors and frozen development/regression gates are active;
- all inputs and outputs required for reproduction are anonymously public.

If localization fails, the correct result is another rejection—not an additive
communication reward. If it passes, the next experiment is a short,
rollback-safe, stage-one RL pilot with no final-evaluation access.

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
- Frozen message-credit admission plan:
  `MESSAGE_CREDIT_AUDIT_PLAN.md`

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
