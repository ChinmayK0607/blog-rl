# Symbolic tool-calling environment — sample task

A real medium task from the eval set: `stc-56c8893db64c3c6c`
(depth 5, `distractor_ratio=0.4`, `recovery_cost=3`, optimal plan length 10).

## World map

Solid arrows are the solution corridor; dashed arrows are dead-end distractor
branches the agent must recognize and back out of.

```mermaid
flowchart LR
    R0["room_00<br/>start"] -->|east| R1["room_01<br/>🔑 key_459"]
    R1 -->|east| R2["room_02<br/>🎚 switch_337"]
    R2 -->|east| R3["room_03"]
    R3 -->|east| R4["room_04<br/>🖥 terminal → code 517603"]
    R4 -->|"east 🔒"| R5["room_05<br/>🎯 vault_447"]
    R0 -.north.-> D3["diagnostic_03<br/>dead-end ×3 → scrap"]
    R1 -.south.-> D2["diagnostic_02<br/>dead-end ×3 → scrap"]
    R2 -.north.-> D0["diagnostic_00<br/>dead-end ×3 → scrap"]
    R2 -.south.-> D1["diagnostic_01<br/>dead-end ×3 → scrap"]
```

## Solve state-machine

The reward is an exact function of hidden state, so the task *is* a state
machine — the agent drives it from `start` to `Verified` through an ordered set
of gates. The final east exit into the vault is locked until **both** the switch
is active **and** the key has released the lock, and a submission only verifies
if the code was actually queried from the terminal first.

```mermaid
stateDiagram-v2
    [*] --> Exploring
    Exploring --> KeyHeld: pickup key_459 @ room_01
    KeyHeld --> SwitchOn: use switch_337 @ room_02
    SwitchOn --> CodeKnown: query terminal @ room_04 (→ 517603)
    CodeKnown --> LockOpen: use key_459 @ room_04
    LockOpen --> AtVault: move east (needs SwitchOn + LockOpen)
    AtVault --> Verified: submit(vault_447, 517603)
    Verified --> [*]
    Exploring --> DeadEnd: enter a diagnostic branch
    DeadEnd --> Exploring: back out (≈recovery_cost turns wasted)
    note right of Verified
      reward = 1.0 iff room==target
      AND system==vault_447 AND code==517603
      AND the code was queried first
    end note
```

## Tool interface

The policy acts through six tools over an OpenAI-compatible tool-calling API
(hermes parser); every response returns the complete visible state after the
action.

| tool | effect |
|---|---|
| `symbolic_inspect` | re-observe the current room (no state change) |
| `symbolic_move` | traverse an exit; fails if the exit is locked/absent |
| `symbolic_pickup` | take an object (e.g. the key) into inventory |
| `symbolic_use` | activate the switch, or use the key to release the vault lock |
| `symbolic_query` | read the access code from a terminal |
| `symbolic_submit` | submit `(system, code)`; terminal, exact-match verified |
