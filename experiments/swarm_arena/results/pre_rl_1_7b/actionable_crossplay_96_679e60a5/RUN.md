# Actionable-prompt 1.7B communication cross-play

Status: completed and verified; collective communication screen marginally
promising, not RL admission.

- Run source: `679e60a57a35abecc989a5a654236f19bbb9182b`
- Analyzer source: `274bbe87643ff36f2eb3c239b7d8bfd9dfb7fe0b`
- Prompt: `arena-episode-v5-actionable-broadcast-priority`
- Focal: 1.7B SFT adapter
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Opponent: model-controlled 1.7B base
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Matrix: 12 new development seeds x 4 focal communication conditions x 2
  side assignments = 96 complete games / 48 side-swapped pairs
- Runtime: 2026-08-14 15:44:39--15:57:32 UTC (about 773 seconds)
- Inference: 7,680 requests, 182,608 completion tokens, 754.53 aggregate
  inference seconds, 242.02 completion tokens/s
- Optimizer steps: zero

## Result

Side-averaged generated-minus-control effects over 12 paired seeds:

| Control | Mean return difference | 95% interval | p | Positive seeds |
| --- | ---: | --- | ---: | ---: |
| Dropped | +0.3167 | [-0.8792, +1.3542] | .6162 | 8/12 |
| Sender-shuffled | -0.0250 | [-1.3625, +1.3000] | .9712 | 6/12 |
| One-turn delayed | +1.2042 | [+0.2208, +2.2833] | .0474 | 7/12 |

Generated versus dropped meets the predeclared exploratory screen exactly:
positive mean and 8/12 positive seeds. The uncertainty interval is wide and
includes zero, so this is not confirmatory evidence. Delaying messages causes a
clear loss, indicating that agents use time-sensitive shared information.
Sender shuffling has no detectable effect, so sender identity or routing is not
yet carrying robust value.

Every focal output in every condition was protocol-valid and grounded, with
zero invalid actions/broadcasts. Generated play had a 0.4375 duplicate-target
turn rate, higher than dropped (0.2361), shuffled (0.3507), and delayed
(0.3611). Current messages therefore do not reliably reduce redundant action
selection.

## Artifacts

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `manifest.json` | 1,788 | `ea944e641599d63924260ddc73b2038a4a4299fbb911eed090565cd52f4d7e13` |
| `summary.json` | 6,840 | `c1e050ed02161b9df8db9e4de0d90851745ec1ba3afd4b927a70a0a990be002d` |
| `side_swapped_summary.json` | 2,507 | `3e65991a1725e484fd688e943b9360fb527fe3c3cf077479ddd089413be8ea21` |
| `verified_summary.json` | 15,898 | `634fcf71a6157d6481b68e677be842a70145fac8eb9d38ff6ab45649c13d9835` |
| `rows.jsonl.gz` | 2,321,505 | `a45f322bd31b0f7f49e4df8861c1d012b1ff57f5df0c6eaad171d39d52d73b06` |

The deterministic gzip expands to a 54,522,394-byte `rows.jsonl` with SHA-256
`2dc69b5bf084eb97792b5aaacba0010dfded3d7b075ff0ad90c2faf5f18334e9`.
Canonical manifest identity:
`04a9b305806a4e0f4e1e792e9a381c0dc1a9a7a31d845da9baf3b20981ea2df7`.

