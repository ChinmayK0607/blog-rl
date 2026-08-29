# V13 ordinary signal seed repair

The first zero-optimizer V13 ordinary screen was rejected after 48/64 cases:
blue-1 had exhausted base, SFT, and historical cells without a variable-return
group. A compact training-only search then replaced exactly eight dead
policy/opponent cells. It did not change rewards, thresholds, policy weights,
or frozen evaluation data.

- Optimizer updates: 0
- Selected cells: 8/8
- Search report SHA-256: `57c9ff0e1d5ba0a15ecd8b01bc2e368398adc0f5904e8a0e4d1078c1c8792de9`
- Repaired screen body SHA-256: `2a93f47e85a235c83e4de8179d9b9f76cddd962c1dd172721c37b946a1f3097a`
- Repaired screen file SHA-256: `4282f69a250e631f3d64a5231019231687ac7dc8c4afe07ff0e066a60dcb0e4f`
- Frozen data opened: no

The selected rows require nonzero return range, four nonzero focused-agent
advantages containing both signs, and at least two distinct focused actions.
The full 64-case screen must still pass before V13 training is authorized.
