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

The repaired broad screen was then stopped at 60/64 when blue-3/historical
exhausted its four cases with zero variable credit. A second targeted round
repaired blue-3/current and blue-3/historical without optimizer updates:

- Round-two search SHA-256: `2691f871225879570e6cd7f7d9a75a441d38e9acd5c70e1dfad07a371ce9ad21`
- Round-two repaired screen body SHA-256: `e46cd2856ea99a223ce1bbbdf67223d57059c328103bbe1f270bad591cb967e0`
- Round-two repaired screen file SHA-256: `884928394ef1216efb918a1e810c2da349c4ff5fa18a9c99ff68c7f08f3a4678`

The selected rows require nonzero return range, four nonzero focused-agent
advantages containing both signs, and at least two distinct focused actions.
The final full 64-case screen must still pass before V13 training is authorized.
