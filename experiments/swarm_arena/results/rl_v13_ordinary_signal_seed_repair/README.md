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

The complete round-two screen then ran all 64 cases and rejected only blue-0:
its two variable groups covered current and historical opponents, while all
eight base/SFT cases were deterministic. The exact assessment and run manifest
are preserved as `ASSESSMENT_ROUND3.json` and `RUN_MANIFEST_ROUND3.json`.
Rather than relying on one lucky seed per family again, the third targeted
search replaced the full four-case blue-0 base band and four-case SFT band with
eight independently signal-bearing cases:

- Round-three search SHA-256: `08caae4af41b692cbbd2303c0896867296815186db36f5b645d0cdd061ca3370`
- Round-three assessment SHA-256: `ccd68989160306f9f94d72d89f020bcea8219b1ff25abc0d634c90a277663cb4`
- Round-three run-manifest SHA-256: `38fd42eb63334aed69a1fda92c6a177ddf558b2c2ee7de75ff2b462aaf8848ee`
- Repaired screen body SHA-256: `f33962d49920a11d2eb34c06db8d233677fcbe0f5e306a7fd3f68df5dca70afb`
- Repaired screen file SHA-256: `c884813ef913e5f8761ed1a6e3503cfd010b53c10ade0bafb608ffb9b3c06aa0`
- Optimizer updates: 0; frozen data opened: no

Rewards, admission thresholds, curriculum proportions, model weights, and
opponent identities remain unchanged. A fresh full screen is still required.

That fresh screen completed 64/64 and rejected two narrowly coupled gaps:
blue-0 lacked a current-opponent variable group, and blue-2 had three variable
groups but no current-opponent group. The fourth targeted search therefore
replaced the complete four-case current band for each of those two policies:

- Round-four search SHA-256: `b394fd1368345535dcbeaa4ecdbc254159db7abcaf51d0639efefbae6797b0d4`
- Round-four assessment SHA-256: `5989d7034c4306b39a40e62049016c670a2d26a1af029759d19a7328d739c38b`
- Round-four run-manifest SHA-256: `1133a8d02ae02464245037f55c3616f4a28cafdbdd00cbf7defd821ab15f7d9b`
- Repaired screen body SHA-256: `5aeecd617b364cf4188a44168a47d58b7abed797b26b2095be7a4fd450213f8d`
- Repaired screen file SHA-256: `277f0fee0163ab21184ebc377b90d84da60bd5804d5717a7a6bc4c9bc758a3bb`
- Optimizer updates: 0; protocol admission: 100%; frozen data opened: no

This retains the original gates while giving each missing policy/family cell
four independent opportunities to provide credit in the final full screen.
