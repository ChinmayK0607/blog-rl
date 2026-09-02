# V14.3 rollout-parity diagnosis

Status: CPU-side diagnosis complete. No GPU is allocated and no rejected V14.3
artifact is being resumed.

## Conclusion

V14.3 stopped because the runtime certificate and the live trainer did not
apply the same aggregation semantics. The certificate pooled completion tokens
from all four policy slots, while the live trainer correctly evaluated each
policy slot independently immediately before its atomic optimizer update. A
small pooled mean can therefore certify a runtime in which one policy-local
mean would fail.

The live rejection is much more consistent with concentration of the known
vLLM/Hugging Face numerical tail in one policy batch than with a stale adapter
or steadily diverging model:

- the 32-sample certificate reported pooled mean mismatch KL
  `0.0000528358942`, but already contained a maximum token mismatch KL of
  `0.0548624992`;
- run 0 passed at the rejected boundary with mean mismatch KL
  `0.0000000138368`, then run 1 failed at `0.0012214113`;
- run 1's seven preceding accepted means were `0.00000000212874`,
  `0.000257989`, `0.0000285794`, `0.000176493`, `0.000246345`,
  `0.00017566`, and `0.00000438786`; there is no monotone drift into the
  failure;
- every completed optimizer interval except one contained at least one packed
  microbatch whose local mean exceeded `0.0005`, while the exact policy-local
  logical batches still passed;
- in the rejected interval, 6 of 46 microbatches exceeded `0.0005`. The two
  largest were `0.00566593325` (maximum token mismatch KL `0.126546621`) and
  `0.00400513411` (maximum `0.0615459681`);
- DPPO masking was zero on the first large microbatch and `0.0625` on the
  second. Masking is therefore not the source of the numerical difference.

This does not prove that every token in the rejected batch was safe. V14.3 did
not enable the existing local token exporter, and the compact incident contains
only aggregate failure evidence. It is enough to reject the hypothesis of a
global stale-model failure and to identify the certificate's pooled-vs-local
contract mismatch.

## Gate decision for the next immutable run

Do not restore the historical `0.15` trainer ceiling merely because it would
have passed. Also do not reuse `0.0005` as a universal default. The committed
4B async-admission contract already declares `0.002` for mean mismatch KL, and
the research log previously records a separate run where accidentally using
the certificate CLI's `0.0005` default instead of that run's declared trainer
threshold was an operational error.

The prospective V14.4 contract should therefore:

1. declare `0.002` explicitly in source and bind it into the production plan;
2. require the runtime certificate to pass the same thresholds globally and
   independently for `blue-0` through `blue-3`;
3. record per-policy sample counts, completion-token counts, branching-token
   counts, and all parity metrics in the public runtime certificate;
4. enable local-only token export for the bounded first stage so any future
   rejection can be diagnosed without another GPU rerun;
5. retain the existing DPPO mask, optimization/reduction dtypes, policy
   isolation, scientific curriculum, and update-10 behavioral gate unchanged;
6. run only the first bounded ten-update stage before deciding whether more GPU
   spend is justified, and decommission immediately after rejection or fully
   synced completion.

The `0.002` value is prospective rather than fitted to the `0.0012214113`
failure: it predates V14.3 in the committed 4B async-admission plan. A fresh
runtime certificate is still mandatory; no old certificate may be rebound to
the new contract.

## CPU repair

`certify_prime_parity.py` now computes and gates all four policy-local token
distributions in addition to the pooled distribution. `bind_runtime_certificate.py`
refuses to bind unless all four slots have exactly eight probe samples, nonzero
completion-token coverage, and a passing policy-local result. The bound
certificate now retains total/branching token coverage and the complete
per-policy metric surface.

These changes affect certification evidence only. They do not change the
optimizer, actor, reward, curriculum, seeds, accepted V14.3 updates, or any
rejected artifact.
