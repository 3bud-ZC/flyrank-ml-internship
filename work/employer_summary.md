# Employer-Facing Summary

I built an end-to-end ML decision-support system on FlyRank’s gated search-performance warehouse that ranks visible content pages by risk of a >20% next-month impression decline.

I designed historical-only GSC features, a transparent rule baseline, March–April training snapshots, May model selection, and a sealed June test; the selected Random Forest achieved **Precision@50 = 0.920** versus **0.680** for the fixed rule on the final test.

I then converted the model output into a reason-coded human review queue and deployed the full public research paper with reproducible metrics, leakage checks, ranked recommendations, explicit limitations, and no automatic-edit policy.
