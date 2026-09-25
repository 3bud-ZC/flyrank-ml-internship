# ML-12 — 5-Minute Demo Outline

## 0:00–0:40 — The decision
Which already-visible content pages should a constrained SEO/content team review first because they are at elevated risk of a >20% impression decline in the next calendar month?

## 0:40–1:30 — Real FlyRank warehouse data
The final capstone reads the gated `FlyRank/internship-warehouse` release directly with DuckDB. The documented release contains 78,835,655 daily performance rows. The final study queries January–June 2026 and creates monthly decision snapshots with only historical GSC features.

## 1:30–2:25 — Leakage-safe design
- March + April snapshots: training.
- May: model selection.
- June: sealed final test.
- Target: next-month impressions fall below 80% of the immediately preceding month.
- No target-month fields, IDs, product decisions, query-window overlap, or GA4 features enter the model.

## 2:25–3:20 — One headline result
On the sealed June test:
- Fixed rule Precision@50 = **0.680**
- Random Forest Precision@50 = **0.920**
- Test base rate = **0.692**

The model also reached Precision@20 = **0.950** and Precision@100 = **0.910**. These are predictive ranking results, not causal evidence.

## 3:20–4:10 — Ranked action engine
The final output is a top-K human review queue with feature-only reason codes such as recent impression decline, worsening position, high visibility with low CTR, and reduced active days. The score never triggers an automatic edit.

## 4:10–5:00 — Honest conclusion
The project shows that, on this warehouse snapshot and time-aware evaluation, the learned model prioritized the defined next-month decline outcome more effectively than the transparent baseline at the top of the queue. It does not prove why a page declined or that refreshing it will cause recovery.

Public research paper: https://flyrank-ml-paper-production.up.railway.app/
Repository: https://github.com/3bud-ZC/flyrank-ml-internship
