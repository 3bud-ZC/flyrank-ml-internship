# Social Post Cut

I built a time-aware ML ranking system on FlyRank’s gated search-performance warehouse to answer one practical question:

**Which visible content pages should an SEO/content team review first because they are at elevated risk of a >20% impression decline next month?**

The final pipeline uses historical GSC signals only, trains on March–April 2026 decision snapshots, selects the model on May, and keeps June sealed for final evaluation.

On the sealed June test, the Random Forest reached **Precision@50 = 0.920**, compared with **0.680** for a transparent fixed-rule baseline. The result becomes a ranked human-review queue with reason codes — not an automatic content-editing system.

The bigger lesson was that the model is only one part of the work. Time boundaries, leakage controls, a fixed baseline, honest claim language, and human review are what make the output usable.

Research paper: https://flyrank-ml-paper-production.up.railway.app/
Repository: https://github.com/3bud-ZC/flyrank-ml-internship
