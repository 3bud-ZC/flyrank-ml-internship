# Social Post Cut

I built an explainable content-review ranking workflow on 30,000 anonymized content items.

Instead of optimizing for generic accuracy, I framed the real decision as: **which pages should an SEO team inspect first?** I froze a transparent baseline, compared learned ranking under client-holdout validation using Precision@50, audited leakage, and turned the result into a reason-coded human review queue.

The biggest lesson: the model score is only one part of a useful ML system. Validation boundaries, reason codes, limits, and a clear no-auto-edit policy matter just as much.

Full research paper: [DEPLOYED_PAPER_URL]
