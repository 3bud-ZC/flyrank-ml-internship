#!/usr/bin/env python3
"""Build the final FlyRank capstone from the gated warehouse release.

Research question:
Which content pages with meaningful search visibility should a constrained
SEO/content team review first because they are at elevated risk of a >20%
impression decline in the next calendar month?

Design:
- Real FlyRank warehouse data via hf:// + DuckDB.
- Features are historical only.
- Target is a future calendar month.
- March/April 2026 snapshots = training.
- May 2026 = model-selection validation.
- June 2026 = sealed final test.
- Transparent baseline compared on the same snapshots/metric.
- Output is a ranked human-review queue, never an automatic edit instruction.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "outputs"
FIG = ROOT / "work" / "figures"
DOCS = ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN is required. Add a Hugging Face READ token as a secret; never commit it.")

REL = "hf://datasets/FlyRank/internship-warehouse"
FACT = f"read_parquet('{REL}/fact_content_daily_performance/month=2026-0*/*.parquet', hive_partitioning=true)"
CLIENTS = f"read_parquet('{REL}/dim_clients.parquet')"

ANCHORS = [
    pd.Timestamp("2026-03-01"),
    pd.Timestamp("2026-04-01"),
    pd.Timestamp("2026-05-01"),
    pd.Timestamp("2026-06-01"),
]
FEATURES = [
    "log_impressions_recent",
    "log_clicks_recent",
    "ctr_recent",
    "position_recent",
    "position_change",
    "impression_momentum",
    "active_days_recent",
    "active_days_prev",
]
K_VALUES = (20, 50, 100)
SEED = 42


def month_start(ts: pd.Timestamp, offset: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(months=offset)).normalize()


def qdate(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def precision_at_k(y_true: Iterable[int], scores: Iterable[float], k: int) -> float:
    y = np.asarray(y_true)
    s = np.asarray(scores)
    if len(y) == 0:
        return float("nan")
    kk = min(k, len(y))
    idx = np.argsort(-s)[:kk]
    return float(y[idx].mean())


def safe_auc(y_true, scores):
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_ap(y_true, scores):
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def metric_block(y_true, scores) -> Dict[str, float]:
    d = {
        "rows": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "average_precision": safe_ap(y_true, scores),
        "roc_auc": safe_auc(y_true, scores),
    }
    for k in K_VALUES:
        d[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
    return d


def make_baseline(frame: pd.DataFrame) -> np.ndarray:
    decline = np.clip(-frame["impression_momentum"].to_numpy(dtype=float), 0.0, 1.0)
    pos_worse = np.clip(frame["position_change"].fillna(0).to_numpy(dtype=float), 0.0, 10.0) / 10.0
    vis = np.log1p(frame["impressions_recent"].to_numpy(dtype=float))
    if np.nanmax(vis) > np.nanmin(vis):
        vis = (vis - np.nanmin(vis)) / (np.nanmax(vis) - np.nanmin(vis))
    else:
        vis = np.zeros_like(vis)
    return 0.55 * decline + 0.25 * pos_worse + 0.20 * vis


def build_snapshot(con: duckdb.DuckDBPyConnection, anchor: pd.Timestamp, gsc_filter: str) -> pd.DataFrame:
    prev_start = month_start(anchor, -2)
    recent_start = month_start(anchor, -1)
    target_start = anchor
    target_end = month_start(anchor, 1)

    sql = f"""
    WITH eligible_clients AS (
      SELECT client_hash_id
      FROM {CLIENTS}
      WHERE gsc_data_start IS NOT NULL
        AND CAST(gsc_data_start AS DATE) <= DATE '{qdate(prev_start)}'
    ),
    agg AS (
      SELECT
        f.client_hash_id,
        f.content_hash_id,
        SUM(CASE WHEN f.report_date >= DATE '{qdate(prev_start)}'
                  AND f.report_date <  DATE '{qdate(recent_start)}'
                 THEN COALESCE(f.gsc_impressions, 0) ELSE 0 END) AS impressions_prev,
        SUM(CASE WHEN f.report_date >= DATE '{qdate(recent_start)}'
                  AND f.report_date <  DATE '{qdate(target_start)}'
                 THEN COALESCE(f.gsc_impressions, 0) ELSE 0 END) AS impressions_recent,
        SUM(CASE WHEN f.report_date >= DATE '{qdate(recent_start)}'
                  AND f.report_date <  DATE '{qdate(target_start)}'
                 THEN COALESCE(f.gsc_clicks, 0) ELSE 0 END) AS clicks_recent,
        AVG(CASE WHEN f.report_date >= DATE '{qdate(prev_start)}'
                  AND f.report_date <  DATE '{qdate(recent_start)}'
                  AND f.gsc_impressions > 0
                 THEN f.gsc_avg_position END) AS position_prev,
        AVG(CASE WHEN f.report_date >= DATE '{qdate(recent_start)}'
                  AND f.report_date <  DATE '{qdate(target_start)}'
                  AND f.gsc_impressions > 0
                 THEN f.gsc_avg_position END) AS position_recent,
        COUNT(DISTINCT CASE WHEN f.report_date >= DATE '{qdate(prev_start)}'
                              AND f.report_date <  DATE '{qdate(recent_start)}'
                              AND f.gsc_impressions > 0
                            THEN f.report_date END) AS active_days_prev,
        COUNT(DISTINCT CASE WHEN f.report_date >= DATE '{qdate(recent_start)}'
                              AND f.report_date <  DATE '{qdate(target_start)}'
                              AND f.gsc_impressions > 0
                            THEN f.report_date END) AS active_days_recent,
        SUM(CASE WHEN f.report_date >= DATE '{qdate(target_start)}'
                  AND f.report_date <  DATE '{qdate(target_end)}'
                 THEN COALESCE(f.gsc_impressions, 0) ELSE 0 END) AS impressions_target
      FROM {FACT} AS f
      INNER JOIN eligible_clients AS c USING (client_hash_id)
      WHERE f.report_date >= DATE '{qdate(prev_start)}'
        AND f.report_date <  DATE '{qdate(target_end)}'
        {gsc_filter}
      GROUP BY 1,2
    )
    SELECT *
    FROM agg
    WHERE impressions_recent >= 100
      AND impressions_prev >= 50
      AND active_days_recent >= 7
    """
    frame = con.sql(sql).df()
    frame["anchor_month"] = anchor.strftime("%Y-%m")
    frame["ctr_recent"] = frame["clicks_recent"] / frame["impressions_recent"].clip(lower=1)
    frame["position_change"] = frame["position_recent"] - frame["position_prev"]
    frame["impression_momentum"] = (
        (frame["impressions_recent"] - frame["impressions_prev"])
        / frame["impressions_prev"].clip(lower=1)
    )
    frame["log_impressions_recent"] = np.log1p(frame["impressions_recent"])
    frame["log_clicks_recent"] = np.log1p(frame["clicks_recent"])
    frame["target_decline"] = (
        frame["impressions_target"] < 0.80 * frame["impressions_recent"]
    ).astype(int)
    frame["baseline_score"] = make_baseline(frame)
    return frame


def make_logit() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


def make_rf() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def reason_codes(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if row["impression_momentum"] <= -0.20:
        reasons.append("recent_impression_decline")
    if pd.notna(row["position_change"]) and row["position_change"] >= 2.0:
        reasons.append("position_worsening")
    if row["impressions_recent"] >= 1000 and row["ctr_recent"] < 0.01:
        reasons.append("high_visibility_low_ctr")
    if row["active_days_recent"] < 20:
        reasons.append("reduced_active_days")
    if row["impressions_recent"] >= 5000:
        reasons.append("high_visibility")
    return reasons or ["model_pattern_only"]


def action_for(row: pd.Series) -> str:
    reasons = set(reason_codes(row))
    if "high_visibility_low_ctr" in reasons:
        return "review_snippet_intent_and_serp_context"
    if "recent_impression_decline" in reasons or "position_worsening" in reasons:
        return "review_for_refresh_or_competition_change"
    if "reduced_active_days" in reasons:
        return "review_tracking_indexation_or_content_state"
    return "monitor_and_human_review"


con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN ?)", [HF_TOKEN])

schema = con.sql(f"DESCRIBE SELECT * FROM {FACT}").df()
fact_columns = set(schema["column_name"].astype(str))
required = {"report_date", "client_hash_id", "content_hash_id", "gsc_impressions", "gsc_clicks", "gsc_avg_position"}
missing_required = sorted(required - fact_columns)
if missing_required:
    raise RuntimeError(f"Warehouse schema missing required columns: {missing_required}")

gsc_filter = "AND f.gsc_data_available IS TRUE" if "gsc_data_available" in fact_columns else ""

snapshots: Dict[str, pd.DataFrame] = {}
for anchor in ANCHORS:
    key = anchor.strftime("%Y-%m")
    print(f"Building warehouse snapshot for {key} ...", flush=True)
    snapshots[key] = build_snapshot(con, anchor, gsc_filter)
    print(
        f"  rows={len(snapshots[key]):,}, clients={snapshots[key]['client_hash_id'].nunique():,}, "
        f"positive_rate={snapshots[key]['target_decline'].mean():.3f}",
        flush=True,
    )

train = pd.concat([snapshots["2026-03"], snapshots["2026-04"]], ignore_index=True)
val = snapshots["2026-05"].copy()
test = snapshots["2026-06"].copy()

if min(len(train), len(val), len(test)) < 100:
    raise RuntimeError("Insufficient eligible rows after volume/history filters.")

models = {
    "logistic_regression": make_logit(),
    "random_forest": make_rf(),
}
validation_metrics: Dict[str, Dict[str, float]] = {
    "fixed_rule": metric_block(val["target_decline"], val["baseline_score"])
}
for name, model in models.items():
    model.fit(train[FEATURES], train["target_decline"])
    scores = model.predict_proba(val[FEATURES])[:, 1]
    validation_metrics[name] = metric_block(val["target_decline"], scores)

selected_name = max(
    models,
    key=lambda name: (
        validation_metrics[name]["precision_at_50"],
        validation_metrics[name]["average_precision"],
    ),
)
print("Selected model from May validation:", selected_name)

train_final = pd.concat([train, val], ignore_index=True)
selected_model = make_rf() if selected_name == "random_forest" else make_logit()
selected_model.fit(train_final[FEATURES], train_final["target_decline"])
test_model_scores = selected_model.predict_proba(test[FEATURES])[:, 1]

test_metrics = {
    "fixed_rule": metric_block(test["target_decline"], test["baseline_score"]),
    selected_name: metric_block(test["target_decline"], test_model_scores),
}

test = test.copy()
test["model_score"] = test_model_scores
test["reason_codes"] = test.apply(lambda r: "|".join(reason_codes(r)), axis=1)
test["action"] = test.apply(action_for, axis=1)
test["rank"] = test["model_score"].rank(method="first", ascending=False).astype(int)
queue = test.sort_values("rank").copy()
queue_cols = [
    "rank", "content_hash_id", "client_hash_id", "model_score",
    "impressions_recent", "ctr_recent", "position_recent",
    "impression_momentum", "position_change", "reason_codes", "action",
]
queue[queue_cols].to_csv(OUT / "capstone_warehouse_queue.csv", index=False)

top10 = queue[queue_cols].head(10).copy()
top10["model_score"] = top10["model_score"].round(4)
top10["ctr_recent"] = top10["ctr_recent"].round(4)
top10["position_recent"] = top10["position_recent"].round(2)
top10["impression_momentum"] = top10["impression_momentum"].round(4)
top10["position_change"] = top10["position_change"].round(2)
(OUT / "capstone_recommendations_top10.json").write_text(
    json.dumps(top10.to_dict(orient="records"), indent=2, default=str),
    encoding="utf-8",
)

receipt = {
    "release": {
        "dataset": "FlyRank/internship-warehouse",
        "build_id": "flyrank_pseudonymized_warehouse_release_v20260703",
        "daily_fact_documented_rows": 78835655,
        "documented_date_min": "2025-01-27",
        "documented_date_max": "2026-06-30",
        "tables_used": ["dim_clients", "fact_content_daily_performance"],
        "queried_months": "2026-01 through 2026-06",
    },
    "question": "Which visible content pages should be reviewed first because they are at elevated risk of a >20% impression decline in the next calendar month?",
    "lane": "Refresh / Content Opportunity Scoring",
    "grain": "one pseudonymized content item at one monthly decision point",
    "target": "next-calendar-month impressions < 80% of immediately preceding month impressions",
    "eligibility": {
        "recent_month_impressions_min": 100,
        "prior_month_impressions_min": 50,
        "recent_active_days_min": 7,
        "client_history_required_before_feature_window": True,
    },
    "features": FEATURES,
    "validation_design": {
        "train_snapshots": ["2026-03", "2026-04"],
        "model_selection_validation": "2026-05",
        "sealed_final_test": "2026-06",
        "seed": SEED,
        "primary_metric": "Precision@50",
    },
    "snapshot_stats": {
        key: {
            "rows": int(len(frame)),
            "clients": int(frame["client_hash_id"].nunique()),
            "positive_rate": float(frame["target_decline"].mean()),
        }
        for key, frame in snapshots.items()
    },
    "validation_metrics": validation_metrics,
    "selected_model": selected_name,
    "sealed_test_metrics": test_metrics,
    "leakage_checks": {
        "future_target_fields_used_as_features": False,
        "hash_ids_used_as_features": False,
        "query_90d_table_used": False,
        "ga4_signals_used": False,
        "target_month_sealed_until_final_evaluation": True,
    },
    "human_review_required": True,
    "automatic_editing": False,
}
(OUT / "capstone_warehouse_metrics.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)

# Plot the final sealed-test comparison.
labels = ["Fixed rule", selected_name.replace("_", " ").title()]
p50 = [
    test_metrics["fixed_rule"]["precision_at_50"],
    test_metrics[selected_name]["precision_at_50"],
]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(labels, p50)
ax.axhline(test_metrics[selected_name]["base_rate"], linestyle="--", linewidth=1)
ax.set_ylabel("Precision@50")
ax.set_ylim(0, 1)
ax.set_title("Sealed June 2026 test: model vs transparent baseline")
for i, value in enumerate(p50):
    ax.text(i, value + 0.025, f"{value:.3f}", ha="center")
fig.tight_layout()
fig.savefig(FIG / "capstone_model_comparison.svg")
fig.savefig(DOCS / "capstone_model_comparison.svg")
plt.close(fig)

# Generate the public research paper from the real warehouse receipts.
m_base = test_metrics["fixed_rule"]
m_model = test_metrics[selected_name]
snapshot_stats = receipt["snapshot_stats"]

rows_html = "\n".join(
    f"""<tr><td>{r['rank']}</td><td><code>{r['content_hash_id']}</code></td><td>{r['model_score']:.4f}</td><td>{r['action']}</td><td>{r['reason_codes'].replace('|', ', ')}</td></tr>"""
    for _, r in top10.iterrows()
)

paper = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predicting Next-Month Search Decline for Content Review Prioritization</title>
<style>
:root{{--bg:#08131d;--card:#0f2130;--text:#eaf4f6;--muted:#a9bcc2;--accent:#7ce0c3;--line:#284252}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,Arial,sans-serif;line-height:1.65}}
main{{max-width:1020px;margin:auto;padding:48px 24px 80px}}h1{{font-size:clamp(2rem,5vw,3.6rem);line-height:1.08}}h2{{margin-top:2.2em;color:var(--accent)}}
p,li{{color:#d9e7ea}}a{{color:var(--accent)}}code{{background:#173042;padding:2px 5px;border-radius:5px}}
.card,.abstract{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:18px 0}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{background:var(--card);border:1px solid var(--line);padding:16px;border-radius:12px}}
.metric b{{display:block;font-size:1.7rem;color:var(--accent)}}table{{width:100%;border-collapse:collapse;background:var(--card);margin:18px 0}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--accent)}}img{{max-width:100%}}
.badge{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:6px 10px;color:var(--accent)}}.muted{{color:var(--muted)}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}table{{display:block;overflow-x:auto}}main{{padding:28px 15px 60px}}}}
</style>
</head><body><main>
<header><span class="badge">FlyRank ML Internship · Capstone</span>
<h1>Predicting Next-Month Search Decline for Content Review Prioritization</h1>
<p class="muted">Abdullah Ragab · Refresh / Content Opportunity Scoring · public-safe decision-support research</p></header>

<section class="abstract"><h2>Abstract</h2>
<p>Which content pages should a constrained SEO/content team review first before search visibility declines? I built monthly decision snapshots from the gated FlyRank warehouse and used only historical Google Search Console signals to predict whether next-month impressions would fall by more than 20%. Models were selected on May 2026 after training on March–April snapshots, then evaluated once on a sealed June 2026 test month against the same transparent baseline and Precision@50 metric. On the sealed test, the selected {selected_name.replace('_',' ')} achieved Precision@50 = <b>{m_model['precision_at_50']:.3f}</b> versus <b>{m_base['precision_at_50']:.3f}</b> for the fixed rule, with a test base rate of <b>{m_model['base_rate']:.3f}</b>. The result supports a ranked human-review queue with reason codes; it does not establish that refreshing a page causes recovery.</p></section>

<section><h2>1. Introduction / Problem Statement</h2>
<p>Large content portfolios create a prioritization problem: review capacity is limited, while search performance changes continuously. The decision supported here is <b>which already-visible pages should be reviewed first at the end of a month because their next-month search impressions appear at elevated decline risk</b>. The output is a review order, not an automatic rewrite, deletion, redirect, or publishing instruction.</p></section>

<section><h2>2. Data</h2>
<div class="grid">
<div class="metric"><b>78,835,655</b>documented daily fact rows in the release</div>
<div class="metric"><b>2026-01 → 2026-06</b>calendar months queried for this study</div>
<div class="metric"><b>{snapshot_stats['2026-06']['rows']:,}</b>eligible June decision rows</div>
</div>
<p>Source: <code>FlyRank/internship-warehouse</code>, build <code>flyrank_pseudonymized_warehouse_release_v20260703</code>. The study uses <code>dim_clients</code> and <code>fact_content_daily_performance</code>. The full release spans 2025-01-27 through 2026-06-30; this capstone deliberately queries January–June 2026 because those months are sufficient to build non-overlapping feature and target windows while keeping June sealed for final evaluation.</p>
<p>One row in the modeling table is one pseudonymized content item at one monthly decision point. Eligibility requires at least 100 impressions in the immediately preceding month, at least 50 impressions in the month before that, at least seven active search days in the recent month, and client history beginning before the feature window. I excluded GA4 and the fixed 90-day query table from the final model to keep the target boundary simple and avoid tracking-availability and overlapping-window leakage.</p></section>

<section><h2>3. Methodology</h2>
<ul>
<li><b>Target:</b> next-calendar-month impressions &lt; 80% of immediately preceding-month impressions.</li>
<li><b>Features:</b> recent impressions/clicks, CTR, average position, position change, impression momentum, and active search days — all measured before the target month.</li>
<li><b>Baseline:</b> transparent weighted score emphasizing recent negative momentum, worsening position, and review value from visibility.</li>
<li><b>Models:</b> Logistic Regression and Random Forest.</li>
<li><b>Time-aware validation:</b> March–April snapshots train the models; May selects the method; June is held sealed until final evaluation.</li>
<li><b>Primary metric:</b> Precision@50, because the operational constraint is a limited review queue.</li>
<li><b>Leakage checks:</b> no target-month features, no hash IDs as features, no product decision flags, no query-window overlap, no automatic actions.</li>
</ul></section>

<section><h2>4. Results</h2>
<table><thead><tr><th>Method</th><th>June base rate</th><th>P@20</th><th>P@50</th><th>P@100</th><th>Average precision</th><th>ROC AUC</th></tr></thead>
<tbody>
<tr><td>Transparent fixed rule</td><td>{m_base['base_rate']:.3f}</td><td>{m_base['precision_at_20']:.3f}</td><td>{m_base['precision_at_50']:.3f}</td><td>{m_base['precision_at_100']:.3f}</td><td>{m_base['average_precision']:.3f}</td><td>{m_base['roc_auc']:.3f}</td></tr>
<tr><td>{selected_name.replace('_',' ').title()}</td><td>{m_model['base_rate']:.3f}</td><td>{m_model['precision_at_20']:.3f}</td><td>{m_model['precision_at_50']:.3f}</td><td>{m_model['precision_at_100']:.3f}</td><td>{m_model['average_precision']:.3f}</td><td>{m_model['roc_auc']:.3f}</td></tr>
</tbody></table>
<p><img src="capstone_model_comparison.svg" alt="Sealed June model versus baseline Precision at 50"></p>
<p>The headline number is the sealed June Precision@50. May was used for method selection; June was not used to choose the winning model. The comparison therefore measures next-month ranking performance under a true past→future boundary, not a same-window proxy.</p></section>

<section><h2>5. Limitations & Honest Framing</h2>
<ul>
<li>This is predictive ranking, not a causal refresh experiment.</li>
<li>The label defines decline as a &gt;20% month-over-month impression drop; alternative thresholds may change the queue.</li>
<li>Seasonality, SERP changes, content consolidation, campaigns, indexing events, and measurement noise can create false positives.</li>
<li>The panel is unbalanced. History eligibility reduces, but does not eliminate, cross-client differences.</li>
<li>The model uses GSC-only features. That improves availability consistency but excludes potentially useful engagement context.</li>
</ul>
<div class="card"><b>Safe claim:</b> on the specified warehouse snapshot and time-aware evaluation, the model measured a different ability than the transparent baseline to prioritize pages that later met the defined decline outcome. The study does not prove why a page declined or that any intervention would reverse it.</div></section>

<section><h2>6. Ranked Recommendations</h2>
<p>The table below is the top of the decision-time review queue. IDs are pseudonymized; reason codes use only information available before the target month.</p>
<table><thead><tr><th>Rank</th><th>Content ID</th><th>Score</th><th>Recommended review</th><th>Reason codes</th></tr></thead><tbody>
{rows_html}
</tbody></table>
<p><b>Action policy:</b> model = ordering, reason codes = explanation, human = decision. No automatic edit, rewrite, merge, deletion, redirect, or publication is authorized by the score.</p></section>

<section><h2>7. Reproducibility</h2>
<p>The complete implementation and executed evidence live in the public repository. The final run records snapshot row counts, model-selection metrics, sealed-test metrics, target definition, leakage checks, and the top recommendation receipt as small committed JSON artifacts. The bulk ranked queue stays out of git by design.</p>
<p><a href="https://github.com/3bud-ZC/flyrank-ml-internship">Repository and executed notebooks</a></p></section>

<section><h2>8. Acknowledgments & Data Credit</h2>
<p><b>Built on the FlyRank ML Internship dataset</b> — <a href="https://flyrank.ai">FlyRank AI</a>. The data release is pseudonymized; no client names, domains, raw URLs, private queries, credentials, or raw identifying exports are published here.</p></section>
</main></body></html>"""
(DOCS / "index.html").write_text(paper, encoding="utf-8")

print(json.dumps({
    "selected_model": selected_name,
    "june_rows": int(len(test)),
    "june_base_rate": m_model["base_rate"],
    "baseline_p50": m_base["precision_at_50"],
    "model_p50": m_model["precision_at_50"],
    "paper": str(DOCS / "index.html"),
}, indent=2))
