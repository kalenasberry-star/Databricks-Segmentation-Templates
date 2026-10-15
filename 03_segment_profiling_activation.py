# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Segment Profiling & Activation
# MAGIC
# MAGIC Consumes the feature table (notebook 01) and the champion model
# MAGIC (notebook 02). Scores **every eligible customer**, turns clusters into
# MAGIC business-readable profiles with auto-suggested names, and publishes the
# MAGIC activation table your campaign tool (Braze, Salesforce MC, …) reads.
# MAGIC
# MAGIC **The naming layer, stated plainly:** clusters are math — frozen the
# MAGIC moment training finished. Names are human annotation applied *afterwards*
# MAGIC by joining a dictionary on `segment_id`. The `suggest()` function below
# MAGIC proposes names from each cluster's standout behaviors; the SEGMENT_NAMES
# MAGIC dict overrides it after your naming session. Guard assertions make the
# MAGIC classic failure — a names dict silently applied to the wrong model's
# MAGIC clusters — impossible: the cell dies instead of rendering a confident
# MAGIC wrong chart.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Parameters & the naming dictionaries

# COMMAND ----------

from pyspark.sql import functions as F
import numpy as np
import pandas as pd

dbutils.widgets.text("output_schema", "main.segmentation")
dbutils.widgets.text("feature_table_name", "customer_features")
dbutils.widgets.text("model_name", "customer_segmentation")
dbutils.widgets.text("model_run_id", "")   # fallback when the registry/@champion isn't reachable:
                                           # paste the run id printed by notebook 02
dbutils.widgets.text("segments_table_name", "customer_segments")

OUT_SCHEMA     = dbutils.widgets.get("output_schema")
FEATURE_TABLE  = f"{OUT_SCHEMA}.{dbutils.widgets.get('feature_table_name')}"
MODEL_NAME     = f"{OUT_SCHEMA}.{dbutils.widgets.get('model_name')}"
MODEL_RUN_ID   = dbutils.widgets.get("model_run_id").strip()
SEGMENTS_TABLE = f"{OUT_SCHEMA}.{dbutils.widgets.get('segments_table_name')}"

DAYPARTS = ["early_morning", "breakfast", "lunch", "afternoon_snack", "dinner", "late_night"]

# Final business names & campaign angles, keyed by segment_id. Leave empty on
# the first run: Cell 3 prints auto-suggestions with each cluster's evidence.
# After the naming session, fill these in and re-run — names flow into the
# activation table, the printed profile, and the charts from this ONE place.
SEGMENT_NAMES: dict[int, str] = {}        # e.g. {0: "Breakfast Loyalist", 1: "Fading Regular", ...}
SEGMENT_OPPORTUNITY: dict[int, str] = {}  # e.g. {0: "Defend the ritual; nudge a second daypart", ...}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Load features and score everyone with the champion
# MAGIC Model loading is registry-first (the production pattern: scoring loads
# MAGIC whatever is `@champion`, never a hardcoded version) with a run-URI
# MAGIC fallback. Scoring uses the **model's own signature** for the feature
# MAGIC list and order — the dataframe is made to fit the model, never the
# MAGIC other way around.

# COMMAND ----------

import mlflow
import mlflow.sklearn

assert spark.catalog.tableExists(FEATURE_TABLE), f"{FEATURE_TABLE} not found — run notebook 01 first."

df = spark.table(FEATURE_TABLE).filter("eligible_for_modeling").toPandas()
num_cols = df.select_dtypes("number").columns
df[num_cols] = df[num_cols].astype("float64")   # match the double-typed training signature

try:
    mlflow.set_registry_uri("databricks-uc")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    model_src = f"models:/{MODEL_NAME}@champion"
except Exception:
    assert MODEL_RUN_ID, ("Registry model unavailable and the model_run_id widget is empty. "
                          "Paste the run id printed by notebook 02 (champion_k* run).")
    model = mlflow.sklearn.load_model(f"runs:/{MODEL_RUN_ID}/model")
    model_src = f"runs:/{MODEL_RUN_ID}/model"

MODEL_INPUT = list(model.feature_names_in_)   # the signature is the source of truth
missing = [c for c in MODEL_INPUT if c not in df.columns]
assert not missing, (f"Model expects features missing from the table: {missing} — "
                     f"the feature notebook and the model are out of sync; re-run 01 (or retrain).")

df["segment_id"] = model.predict(df[MODEL_INPUT])
k = df["segment_id"].nunique()
print(f"Scored {len(df):,} customers into {k} segments using {model_src}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Profile, signatures, and auto-suggested names
# MAGIC For each segment: size, share, behavioral averages, and an over/
# MAGIC under-index *signature* — the segment's mean on each feature vs the
# MAGIC population, in population standard deviations (±1.0 sd is a meaningful
# MAGIC behavioral separation). The signature drives the suggestions; it is also
# MAGIC your QA instrument — **when a label contradicts the signature, the
# MAGIC signature wins.**

# COMMAND ----------

pop_mean = df[MODEL_INPUT].mean()
pop_std = df[MODEL_INPUT].std().replace(0, 1)

AGG = dict(
    customers=("customer_id", "count"),
    total_spend=("target_total_spend", "sum"),
    txns_per_customer=("target_txn_count", "mean"),
    avg_ticket=("avg_ticket", "mean"),
    items_per_txn=("avg_items_per_txn", "mean"),
    freq_trend=("freq_trend", "mean"),
    active_days=("active_days", "mean"),
    recency_days=("recency_days", "mean"),
    discount=("pct_discount", "mean"),
    weekend=("pct_weekend", "mean"),
    avg_purchase_gap=("avg_days_between_purchases", "mean"),
    median_purchase_gap=("median_days_between_purchases", "mean"),
    breakfast=("pct_breakfast", "mean"),
    lunch=("pct_lunch", "mean"),
    afternoon=("pct_afternoon_snack", "mean"),
    dinner=("pct_dinner", "mean"),
    late_night=("pct_late_night", "mean"),
)
# Optional columns (present only when notebook 01 ran with a secondary category).
for name, col in [("pairing", "secondary_pairing_pct"),
                  ("secondary_share", "secondary_spend_share"),
                  ("other_share", "other_spend_share"),
                  ("target_share", "target_spend_share")]:
    if col in df.columns:
        AGG[name] = (col, "mean")

profile = df.groupby("segment_id").agg(**AGG).round(2)
profile["share_pct"] = (profile["customers"] / profile["customers"].sum() * 100).round(1)

def signature(seg_id: int, n: int = 3):
    """Top over- and under-indexed features for one segment (z-scores)."""
    z = (df.loc[df.segment_id == seg_id, MODEL_INPUT].mean() - pop_mean) / pop_std
    return z.sort_values(ascending=False).head(n), z.sort_values().head(n), z

def suggest(seg_id: int) -> tuple[str, str]:
    """Rule-based archetype + campaign angle from standout behaviors.
    A field guide, not a roster: more rules exist than segments, and each
    cluster matches whichever rule fires first. Suggestions only — the
    SEGMENT_NAMES dict always wins. `gz` makes optional features safe."""
    _, _, z = signature(seg_id)
    gz = lambda f: float(z.get(f, 0.0))
    seg = df[df.segment_id == seg_id]

    if gz("secondary_pairing_pct") > 1.0:
        return ("Cross-Shop Pairer",
                "Deepen the existing pairing habit: timely category offers at the moment of the paired purchase")
    if gz("pct_discount") > 1.0 and (gz("avg_items_per_txn") > 1.0 or gz("avg_ticket") > 1.0):
        return ("Big Basket Deal Buyer",
                "Protect the basket while restructuring the deal: fixed-price bundles over blanket discounts")
    if gz("pct_discount") > 1.0:
        return ("Deal Seeker",
                "Strategic offer timing to hold frequency without eroding margin; test smaller, targeted offers")
    if gz("log_txn_count") > 1.0 or gz("active_days") > 1.0:
        return ("Heavy Regular",
                "Protect and grow: recognition, early access, cross-daypart nudges — never discount an existing habit")
    if gz("recency_days") > 1.0 or gz("freq_trend") < -0.75:
        return ("Fading Regular",   # NOT "Lapsed": these customers still cleared the recency bar
                "Early-save re-engagement: a warm, short-expiry reason to come back this week")
    if gz("freq_trend") > 0.75:
        return ("Rising Regular",
                "Ride the momentum: frequency plays into a habit that is already forming")
    if gz("pct_breakfast") > 1.0:
        return ("Morning Routine", "Morning-window timing beats offer depth; attach complements")
    if gz("pct_late_night") > 1.0 or gz("pct_dinner") > 1.0:
        return ("Evening Crowd", "Evening/overnight timing; meal-sized bundles with beverages")
    if gz("pct_weekend") > 1.0:
        return ("Weekender", "Weekend-timed promotions and family-size bundles")
    if gz("avg_ticket") > 1.0 or gz("avg_items_per_txn") > 1.0:
        return ("Big Basket", "Attach and bundle: grow an already-large basket, protect ticket size")
    if "other_spend_share" in seg.columns and seg["other_spend_share"].mean() > 0.5:
        return ("In-Store Casual", "They're already in the store: cross-promote at checkout to build the habit")
    return ("Occasional", "Frequency building: light-touch reminders and low-cost trial offers")

for seg in sorted(profile.index):
    auto_name, auto_opp = suggest(seg)
    profile.loc[seg, "segment_name"] = SEGMENT_NAMES.get(seg, auto_name)
    profile.loc[seg, "opportunity"] = SEGMENT_OPPORTUNITY.get(seg, auto_opp)
    over, under, _ = signature(seg)
    print(f"\nSegment {seg} — {profile.loc[seg, 'segment_name']}  "
          f"({int(profile.loc[seg, 'customers']):,} customers, {profile.loc[seg, 'share_pct']}%)")
    print("  over-indexes:  " + ", ".join(f"{f} ({v:+.1f} sd)" for f, v in over.items()))
    print("  under-indexes: " + ", ".join(f"{f} ({v:+.1f} sd)" for f, v in under.items()))
    print(f"  opportunity:   {profile.loc[seg, 'opportunity']}")

# --- Guards: make the naming-crossover bug class impossible -------------------
if SEGMENT_NAMES:
    assert set(SEGMENT_NAMES) == set(profile.index), (
        f"SEGMENT_NAMES keys {sorted(SEGMENT_NAMES)} don't match this model's segment ids "
        f"{sorted(profile.index)} — wrong dict or wrong model.")
assert profile["segment_name"].notna().all() and profile["segment_name"].nunique() == len(profile), \
    "Segment names must be complete and unique."
# Behavior-consistency checks: a name must match the behavior it claims.
# Add your own pairs as naming conventions evolve.
_claims = [(("fading", "lapsed", "lapsing"), "recency_days", "max"),
           (("rising", "momentum"), "freq_trend", "max")]
for keywords, col, how in _claims:
    claimed = [s for s in profile.index if any(kw in profile.loc[s, "segment_name"].lower() for kw in keywords)]
    if claimed:
        actual = profile[col].idxmax() if how == "max" else profile[col].idxmin()
        assert actual in claimed, (
            f"NAMES CROSSED WITH BEHAVIOR: the {how}-{col} segment is id {actual} "
            f"('{profile.loc[actual, 'segment_name']}'), but {[profile.loc[s, 'segment_name'] for s in claimed]} "
            f"claim that behavior. Fix SEGMENT_NAMES before any chart or export leaves this notebook.")

display(spark.createDataFrame(profile.reset_index()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Publish the activation table
# MAGIC One row per customer. Curate what syncs to your campaign tool: a
# MAGIC categorical like `primary_daypart` is both more usable to marketers and
# MAGIC cheaper (data points are billed per attribute per user) than raw floats.
# MAGIC Holdout/control-group logic is deliberately **not** here — manage a
# MAGIC universal control group in the campaign tool, confirm it is globally
# MAGIC excluded from every send, and make sure membership flows back for lift
# MAGIC measurement.

# COMMAND ----------

DAYPART_SHARE_COLS = [f"pct_{d}" for d in DAYPARTS]

segments_out = df[["customer_id", "segment_id",
                   "avg_days_between_purchases", "median_days_between_purchases",
                   *DAYPART_SHARE_COLS, "pct_weekend"]].copy()
segments_out[DAYPART_SHARE_COLS + ["pct_weekend"]] = \
    segments_out[DAYPART_SHARE_COLS + ["pct_weekend"]].round(3)
# The marketer-facing attribute: each customer's single strongest window.
segments_out["primary_daypart"] = (
    segments_out[DAYPART_SHARE_COLS].idxmax(axis=1).str.replace("pct_", "", regex=False))
segments_out["segment_name"] = segments_out["segment_id"].map(profile["segment_name"])
segments_out["scored_at"] = pd.Timestamp.utcnow()

assert segments_out["segment_name"].notna().all(), "Unmapped segment ids reached the activation table."
_share_sum = segments_out[DAYPART_SHARE_COLS].sum(axis=1)
assert _share_sum.between(0.99, 1.01).all(), "Daypart shares don't sum to 1 — upstream feature issue."

(spark.createDataFrame(segments_out).write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(SEGMENTS_TABLE))
# The table is a full snapshot per run: values move between runs and carry no
# history. If "what segment was this customer last month?" will ever be asked,
# add an append-mode history table NOW, not after the question arrives.
print(f"Published {SEGMENTS_TABLE} ({len(segments_out):,} customers)\n")
print(segments_out.groupby("segment_name").size().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Profile visuals (real units only)
# MAGIC Every number is in dollars, days, or percent — stakeholders don't think
# MAGIC in standard deviations, and a z-score heatmap invites "plus 1.3 *what*?".
# MAGIC All series read from the same `profile` dataframe as the table above, so
# MAGIC charts and tables cannot disagree.

# COMMAND ----------

import matplotlib.pyplot as plt

names = profile["segment_name"].values
n_seg = len(names)
x = np.arange(n_seg)
palette = ["#2A9D8F", "#E9C46A", "#E76F51", "#264653", "#8AB17D", "#7B2D8B",
           "#B5838D", "#457B9D"]
seg_colors = [palette[i % len(palette)] for i in range(n_seg)]

fig, axes = plt.subplots(2, 3, figsize=(17, 9))

# Panel 1 — size vs spend: the revenue-concentration story in one picture.
ax = axes[0, 0]
spend_share = (profile["total_spend"] / profile["total_spend"].sum() * 100).values
width = 0.38
ax.bar(x - width / 2, profile["share_pct"].values, width, label="Customer %", color="#2A9D8F")
ax.bar(x + width / 2, spend_share, width, label="Category spend %", color="#E76F51")
for i in range(n_seg):
    ax.text(i + width / 2, spend_share[i] + 0.5, f"{spend_share[i]:.0f}%",
            ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Share (%)")
ax.set_title("Who they are vs. what they spend", fontweight="bold")
ax.legend(fontsize=8)

# Panel 2 — daypart mix (stacked shares).
ax = axes[0, 1]
dp_cols = ["breakfast", "lunch", "afternoon", "dinner", "late_night"]
dp_colors = ["#F4A261", "#2A9D8F", "#E76F51", "#264653", "#7B2D8B"]
bottom = np.zeros(n_seg)
for col, color in zip(dp_cols, dp_colors):
    vals = profile[col].values
    ax.bar(x, vals, bottom=bottom, label=col.replace("_", " ").title(), color=color)
    bottom += vals
ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("When they visit", fontweight="bold")
ax.legend(fontsize=7, loc="upper right")

# Panel 3 — share of wallet (only when the spend split is available).
ax = axes[0, 2]
if {"target_share", "other_share"}.issubset(profile.columns):
    mix = [("target_share", "Target category", "#E76F51"),
           ("secondary_share", "Secondary", "#264653"),
           ("other_share", "Everything else", "#8AB17D")]
    bottom = np.zeros(n_seg)
    for col, label, color in mix:
        if col not in profile.columns:
            continue
        vals = profile[col].values
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        for i, v in enumerate(vals):
            if v > 0.06:
                ax.text(i, bottom[i] + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += vals
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Where their dollar goes", fontweight="bold")
    ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
else:
    ax.text(0.5, 0.5, "Share-of-wallet split not available\n(run 01 with spend columns)",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_axis_off()

# Bottom row — one metric per panel, real units, value labeled on each bar.
def metric_panel(ax, col, title, fmt, unit=""):
    vals = profile[col].values
    for b, v in zip(ax.bar(x, vals, color=seg_colors), vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(unit)
    ax.spines[["top", "right"]].set_visible(False)

metric_panel(axes[1, 0], "avg_ticket", "Average category ticket", "${:.2f}", "$ per visit")
metric_panel(axes[1, 1], "recency_days", "Days since last visit (avg)", "{:.0f}d", "days")
metric_panel(axes[1, 2], "discount", "Visits with a discount", "{:.0%}", "share of visits")

plt.suptitle("Customer Segmentation — Profile Overview", fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.show()
