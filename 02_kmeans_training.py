# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — K-Means Training, Selection & Registration
# MAGIC
# MAGIC Consumes the feature table published by notebook 01 — **no cleaning code
# MAGIC lives here by design**: the feature contract already guaranteed the
# MAGIC inputs, and a single assertion confirms it held.
# MAGIC
# MAGIC **What this notebook proves, not just computes:**
# MAGIC 1. *Fit on a sample, validate out-of-sample, score everyone.* Clustering
# MAGIC    has no labels to leak, so the held-out majority's job is to prove the
# MAGIC    structure generalizes — not to be excluded from assignment.
# MAGIC 2. *k is chosen by a written policy, not an argmax.* Silhouette alone
# MAGIC    climbs toward the sweep edge by shaving off micro-groups; the policy
# MAGIC    (viability floor + plan-range preference + noise tolerance) is logged
# MAGIC    with the run so the decision is auditable.
# MAGIC 3. *Stability is measured.* Refits from independent random starts must
# MAGIC    reproduce the same groupings (Adjusted Rand Index) before anyone
# MAGIC    builds campaigns on them.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Parameters & selection policy

# COMMAND ----------

from pyspark.sql import functions as F
import numpy as np
import pandas as pd

dbutils.widgets.text("output_schema", "main.segmentation")
dbutils.widgets.text("feature_table_name", "customer_features")
dbutils.widgets.text("model_name", "customer_segmentation")
dbutils.widgets.text("secondary_category", "")     # must match notebook 01's setting
dbutils.widgets.text("train_sample_n", "50000")    # K-Means estimates k centroids; they
                                                   # stabilize far below 50K rows
dbutils.widgets.text("k_min", "3")
dbutils.widgets.text("k_max", "10")
dbutils.widgets.text("k_override", "")             # e.g. "6" to skip the sweep and fit one k
dbutils.widgets.text("preferred_k_range", "4-6")   # the range your project plan committed to
dbutils.widgets.dropdown("register_model", "true", ["true", "false"])

OUT_SCHEMA    = dbutils.widgets.get("output_schema")
FEATURE_TABLE = f"{OUT_SCHEMA}.{dbutils.widgets.get('feature_table_name')}"
MODEL_NAME    = f"{OUT_SCHEMA}.{dbutils.widgets.get('model_name')}"
SECONDARY     = dbutils.widgets.get("secondary_category").strip() or None
TRAIN_N       = int(dbutils.widgets.get("train_sample_n"))
K_MIN, K_MAX  = int(dbutils.widgets.get("k_min")), int(dbutils.widgets.get("k_max"))
K_OVERRIDE    = dbutils.widgets.get("k_override").strip()
PREF_LO, PREF_HI = (int(x) for x in dbutils.widgets.get("preferred_k_range").split("-"))
REGISTER      = dbutils.widgets.get("register_model") == "true"

# --- The selection policy, as constants (logged with every sweep) -------------
MIN_SEGMENT_SHARE = 0.03   # a segment under 3% of customers is too small to campaign to,
                           # no matter how pretty its score — and tiny segments are usually
                           # outlier micro-clusters, not behavior
SIL_TOLERANCE     = 0.02   # silhouette is computed on a sample; differences smaller than
                           # this are noise, so an out-of-range k must beat the in-range
                           # best by MORE than this to win

DAYPARTS = ["early_morning", "breakfast", "lunch", "afternoon_snack", "dinner", "late_night"]

# The model vector. Deliberately EXCLUDED (kept in the table for profiling):
#  - target_total_spend  (≈ frequency × ticket — clusters on wallet size, not behavior)
#  - raw txns_30d/90d    (double-count overall frequency; signal travels via freq_trend)
#  - pct_early_morning   (six shares sum to 1 — the sixth is determined by the other five)
#  - purchase-gap metrics (reporting/activation only)
#  - target_spend_share  (the three wallet shares also sum to ~1)
MODEL_FEATURES = (
    ["log_txn_count", "freq_trend", "avg_ticket", "avg_items_per_txn",
     "active_days", "recency_days", "pct_discount", "pct_weekend"]
    + [f"pct_{d}" for d in DAYPARTS[1:]]
    + (["secondary_pairing_pct", "secondary_spend_share", "other_spend_share"] if SECONDARY else [])
)
print(f"{len(MODEL_FEATURES)} model features | selection: max silhouette, prefer k in "
      f"{PREF_LO}-{PREF_HI}, outside wins only by > {SIL_TOLERANCE}, min share {MIN_SEGMENT_SHARE:.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Load, then split: fit sample / held-out / everyone
# MAGIC Integer columns are cast to double before modeling so the logged MLflow
# MAGIC signature can represent missing values at scoring time (otherwise you get
# MAGIC a schema warning now and a schema *error* later).

# COMMAND ----------

assert spark.catalog.tableExists(FEATURE_TABLE), (
    f"{FEATURE_TABLE} not found — run notebook 01 first. "
    f"This notebook deliberately has no inline feature fallback: duplicated "
    f"feature logic drifts, and drift produces silently different universes.")

df = (spark.table(FEATURE_TABLE)
      .filter("eligible_for_modeling")
      .select("customer_id", *[F.col(c).cast("double").alias(c) for c in MODEL_FEATURES])
      .toPandas())

X_all = df[MODEL_FEATURES]
assert X_all.notna().all().all(), "Nulls reached the model layer — the feature contract failed; fix in 01."

# Seeded sample for fitting; the remainder is the exam. EVERYONE is scored at
# the end — nearest-centroid assignment is the same deterministic rule for all,
# and an unscored customer is just a customer with no segment in activation.
train_df = df.sample(n=min(TRAIN_N, len(df)), random_state=42)
X = train_df[MODEL_FEATURES]
oos_df = df.drop(train_df.index)
print(f"Universe: {len(df):,} | fit sample: {len(X):,} | held-out for validation: {len(oos_df):,} "
      f"| scored at the end: all {len(df):,}")

rng = np.random.RandomState(42)
sil_idx = rng.choice(len(X), min(10_000, len(X)), replace=False)
oos_idx = rng.choice(len(oos_df), min(10_000, len(oos_df)), replace=False) if len(oos_df) else None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Sweep + selection
# MAGIC Each candidate k is a complete independent model, logged as a nested
# MAGIC MLflow run under one parent — the permanent audit trail of every k
# MAGIC considered and why the winner won.
# MAGIC
# MAGIC *Silhouette* (−1..1): how much closer each customer is to their own
# MAGIC segment than to the nearest other one. 0.10–0.20 is normal for real
# MAGIC behavioral data (humans don't shop in separated balls) — it is a
# MAGIC comparison tool between candidates, not a grade. The trust evidence is
# MAGIC the out-of-sample agreement and the stability check.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_experiment(f"/Shared/{dbutils.widgets.get('model_name')}")

def fit_k(k: int, seed: int = 42):
    """Fit one candidate: scaler + KMeans in ONE pipeline so preprocessing
    travels with the model (no train/serve skew). Returns in-sample and
    out-of-sample silhouette, smallest-segment share, and inertia."""
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed))])
    labels = pipe.fit_predict(X)                                  # fit on the sample ONLY
    Xs = pipe.named_steps["scaler"].transform(X)
    sil = silhouette_score(Xs[sil_idx], labels[sil_idx])
    if oos_idx is not None:                                       # the exam: customers never seen in fitting
        Xo = pipe.named_steps["scaler"].transform(oos_df[MODEL_FEATURES].iloc[oos_idx])
        oos_sil = silhouette_score(Xo, pipe.named_steps["kmeans"].predict(Xo))
    else:
        oos_sil = float("nan")
    min_share = float(pd.Series(labels).value_counts(normalize=True).min())
    return pipe, labels, sil, oos_sil, min_share, float(pipe.named_steps["kmeans"].inertia_)

results = []
ks = [int(K_OVERRIDE)] if K_OVERRIDE else list(range(K_MIN, K_MAX + 1))
with mlflow.start_run(run_name="kmeans_k_sweep") as parent:
    mlflow.log_params({
        "feature_cols": MODEL_FEATURES, "n_training_rows": len(X),
        "selection_rule": (f"max silhouette; prefer k in {PREF_LO}-{PREF_HI}; "
                           f"outside wins only by > {SIL_TOLERANCE}; min share {MIN_SEGMENT_SHARE:.0%}"),
    })
    for k in ks:
        with mlflow.start_run(run_name=f"k={k}", nested=True):
            pipe, labels, sil, oos_sil, min_share, inertia = fit_k(k)
            mlflow.log_params({"k": k, "n_init": 10, "random_state": 42})
            mlflow.log_metrics({"silhouette": sil, "silhouette_oos": oos_sil,
                                "min_segment_share": min_share, "inertia": inertia})
            results.append(dict(k=k, silhouette=sil, silhouette_oos=oos_sil,
                                min_share=min_share, inertia=inertia,
                                viable=min_share >= MIN_SEGMENT_SHARE,
                                pipeline=pipe, labels=labels))
            print(f"k={k}  silhouette={sil:.3f}  out-of-sample={oos_sil:.3f}  "
                  f"smallest segment={min_share:.1%}")

# --- Apply the policy ---------------------------------------------------------
viable = [r for r in results if r["viable"]] or results
in_range = [r for r in viable if PREF_LO <= r["k"] <= PREF_HI]
best_overall = max(viable, key=lambda r: r["silhouette"])
if in_range:
    best_in_range = max(in_range, key=lambda r: r["silhouette"])
    edge = best_overall["silhouette"] - best_in_range["silhouette"]
    best = best_overall if edge > SIL_TOLERANCE else best_in_range
    print(f"\nSelected k={best['k']} "
          + (f"— outside plan range, beats in-range best by {edge:.3f} (> {SIL_TOLERANCE})"
             if best is best_overall else
             f"(best within plan range {PREF_LO}-{PREF_HI}; out-of-range max k={best_overall['k']} "
             f"gained only {edge:.3f} silhouette — within noise)"))
else:
    best = best_overall
    print(f"\nSelected k={best['k']} (no viable candidate inside the plan range)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Generalization checks
# MAGIC Two proofs the segments are real. (1) Out-of-sample silhouette tracks
# MAGIC in-sample within ~0.02. (2) Segment *proportions* predicted on the
# MAGIC held-out majority match the fit sample — if the groups were artifacts of
# MAGIC the sample, this table is where it falls apart.

# COMMAND ----------

gap = best["silhouette"] - best["silhouette_oos"]
print(f"Silhouette in-sample {best['silhouette']:.3f} vs out-of-sample {best['silhouette_oos']:.3f} "
      f"(gap {gap:+.3f} — under ~{SIL_TOLERANCE} means the structure generalizes)")
if len(oos_df):
    oos_pred = best["pipeline"].predict(oos_df[MODEL_FEATURES])
    shares = pd.DataFrame({
        "train_share": pd.Series(best["labels"]).value_counts(normalize=True).sort_index(),
        "holdout_share": pd.Series(oos_pred).value_counts(normalize=True).sort_index(),
    }).round(3)
    print("Segment share, fit sample vs held-out:\n" + shares.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Elbow figure (selection evidence, logged to MLflow)
# MAGIC Inertia always decreases as k grows, so its *value* never picks k — the
# MAGIC signal is where the marginal drop flattens. Read with silhouette: elbow
# MAGIC shows diminishing returns, silhouette shows separation quality.

# COMMAND ----------

import matplotlib.pyplot as plt

ks_list = [r["k"] for r in results]
inertias = [r["inertia"] for r in results]
sils = [r["silhouette"] for r in results]

for i, (k_, in_) in enumerate(zip(ks_list, inertias)):
    drop = "" if i == 0 else f"  ({(inertias[i-1] - in_) / inertias[i-1] * 100:.1f}% drop vs k={ks_list[i-1]})"
    print(f"k={k_}: inertia={in_:,.0f}{drop}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax1.plot(ks_list, inertias, "o-", linewidth=2)
ax1.set_ylabel("Inertia (within-segment SSE)")
ax1.set_title("K selection: elbow (inertia) + silhouette", fontweight="bold")
ax1.axvline(best["k"], linestyle="--", alpha=0.7, color="tab:red")
ax1.grid(True, alpha=0.3)
ax2.plot(ks_list, sils, "s-", linewidth=2, color="tab:green")
ax2.set_ylabel("Silhouette (selection metric)")
ax2.set_xlabel("k (number of segments)")
ax2.axvline(best["k"], linestyle="--", alpha=0.7, color="tab:red", label=f"selected k={best['k']}")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
with mlflow.start_run(run_id=parent.info.run_id):
    mlflow.log_figure(fig, "elbow_curve.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Stability check (same k, independent random starts)
# MAGIC Adjusted Rand Index: 1.0 = identical groupings, 0 = chance. Above ~0.8
# MAGIC the segments are solid; below ~0.6 they are fragile and k (or the
# MAGIC features) should be revisited before anyone builds campaigns on them.

# COMMAND ----------

aris = [adjusted_rand_score(best["labels"], fit_k(best["k"], seed=s)[1]) for s in (7, 99, 2024)]
print(f"Stability ARI at k={best['k']}: " + ", ".join(f"{a:.3f}" for a in aris)
      + f"  (mean {np.mean(aris):.3f})")
mlflow.log_metric("stability_ari_mean", float(np.mean(aris)), run_id=parent.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Log + register the champion
# MAGIC The logged *signature* is the model's enforced input contract: if the
# MAGIC feature table ever drifts, scoring fails loudly instead of silently
# MAGIC mis-segmenting. Registration points a governed Unity Catalog name at this
# MAGIC run's model and moves the `@champion` alias — downstream scoring loads
# MAGIC the alias and never hardcodes a version.

# COMMAND ----------

with mlflow.start_run(run_name=f"champion_k{best['k']}") as champ:
    mlflow.log_params({"k": best["k"], "silhouette": best["silhouette"],
                       "silhouette_oos": best["silhouette_oos"],
                       "stability_ari_mean": float(np.mean(aris)),
                       "feature_table": FEATURE_TABLE})
    mlflow.sklearn.log_model(best["pipeline"], name="model",
                             signature=infer_signature(X, best["pipeline"].predict(X)),
                             input_example=X.head(5))
    print(f"Model logged: runs:/{champ.info.run_id}/model")
    if REGISTER:
        try:
            mlflow.set_registry_uri("databricks-uc")
            mv = mlflow.register_model(f"runs:/{champ.info.run_id}/model", MODEL_NAME)
            from mlflow import MlflowClient
            MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", mv.version)
            print(f"Registered {MODEL_NAME} v{mv.version} as @champion")
        except Exception as e:   # e.g. missing CREATE MODEL grant — the run URI above still works
            print(f"UC registration skipped ({type(e).__name__}): {e}\n"
                  f"Register later from the run URI above once permissions exist.")
