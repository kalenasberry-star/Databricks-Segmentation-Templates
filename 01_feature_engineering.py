# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Feature Engineering (Behavioral Segmentation Template)
# MAGIC
# MAGIC Turns raw transaction lines into a **customer-level behavioral feature table**
# MAGIC for K-Means segmentation of a target product category (e.g. "Hot Food",
# MAGIC "Coffee", "Bakery" — configurable via widget).
# MAGIC
# MAGIC **Flow:** source lines → normalize columns → clean & classify →
# MAGIC transaction (visit) grain → customer grain → validate → publish.
# MAGIC
# MAGIC **Design principles baked in (learned the hard way):**
# MAGIC 1. *Visit grain before customer grain* — some facts (category pairing,
# MAGIC    "was this visit discounted") only exist at the visit level.
# MAGIC 2. *Discount-row guard* — many POS systems store discounts as separate
# MAGIC    zero-dollar rows that inherit the item's category. Counting them as
# MAGIC    items inflates basket metrics (~30%+) and can manufacture phantom
# MAGIC    segments. Only rows with positive sales count as items.
# MAGIC 3. *Sanity caps* — a handful of outlier transactions (catering-scale
# MAGIC    orders, data glitches) will otherwise earn their own micro-cluster.
# MAGIC 4. *Stage → validate → publish* — a table that fails validation is never
# MAGIC    published; the staging copy is left behind for inspection.
# MAGIC 5. *Serverless-safe* — no `.cache()`/persist; reuse is handled by
# MAGIC    materializing Delta tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Parameters
# MAGIC Everything adjustable is a widget so environment changes are parameter
# MAGIC edits, not code edits. Review the daypart brackets and eligibility
# MAGIC thresholds with your business stakeholders **before** the first run and
# MAGIC record their approval — definitions you didn't get approved are
# MAGIC definitions you will implement twice.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -- Locations -----------------------------------------------------------------
dbutils.widgets.text("source_table", "main.raw.pos_transaction_lines")
dbutils.widgets.text("output_schema", "main.segmentation")
dbutils.widgets.text("feature_table_name", "customer_features")

# -- Business definitions ------------------------------------------------------
dbutils.widgets.text("target_category", "Hot Food")   # the category being segmented
dbutils.widgets.text("secondary_category", "")        # optional cross-shop category
                                                      # (e.g. "Fuel"); blank = disabled
dbutils.widgets.text("window_months", "18")           # behavioral lookback
dbutils.widgets.text("min_target_txns", "5")          # eligibility: min category purchases
dbutils.widgets.text("recency_days", "90")            # eligibility: active within N days (0 = off)

SOURCE_TABLE  = dbutils.widgets.get("source_table")
OUT_SCHEMA    = dbutils.widgets.get("output_schema")
FEATURE_TABLE = f"{OUT_SCHEMA}.{dbutils.widgets.get('feature_table_name')}"
TARGET_CAT    = dbutils.widgets.get("target_category")
SECONDARY_CAT = dbutils.widgets.get("secondary_category").strip() or None
WINDOW_MONTHS = int(dbutils.widgets.get("window_months"))
MIN_TXNS      = int(dbutils.widgets.get("min_target_txns"))
RECENCY_DAYS  = int(dbutils.widgets.get("recency_days")) or None

# -- Data-quality caps (tune to your business; see README) ---------------------
MAX_LINE_AMOUNT = 1000.0   # single line above this is treated as an outlier/error
MAX_LINE_QTY    = 100.0

# -- Dayparts: 6 windows at MINUTE precision. Boundary minute belongs to the
#    NEW window (10:30:00 => lunch). Adjust to your approved definitions. ------
DAYPARTS = ["early_morning", "breakfast", "lunch", "afternoon_snack", "dinner", "late_night"]

def daypart_expr(ts_col):
    """Map a timestamp to a daypart label using minutes-since-midnight,
    so half-hour boundaries are exact. `otherwise` covers the overnight
    wrap (20:00–03:59)."""
    m = F.hour(ts_col) * 60 + F.minute(ts_col)
    return (F.when((m >= 4 * 60) & (m < 7 * 60), "early_morning")        # 04:00–06:59
             .when((m >= 7 * 60) & (m < 10 * 60 + 30), "breakfast")      # 07:00–10:29
             .when((m >= 10 * 60 + 30) & (m < 14 * 60), "lunch")         # 10:30–13:59
             .when((m >= 14 * 60) & (m < 16 * 60 + 30), "afternoon_snack")  # 14:00–16:29
             .when((m >= 16 * 60 + 30) & (m < 20 * 60), "dinner")        # 16:30–19:59
             .otherwise("late_night"))                                   # 20:00–03:59

print(f"Target category: {TARGET_CAT!r} | secondary: {SECONDARY_CAT!r} | "
      f"window: {WINDOW_MONTHS}mo | eligibility: >={MIN_TXNS} txns"
      + (f", active <= {RECENCY_DAYS}d" if RECENCY_DAYS else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Normalize, clean, classify (line grain)
# MAGIC A case-insensitive column map lets the same notebook run against table
# MAGIC variants with different casing. Required columns fail **once** with the
# MAGIC full list of what's missing; optional columns degrade with a warning.

# COMMAND ----------

src = spark.table(SOURCE_TABLE)
have = {c.lower(): c for c in src.columns}

def pick(*candidates):
    """Return the first matching source column (case-insensitive), else None."""
    return next((F.col(have[c.lower()]) for c in candidates if c.lower() in have), None)

missing, warnings = [], []

def required(label, *candidates):
    col = pick(*candidates)
    if col is None:
        missing.append(f"{label} (looked for: {', '.join(candidates)})")
    return col

def optional(label, default, *candidates):
    col = pick(*candidates)
    if col is None:
        warnings.append(f"{label} not found — defaulting to {default!r}; dependent features will be flat")
        return F.lit(default)
    return col

cust_c  = required("customer id", "customer_id", "wallet_id", "member_id", "loyalty_id")
txn_c   = required("transaction id", "transaction_id", "txn_id", "order_id")
date_c  = required("business date", "business_date", "transaction_date_key", "sale_date")
ts_c    = required("transaction timestamp", "transaction_ts", "transaction_datetime", "transaction_date")
amt_c   = required("sales amount", "sales_amount", "sale_amt", "sales_volume", "net_sales")
cat_c   = required("product category", "product_category", "item_category", "item_gl_category")
disc_c  = optional("discount amount", 0.0, "discount_amount", "discount_amt")
qty_c   = optional("quantity", 1.0, "quantity", "sale_qty", "sales_quantity")

if missing:
    raise RuntimeError("Required columns not found in source:\n  - " + "\n  - ".join(missing))
for w in warnings:
    print("WARNING:", w)

base = (
    src.select(
        cust_c.alias("customer_id"),
        txn_c.alias("transaction_id"),
        date_c.cast("date").alias("business_date"),
        ts_c.cast("timestamp").alias("txn_ts"),
        amt_c.cast("double").alias("sales_amount"),
        disc_c.cast("double").alias("discount_amount"),
        qty_c.cast("double").alias("quantity"),
        cat_c.cast("string").alias("product_category"),
    )
    # Lookback window — rolling, anchored to run date. For reproducible model
    # training, snapshot the OUTPUT table per modeling cycle (see README).
    .filter(F.col("business_date") >= F.add_months(F.current_date(), -WINDOW_MONTHS))
    .filter(F.col("customer_id").isNotNull())
    # Sanity caps: keep the universe, drop the data errors.
    .filter((F.col("sales_amount") >= 0)
            & (F.col("sales_amount") < MAX_LINE_AMOUNT)
            & (F.coalesce(F.col("quantity"), F.lit(1.0)) < MAX_LINE_QTY))
    # --- Row typing (the discount-row guard) ----------------------------------
    # ITEM rows carry positive sales; DISCOUNT rows are zero-dollar lines with a
    # nonzero (usually negative) discount amount. A row can be both when the
    # discount is stored inline on the item line. Classification uses ITEM rows
    # only; the promo flag uses ANY nonzero discount, covering both encodings.
    .withColumn("is_item_row", F.col("sales_amount") > 0)
    .withColumn("has_discount", F.col("discount_amount") != 0)   # NOT `> 0`: stored negative
    .filter(F.col("is_item_row") | F.col("has_discount"))
    .withColumn("is_target", F.col("is_item_row") & (F.col("product_category") == TARGET_CAT))
    .withColumn("is_secondary",
                F.col("is_item_row") & (F.col("product_category") == F.lit(SECONDARY_CAT))
                if SECONDARY_CAT else F.lit(False))
    .withColumn("daypart", daypart_expr(F.col("txn_ts")))
    .withColumn("is_weekend", F.dayofweek("txn_ts").isin([1, 7]))   # Sun=1, Sat=7
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Share of wallet (customer grain, over ALL spend)
# MAGIC Where the customer's money goes across their *entire* relationship, not
# MAGIC just target-category visits. This is what separates "big spender in the
# MAGIC category" from "big customer who barely touches the category".

# COMMAND ----------

wallet_spend = (
    base.filter("is_item_row")
    .groupBy("customer_id")
    .agg(
        F.sum(F.when(F.col("is_target"), F.col("sales_amount")).otherwise(0.0)).alias("target_spend"),
        F.sum(F.when(F.col("is_secondary"), F.col("sales_amount")).otherwise(0.0)).alias("secondary_spend"),
        F.sum(F.when(~F.col("is_target") & ~F.col("is_secondary"),
                     F.col("sales_amount")).otherwise(0.0)).alias("other_spend"),
    )
    .withColumn("total_spend", F.col("target_spend") + F.col("secondary_spend") + F.col("other_spend"))
    # F.greatest(.., 0.01) guards divide-by-zero without null-handling branches.
    .withColumn("target_spend_share", F.col("target_spend") / F.greatest("total_spend", F.lit(0.01)))
    .withColumn("secondary_spend_share", F.col("secondary_spend") / F.greatest("total_spend", F.lit(0.01)))
    .withColumn("other_spend_share", F.col("other_spend") / F.greatest("total_spend", F.lit(0.01)))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Transaction (visit) grain
# MAGIC One row per qualified visit: a transaction containing **at least one
# MAGIC target-category item**. Pairing ("secondary category in the same visit")
# MAGIC and "was a deal involved" are visit-level facts — computed here, where
# MAGIC they are knowable, then averaged at the customer level.

# COMMAND ----------

txn = (
    base.groupBy("customer_id", "business_date", "transaction_id")
    .agg(
        F.max("is_target").alias("has_target"),
        F.max("is_secondary").alias("secondary_attached"),
        F.min(F.when(F.col("is_item_row"), F.col("txn_ts"))).alias("txn_ts"),
        F.first("daypart").alias("daypart"),
        F.max("is_weekend").alias("is_weekend"),
        F.max("has_discount").alias("visit_discounted"),
        F.sum(F.when(F.col("is_target"), F.col("sales_amount"))).alias("target_amount"),
        F.count(F.when(F.col("is_target"), F.lit(1))).alias("target_item_count"),
    )
    .filter("has_target AND target_amount > 0")   # the category-qualification rule
)

# --- Inter-purchase gaps (reporting-only; drives "send at M+N days" triggers) --
# distinct() first: two purchases on the same DAY must not create a 0-day gap.
# The first visit's lag is null and is filtered — never coalesced to zero.
visit_w = Window.partitionBy("customer_id").orderBy("business_date")
purchase_gaps = (
    txn.select("customer_id", "business_date").distinct()
    .withColumn("prev_date", F.lag("business_date").over(visit_w))
    .withColumn("gap_days", F.datediff("business_date", "prev_date"))
    .filter(F.col("gap_days").isNotNull())
    .groupBy("customer_id")
    .agg(F.avg("gap_days").alias("avg_days_between_purchases"),
         F.percentile_approx("gap_days", 0.5).alias("median_days_between_purchases"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Customer features (one aggregation pass)
# MAGIC All feature families in a single conditional-aggregation pass — no pivots,
# MAGIC no join chains. Columns kept **out of the model vector** on purpose:
# MAGIC `target_total_spend` (≈ frequency × ticket: including it clusters on wallet
# MAGIC size, not behavior), raw 30/60/90 counts (their signal travels through
# MAGIC `freq_trend`), one daypart share (six shares sum to 1), and the gap
# MAGIC metrics (reporting/activation only).

# COMMAND ----------

ref = F.current_date()   # recency anchor; windows below are trailing from today

features = (
    txn.groupBy("customer_id")
    .agg(
        # Frequency family -----------------------------------------------------
        F.count("*").alias("target_txn_count"),
        F.countDistinct("business_date").alias("active_days"),
        F.sum(F.when(F.datediff(ref, F.col("business_date")) <= 30, 1).otherwise(0)).alias("txns_30d"),
        F.sum(F.when(F.datediff(ref, F.col("business_date")) <= 90, 1).otherwise(0)).alias("txns_90d"),
        # Basket family ----------------------------------------------------------
        F.avg("target_amount").alias("avg_ticket"),
        F.sum("target_amount").alias("target_total_spend"),
        F.avg("target_item_count").alias("avg_items_per_txn"),
        # Recency ----------------------------------------------------------------
        F.datediff(ref, F.max("business_date")).alias("recency_days"),
        # Timing family ----------------------------------------------------------
        *[F.sum(F.when(F.col("daypart") == d, 1).otherwise(0)).alias(f"n_{d}") for d in DAYPARTS],
        F.sum(F.when(F.col("is_weekend"), 1).otherwise(0)).alias("n_weekend"),
        # Behavior flags (visit-level facts averaged to rates) -------------------
        F.avg(F.col("secondary_attached").cast("double")).alias("secondary_pairing_pct"),
        F.avg(F.col("visit_discounted").cast("double")).alias("pct_discount"),
    )
    # Daypart COUNTS → SHARES of the customer's own visits (shares keep the
    # time-of-day signal independent of overall frequency).
    .select("*", *[(F.col(f"n_{d}") / F.col("target_txn_count")).alias(f"pct_{d}") for d in DAYPARTS])
    .withColumn("pct_weekend", F.col("n_weekend") / F.col("target_txn_count"))
    # log1p tames the heavy right tail of frequency so it can't dominate the
    # K-Means distance math on its own.
    .withColumn("log_txn_count", F.log1p("target_txn_count"))
    # Momentum: 30-day pace vs 90-day pace. ~1 steady, >1 accelerating, ~0
    # stopped. Flags a fading regular WEEKS before recency alone moves.
    # Cap at 3 bounds brand-new customers; txns_90d >= 1 for every eligible
    # customer (recency rule), so the ratio is always defined.
    .withColumn("freq_trend",
                F.least(F.lit(3.0), (F.col("txns_30d") * 3.0) / F.greatest(F.col("txns_90d"), F.lit(1))))
    .join(wallet_spend, "customer_id", "left")
    .join(purchase_gaps, "customer_id", "left")     # null = <2 distinct visit days (rare, valid)
    .fillna(0.0, subset=["target_spend", "secondary_spend", "other_spend", "total_spend",
                         "target_spend_share", "secondary_spend_share", "other_spend_share"])
    .withColumn("eligible_for_modeling",
                (F.col("target_txn_count") >= MIN_TXNS)
                & ((F.col("recency_days") <= RECENCY_DAYS) if RECENCY_DAYS else F.lit(True)))
    .withColumn("ref_date", ref)
    .withColumn("feature_ts", F.current_timestamp())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Validate on staging, then publish atomically
# MAGIC The expensive aggregation is computed **once** (the staging write); every
# MAGIC check runs against the written data; the final table is replaced only on
# MAGIC a clean pass. A failed run leaves `_staging` behind for inspection and
# MAGIC never touches the published table.

# COMMAND ----------

# Keep this list in sync with MODEL_FEATURES in notebook 02.
VALIDATED_FEATURES = (
    ["log_txn_count", "freq_trend", "avg_ticket", "avg_items_per_txn",
     "active_days", "recency_days", "pct_discount", "pct_weekend"]
    + [f"pct_{d}" for d in DAYPARTS]
    + (["secondary_pairing_pct", "secondary_spend_share", "other_spend_share"] if SECONDARY_CAT else [])
)

STAGING = f"{FEATURE_TABLE}_staging"
(features.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(STAGING))
staged = spark.table(STAGING)

checks = staged.agg(
    F.count("*").alias("n"),
    F.countDistinct("customer_id").alias("n_customers"),
    F.sum(F.when(sum(F.col(f"pct_{d}") for d in DAYPARTS).between(1 - 1e-6, 1 + 1e-6), 0)
           .otherwise(1)).alias("bad_daypart_rows"),
    *[F.sum(F.col(c).isNull().cast("int")).alias(f"null__{c}") for c in VALIDATED_FEATURES],
).collect()[0].asDict()

problems = []
if checks["n"] != checks["n_customers"]:
    problems.append(f"{checks['n'] - checks['n_customers']} duplicate customer_id rows")
if checks["bad_daypart_rows"]:
    problems.append(f"{checks['bad_daypart_rows']} rows whose daypart shares don't sum to 1")
nulls = {k[6:]: v for k, v in checks.items() if k.startswith("null__") and v}
if nulls:
    problems.append(f"nulls in feature columns: {nulls}")

# Row-count tripwire: a >20% swing vs the prior run usually means an upstream
# feed or filter changed, not that the customer base did.
HISTORY = f"{FEATURE_TABLE}_rowcount_history"
if spark.catalog.tableExists(HISTORY):
    prior_rows = spark.table(HISTORY).orderBy(F.desc("run_ts")).limit(1).collect()
    if prior_rows and abs(checks["n"] - prior_rows[0]["row_count"]) / prior_rows[0]["row_count"] > 0.20:
        problems.append(f"row count swung >20% vs prior run ({prior_rows[0]['row_count']:,} -> {checks['n']:,})")

if problems:
    raise ValueError("FEATURE CONTRACT VIOLATION — fix at the feature stage, not downstream:\n  - "
                     + "\n  - ".join(problems) + f"\n(staging left for inspection: {STAGING})")

spark.sql(f"CREATE OR REPLACE TABLE {FEATURE_TABLE} AS SELECT * FROM {STAGING}")
spark.sql(f"DROP TABLE IF EXISTS {STAGING}")
(spark.createDataFrame([(checks["n"],)], "row_count long")
 .withColumn("run_ts", F.current_timestamp())
 .write.mode("append").saveAsTable(HISTORY))
spark.sql(
    f"COMMENT ON TABLE {FEATURE_TABLE} IS "
    f"'Behavioral features for {TARGET_CAT} segmentation. Grain: 1 row per customer_id. "
    f"{WINDOW_MONTHS}-month window; eligibility >= {MIN_TXNS} txns"
    f"{f', active <= {RECENCY_DAYS}d' if RECENCY_DAYS else ''}. "
    f"Raw values only: scaling lives inside the model pipeline (notebook 02).'"
)

eligible = staged.filter("eligible_for_modeling").count()
print(f"Published {FEATURE_TABLE}: {checks['n']:,} customers "
      f"({eligible:,} eligible for modeling, {eligible / max(checks['n'], 1):.0%}). All checks passed.")
