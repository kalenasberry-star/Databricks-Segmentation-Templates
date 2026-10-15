# Behavioral Segmentation Template for Databricks

A production-shaped, three-notebook template for **K-Means behavioral segmentation of a
product category's customers** on Databricks feature engineering, model training with
a defensible k-selection policy, and profiling/activation output for a campaign tool
(Braze, Salesforce MC, etc.).

Built from a real consulting engagement on **1.3B+ transaction rows**, then genericized.
Every guard, assertion, and design note in this repo exists because its absence bit us
once. The goal is that the next team answers the hard questions on day one instead of
week six.

## What questions does this answer?

The technical ones  *how many segments exist, are they real, who is in each one*  and,
just as important, the **stakeholder questions** that every segmentation project gets
asked. Those are collected with ready answers in
[`docs/STAKEHOLDER_QA.md`](docs/STAKEHOLDER_QA.md), and the questions to settle *before*
writing any code are in [`docs/DAY_ONE_CHECKLIST.md`](docs/DAY_ONE_CHECKLIST.md).

## Repository layout

```
notebooks/
  01_feature_engineering.py          lines → visits → customer features → validated Delta table
  02_kmeans_training.py              k sweep, selection policy, OOS validation, stability, MLflow + UC
  03_segment_profiling_activation.py score everyone, profiles + auto-named segments, activation table, charts
docs/
  STAKEHOLDER_QA.md                  the questions you WILL be asked, with answers
  DAY_ONE_CHECKLIST.md               access, definitions, and data questions to settle before coding
```

Notebooks are in Databricks **source format**  import them directly
(Workspace → Import → File) and they render with markdown cells.

## Quickstart

1. **Provision access first** (see the checklist  this is the #1 schedule risk):
   ```sql
   CREATE SCHEMA IF NOT EXISTS main.segmentation;
   GRANT USE SCHEMA, CREATE TABLE, MODIFY, SELECT, CREATE MODEL
     ON SCHEMA main.segmentation TO `your_user_or_group`;
   ```
2. **Run `01_feature_engineering`.** Set the widgets: your source table, target
   category (e.g. `Hot Food`), optional secondary cross-shop category (e.g. `Fuel`),
   lookback window, and eligibility thresholds. It publishes `customer_features`
   through a stage → validate → publish flow that refuses to publish a broken table.
3. **Run `02_kmeans_training`.** It fits candidate models on a 50K sample, validates
   on the held-out majority, applies the written selection policy, checks stability,
   and logs/registers the champion (`@champion` alias) in MLflow / Unity Catalog.
4. **Run `03_segment_profiling_activation`.** It scores every eligible customer with
   the champion, prints each segment's evidence with auto-suggested names, and
   publishes `customer_segments`  the table your campaign tool syncs from. After
   your naming session, fill in `SEGMENT_NAMES` and re-run; guard assertions verify
   the names match the behavior (the chart cannot lie).

Order matters: 01 → 02 → 03. Notebook 03 is signature-driven  it asks the model
which features it needs  so 01/02/03 can only drift apart loudly, never silently.

## Design principles (the short version)

- **Visit grain before customer grain.** Pairing and "was this visit discounted" only
  exist at the visit level; compute facts where they are knowable.
- **The discount-row guard.** Many POS systems store discounts as separate zero-dollar
  rows that inherit the item's category. Count them as items and basket metrics
  inflate ~30%  enough to manufacture a phantom segment. Also: discount amounts are
  usually **negative**, so the promo test is `!= 0`, never `> 0`.
- **Sanity caps.** Without them, a handful of outlier transactions earn their own
  0.0%-share micro-cluster at every k and poison the sweep.
- **Fit on a sample, validate out-of-sample, score everyone.** Clustering has no
  labels to leak; the held-out majority proves the structure generalizes, and every
  customer still gets a segment (an unscored customer is just untargetable).
- **k is a policy, not an argmax.** Viability floor (≥3% share), plan-range
  preference, and a noise tolerance  logged with the run, auditable forever.
- **Scaling lives inside the model pipeline**, never in the feature table, so the
  champion carries its own preprocessing and refreshes can't cause train/serve skew.
- **Names are annotation, and they are guarded.** One dictionary, joined by
  `segment_id` everywhere, plus assertions: dict size must match the model's k, no
  unmapped ids, and a segment *named* "Fading" must *be* the max-recency segment.
  (This bug class bit us three times before the assertions. Zero times after.)
- **Reporting columns ride along, model columns are curated.** Total spend, raw
  window counts, and purchase-gap metrics stay out of the model vector  each
  exclusion is commented with the reason, so the model layer never re-litigates it.
- **Pin your runs.** Features recompute from a rolling window, so every run sees a
  slightly different universe. Train against a snapshot, cite one champion run ID in
  every chart/table/doc, and never retrain the week of a stakeholder meeting.

## Requirements & environment notes

- Databricks ML Runtime (sklearn, MLflow) or serverless notebooks + `%pip` as needed.
- **Serverless-safe**: no `.cache()`/persist (unsupported there); reuse is handled by
  materializing Delta tables, which is the better architecture anyway.
- Unity Catalog registration needs `CREATE MODEL` on the output schema; the notebook
  degrades gracefully (logs the run URI) if the grant is missing.

## Customization points

| What | Where |
|---|---|
| Target / secondary category | widgets in 01 (and 02's `secondary_category`) |
| Daypart windows | `daypart_expr` in 01  get stakeholder sign-off, then encode exactly |
| Eligibility thresholds | widgets in 01 |
| Selection policy | `MIN_SEGMENT_SHARE`, `SIL_TOLERANCE`, `preferred_k_range` in 02 |
| Archetype rules | `suggest()` in 03  a field guide, not a roster |
| Activation attributes | Cell 4 of 03  curate what syncs; band, don't ship raw floats |

## License

MIT  see [LICENSE](LICENSE). Use it, fork it, ship it.
