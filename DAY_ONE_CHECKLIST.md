# Day-One Checklist — settle these before writing any code

Every item below was eventually asked on the engagement this template comes from.
The lesson is only the word *eventually*. Work the list in week zero and the project
loses its two biggest schedule risks: access and definitions.

## 1. Access & environment (the #1 schedule risk)

- [ ] **A writable schema exists before kickoff**, with these grants to the build
      identity (run the smoke test the same day — read a source, write a table,
      log a model):
  ```sql
  CREATE SCHEMA IF NOT EXISTS <catalog>.<project_schema>;
  GRANT USE SCHEMA, CREATE TABLE, MODIFY, SELECT, CREATE MODEL
    ON SCHEMA <catalog>.<project_schema> TO `<user_or_group>`;
  ```
- [ ] `SELECT` confirmed on every source schema you'll read.
- [ ] **Compute type named** (serverless vs classic) and its constraints understood —
      e.g. serverless has no `.cache()`, which pushes you (correctly) toward
      materializing Delta tables for reuse.
- [ ] A named platform admin with an SLA for grant requests.

*Cost of skipping: weeks of read-only workarounds, feature logic copy-pasted into
multiple notebooks "temporarily," and deferred model registration.*

## 2. Definitions & scope (get them approved in writing)

- [ ] **Which exact table and filter produced the headline number in the proposal/SOW?**
      Inherit it verbatim — it anchors the universe, the category definition, and the
      client's trust in your counts.
- [ ] The **governed category definition** (ideally a single column equality like
      `category = 'X'`, not a name-pattern approximation), signed off by the data owner.
- [ ] **Daypart brackets** proposed with exact boundaries and boundary-minute rule
      (e.g. 10:30:00 belongs to lunch), approved in writing. One email, zero rework.
- [ ] **Eligibility thresholds** (min purchases, recency rule) and what happens to
      everyone excluded (lapsed → rule-based win-back audience; rare buyers → optional
      rule-based trial audience).
- [ ] A named **data owner** for field definitions, and a named **identity owner** for
      the customer-id → activation-tool mapping, each with a date.

## 3. Data quirks pass (profile before you trust)

Ask to see *rows*, not documentation, for each of these:

- [ ] **How are discounts physically stored?** Inline adjustments, or separate
      zero-dollar rows that inherit the item's category? (The latter inflates basket
      metrics ~30% and can manufacture a phantom segment without a guard.)
- [ ] **Sign conventions** — discount amounts are usually negative; a `> 0` test
      silently never fires.
- [ ] **Refunds and voids** — negative sales rows, or a transaction type? What's the
      netting rule?
- [ ] **Timestamps: store-local or UTC?** If UTC, where is the store→timezone table?
      (Your daypart features are wrong by hours otherwise. Sanity check: does the
      lunch peak land at lunch?)
- [ ] **Outlier scale** — what does a catering-sized or system-error transaction look
      like? Set the sanity caps accordingly; without them K-Means hands outliers
      their own micro-cluster at every k.
- [ ] Known **system/test accounts** to exclude.
- [ ] Customer identity: is there a **merged person-level id** (one person, two cards
      = one customer), or do you need to build the bridge?

## 4. Reproducibility policy (decide it, don't discover it)

- [ ] Features recompute from a **rolling window** → every run sees a different
      universe. Decide the snapshot cadence for modeling now.
- [ ] One **canonical champion run ID** cited in every chart, table, and document.
      If two artifacts cite different IDs, one of them is wrong.
- [ ] Retraining is a **reviewed event** with centroid/name matching — never a casual
      re-run, and never the week of a stakeholder meeting.

## 5. Measurement & activation (design it on day one — lift is the product)

- [ ] Who owns the **control group**, where does it live, and is it **globally
      excluded** from every send (not per-campaign control variants)?
- [ ] Does control membership **flow back** to the data platform for the lift math?
- [ ] **Power check**: control count *inside your smallest segment* — small segments
      read slowly by design; set that expectation in the brief.
- [ ] **What else will be marketed to these customers** during the measurement window?
      Get the campaign calendar; competing promotions contaminate lift.
- [ ] Which identifier is the activation tool's `external_id`, verified against ~100
      real profiles **before** any sync (unmatched ids create ghost profiles,
      silently)?
- [ ] **Contactability**: segments are sized in customers; campaigns are sized in
      opted-in, reachable humans. Get the opt-in share before promising audience sizes.
- [ ] Attribute naming agreed now = the names the campaign tool keeps forever; band
      values for sync (categoricals over raw floats) — cheaper and more usable.

## 6. Nice-to-haves that make the model better (request in the SOW)

- Margin by item/category (segments optimized on spend; the business runs on margin)
- Promo *type* detail (loyalty offer vs storewide markdown vs mix-match)
- Store format / geography / timezone table
- Campaign-engagement flow-back (opens, clicks, redemptions) to close the loop
