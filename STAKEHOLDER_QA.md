# Stakeholder Q&A — the questions every segmentation project gets asked

Collected from a real engagement. Each answer is written so you can say it out loud in
a review; the technical backing sits one level down in the notebooks.

---

### "Is this based on the past 90/180 days of data?"
No — the behavioral lookback is a configurable window (18 months by default). The
30/60/90-day numbers you may have seen are features *inside* that window measuring
recent pace. The recency rule (active within 90 days) is an **eligibility** filter, a
different layer from the lookback. Three time concepts, three jobs: lookback = what
history is visible; recent windows = momentum; recency rule = who qualifies.

### "Are all metrics category-only? When you say 'last visit 56 days ago', last visit to anything, or to the category?"
Category-only, by design. Recency means days since the last *target-category*
purchase; transaction counts are category transactions. A customer can have bought
fuel yesterday and still show 56 days — which is often the insight: **they didn't
leave the store, they left the category.** The one deliberate exception is the
share-of-wallet split, which looks at *all* spend precisely to tell those stories apart.

### "If a customer isn't in any segment, why not?"
Entry criteria, not an edge-case segment. To be in the universe: a resolvable customer
id, N+ category purchases in the window, and (optionally) a purchase in the last 90
days. Miss any and you're out entirely — no leftover bucket. Everyone who clears the
bar lands in **exactly one** segment, so briefs never collide and results never
double-count. Truly lapsed customers get a separate *rule-based* win-back audience;
there's no recent behavior to cluster them on.

### "Do any segments capture rare/infrequent buyers? Should they?"
No, on purpose. Two purchases in 18 months isn't enough behavior to cluster; forcing
such customers in smears the segments for everyone else. If you want to reach them,
build a rule-based trial audience ("1–4 purchases ever, active member") — targetable,
just not a cluster.

### "Shouldn't you train on a subset? And shouldn't you avoid scoring the training data?"
Half right, and the template does the right half. We fit on a seeded 50K sample —
K-Means only estimates k centroid positions, which stabilize far below 50K rows.
But excluding training customers from *scoring* imports a supervised-learning rule
into a place it doesn't apply: there are no labels to memorize, assignment is the
same nearest-centroid rule for everyone, and skipping them just leaves real customers
with no segment in activation. The rigor lives where it belongs: silhouette measured
on hundreds of thousands of held-out customers the model never saw, plus the
segment-share table (fit sample vs held-out composition matching within fractions of
a point).

### "Is a silhouette of 0.12 good? It seems low."
For real behavioral data, it's normal — the honest band is roughly 0.10–0.20, because
human behavior is continuous; people don't shop in tight, well-separated balls.
Silhouette is a comparison tool *between candidate k values*, not a grade. The trust
evidence is (a) out-of-sample silhouette tracking in-sample within ~0.02 and (b) the
stability ARI.

### "A higher k scored better on the chart — why didn't you pick it?"
Because the gain was inside the noise tolerance of a sampled metric, and the extra
segments were thin slices (often outlier micro-clusters) with nothing new to say to
them. The policy — viability floor, plan-range preference, noise tolerance — was set
*before* the run and is logged with it. Fewer, nameable, campaignable segments beat a
marginally higher score every time. Also watch the smallest-segment column: a share
collapsing toward 0% as k grows is K-Means shaving slivers, not discovering behavior.

### "What does the elbow chart add if silhouette picks k?"
A second, independent instrument. Inertia always decreases with k, so its value never
selects — the *flattening* of the marginal drop shows where added segments stop buying
cohesion. Elbow shows diminishing returns; silhouette shows separation quality. When
both flatten at the same k, you have two lines of evidence, not one.

### "What does 'stability ARI 0.99' mean in plain words?"
Rebuild the model from scratch with different random starting points and 99%+ of
customers land in the same group. The segments are a property of the data, not of the
run. Above ~0.8 is solid; below ~0.6, revisit k or the features before campaigns.

### "Aren't the cluster definitions and the segment names the same thing?"
Same *content*, different layers. The clusters are math — frozen centroids. The
profile is each cluster's descriptive statistics. The name is a human label joined on
`segment_id` afterwards; delete the naming code and every customer's assignment is
byte-identical. That separation is also why names can be corrected without touching
the model — and why the template asserts names against behavior (the segment named
"Fading" must be the max-recency one), because a crossed label renders a confident
wrong chart.

### "Can a guest be assigned one segment but personally lean another way (e.g. in the 'fuel' segment but huge on breakfast)?"
Yes — assignment is to the *nearest* centroid, not a perfect match; every customer
keeps their own feature values. If useful, log each customer's distance to their
centroid and their second-closest segment: "members of A whose runner-up is B" is a
free sub-audience for cross-over offers.

### "If we add columns to the output table, does that change the model?"
No, mechanically. The model scores only the columns in its signature; everything else
in the dataframe is invisible to it. Keep new columns out of the model feature list
and assignments are byte-identical — no retraining. (Curate what syncs to the campaign
tool, though: attributes are billed per user, and marketers filter on bands, not raw
floats.)

### "Can we get average AND median days between purchases, for an 'M + N days' trigger?"
Yes (built into notebook 01), and drive the trigger on the **median**: purchase gaps
are heavily skewed, so one vacation-sized gap drags the average up while the real
rhythm might be 9 days. Three implementation details that matter: dedupe to one row
per customer per day first (same-day repeats must not create 0-day gaps), never
coalesce the first visit's null gap to zero, and left-join so nobody drops. One design
flag: "median + 30" is a patient trigger — a 10-day-rhythm customer gets the message
at day 40, well into fading — so test "median + 15" or "median × 2" variants too.

### "Why did the numbers/names change from the last version?"
Usually one of two honest reasons. (1) A data correction — e.g. discovering that
discounts are stored as separate zero-dollar rows inflating basket counts ~30% and
manufacturing a phantom segment; the corrected view is the trustworthy one. (2) Drift —
features recompute from a rolling window, so each run sees a slightly different
universe, and near a k decision boundary that can flip the selection. The cure is
process: snapshot for modeling, pin one champion run ID, cite it in every artifact,
and don't retrain the week of a review.

### "How fresh are the segments? Can a customer move between them?"
Yes, automatically. Rescoring runs on a schedule against the frozen champion, so a
regular who slows down shows up in the fading segment without anyone touching a list.
Retraining (new centroids) is a separate, reviewed event — with name-matching so
"Segment X" can't silently change meaning between refreshes.

### "How is campaign impact measured?"
A control group that receives no campaigns, managed in the campaign tool and
**globally excluded from every send** (not a per-campaign control variant — those
contaminate each other). Lift = how targeted customers changed vs how control
customers changed over the same window (difference-in-differences), which cancels
seasonality. Two requirements to state early: control membership must flow back to the
data platform or the lift math can't see it, and small segments carry small control
counts — their reads take longer *by design*, so say "patience" in the brief, not in
week ten.
