# Smart Birmingham Traffic Intelligence
### Explainable Traffic Flow Prediction using Genetic Programming

## Project Overview

This project explores how **Computational Intelligence** can support **smart transportation** by modelling and predicting traffic flow into and out of Birmingham.

The core focus is **Genetic Programming (GP)**, applied via **Symbolic Regression**, to evolve a mathematical expression that predicts traffic flow directly from road traffic data. Unlike black-box ensemble models, symbolic regression produces a human-readable formula — in principle making the model's reasoning inspectable rather than opaque.

This is a personal continuation of a university dissertation project, rebuilt with a more rigorous evaluation methodology than the original coursework version (see **Key Findings** below for what changed and why).

Birmingham is the case study location because the Department for Transport publishes long-running, public road traffic count data for the city. The extract used in this project covers **2000–2025** across **562 count points** (72,948 raw hourly observations), including vehicle class, directional flow, and raw hourly counts.

---

## Project Aim

Develop an explainable computational intelligence model that predicts traffic flow into and out of Birmingham using road traffic data and Genetic Programming, and rigorously benchmark it against conventional machine learning baselines.

---

## Project Objectives

1. Collect and prepare Birmingham road traffic count data.
2. Explore traffic patterns by road, year, hour, and direction of travel.
3. Build a Genetic Programming symbolic regression model to predict traffic flow.
4. Benchmark GP against Linear Regression, Random Forest, and Gradient Boosting.
5. Evaluate every model using MAE, RMSE, and R², under an evaluation scheme that doesn't let information leak between count points.
6. Extract and interpret the best evolved symbolic expression, including its complexity.
7. Identify which features actually drive predictions, correcting for known biases in feature importance methods.

---

## Background

Intelligent Transportation Systems treat traffic modelling and prediction as central to smart city infrastructure. Transport for West Midlands' Data Insight programme similarly emphasises data-driven forecasting and nowcasting for transport planning across the region.

This project builds a simplified but methodologically honest prototype of that kind of system, using Birmingham as the case study.

---

## Why Birmingham?

Birmingham is a major UK city with substantial, publicly documented traffic movement. DfT reports **3,738 million vehicle miles** travelled on Birmingham's roads in 2025 alone. The available datasets — count points, vehicle class breakdowns, annual average daily flow, directional flow, and raw counts — make it well suited to a directional traffic-flow prediction task.

---

## Computational Intelligence Approach

Genetic Programming is an evolutionary algorithm that performs symbolic regression:

1. Generate an initial population of random mathematical expressions.
2. Evaluate each expression's prediction error on the training data.
3. Select better-performing expressions.
4. Produce new expressions via crossover and mutation.
5. Repeat over many generations.
6. Return the best-evolved expression.

The appeal is explainability: instead of a black-box prediction, GP can in principle return something like *"traffic flow ≈ f(road type, hour, location)"* as an actual formula.

---

## Methodology / Pipeline

```
Raw DfT CSV (72,948 rows)
        ↓
Cleaning & type fixes (mixed-type columns, count_point_id → categorical)
        ↓
Feature engineering (is_morning_peak, is_evening_peak, is_peak_hour, flow_direction)
        ↓
Grouped train/test split (grouped by count_point_id — see Key Findings)
        ↓
Encoding:
   - Target encoding  → count_point_id, road_name   (high-cardinality)
   - One-hot encoding → road_type, direction_of_travel, flow_direction (low-cardinality)
        ↓
Baseline models: Linear Regression · Random Forest · Gradient Boosting
        ↓
Hyperparameter tuning (RandomizedSearchCV, grouped CV)
        ↓
Genetic Programming symbolic regression (gplearn)
        ↓
Evaluation: MAE · RMSE · R² (single split AND GroupKFold CV)
        ↓
Feature importance: impurity-based AND permutation-based
```

---

## Results

**GroupKFold cross-validated performance** (5 folds, grouped by `count_point_id` — this is the realistic "predicting at a new/unseen location" number):

| Model | RMSE | R² |
|---|---|---|
| Gradient Boosting | 199.7 | 0.928 |
| Linear Regression | 203.6 | 0.921 |
| Random Forest | 213.0 | 0.916 |

**Final holdout comparison** (tuned models + GP, single train/test split):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (tuned) | 57.6 | 127.7 | 0.974 |
| Gradient Boosting (tuned) | 87.0 | 175.8 | 0.950 |
| Linear Regression | 107.0 | 207.4 | 0.931 |
| Genetic Programming (parsimony=0.05) | — | 212.2 | 0.927 |
| Genetic Programming (unconstrained) | — | 214.4 | 0.926 |

GP's evolved expression length is highly run-dependent (see Key Finding 5) — this run's unconstrained expression reached **length 233**; see `outputs/gp_expression.txt` for the full symbolic expression at every parsimony setting tested.

**Ablation — Random Forest without `count_point_id`:**

| Model variant | MAE | RMSE | R² |
|---|---|---|---|
| Full model (with location) | 57.6 | 127.7 | 0.974 |
| Location excluded | 68.7 | 136.9 | 0.970 |

See **Key Findings (4)** below for interpretation.

---

## Key Findings

These are the methodologically important findings from this project — including places where an initial naive approach gave misleading results, and what fixed them.

**1. A single train/test split overstates performance.**
An ungrouped split gave Random Forest an RMSE of ~117 and R² of 0.978. Grouping by `count_point_id` (so a location's data can't appear in both train and validation) revealed the realistic RMSE is closer to **200–213** — the original split let the model partly memorise each location's typical traffic rather than generalise.

**2. One-hot encoding a high-cardinality location ID breaks Linear Regression on unseen locations.**
Early versions one-hot encoded `count_point_id` (~600 dummy columns). Under grouped CV, every dummy for an unseen location is zero, causing Linear Regression to extrapolate wildly (RMSE ~3,600, R² ≈ −25.6). Switching to **target encoding** for `count_point_id` and `road_name` fixed this and cut the feature space from 623 columns down to 16.

**3. Genetic Programming is competitive but not obviously more interpretable.**
GP comes close to Linear Regression's accuracy (R² 0.926–0.927 vs. 0.931) — a genuinely good result for an evolved expression. But its winning unconstrained expression had a **program length of 233**, i.e. a deeply nested formula, not the simple, human-readable equation symbolic regression is often pitched as producing. This is a fair, reportable limitation rather than a failure.

**4. Feature importance needs a bias check — and an ablation confirms something more interesting than it first looked.**
Random Forest's built-in (impurity-based) feature importance is known to be biased toward high-cardinality features. Once `count_point_id` was target-encoded into a single column, it dominated both impurity-based (40.9%) and, even more so, permutation-based importance (65.1%), appearing to confirm that exact monitoring location is by far the single biggest driver of traffic volume.

A follow-up ablation — refitting Random Forest with `count_point_id` removed entirely — tells a more precise story. Accuracy barely drops at all, from **R² 0.974 (full model) to R² 0.970 (location excluded)**, a gap of just 0.004. In other words, almost all of what `count_point_id` appeared to be contributing was recoverable from `road_name` and `latitude`/`longitude` alone — once location is removed, **`road_name` becomes the dominant feature (38.5% impurity importance, 58.6% permutation importance)**, followed by geographic coordinates (longitude 15.4–15.6%, latitude 9.3–14.1%).

This refines the original conclusion: it isn't granular per-count-point detail that drives traffic flow, but road identity (e.g. the M6 vs a residential side road) and broad geographic position. `count_point_id` was mostly acting as a fine-grained proxy for information already present in `road_name` and coordinates, not contributing large amounts of genuinely new signal — removing it costs almost nothing.

**5. A parsimony-constrained Genetic Programming run found a genuinely interpretable formula, with essentially no accuracy cost.**
The unconstrained GP run (as reported above) produced a 233-node expression achieving R² 0.926 — accurate, but no more readable than a black box. Sweeping `parsimony_coefficient` (which penalises program length during evolution) across `[0.0, 0.001, 0.005, 0.01, 0.05]` found a dramatically shorter alternative at `parsimony_coefficient=0.05`: a **5-node expression achieving R² 0.927** — matching (in this run, fractionally exceeding) the unconstrained result, for a **98% reduction in complexity** (233 nodes → 5).

That expression, translated into feature names, is:

```
predicted_traffic ≈ location_average + 2 × longitude
```

Birmingham-area longitudes are small negative numbers (around −1.9 to −2.1), so the second term is a near-constant adjustment of only a few units either side of zero — negligible next to typical traffic volumes in the hundreds to thousands. In plain terms, GP rediscovered a simple, explainable rule: *predict each location's historical average traffic, with a small positional nudge* — a genuinely human-readable model, not a black box. This is the strongest evidence in the project for the "explainable computational intelligence" aim stated at the outset.

One caveat: the specific formula above is itself a single run's result — GP's search is stochastic, so a different run of the same `parsimony_coefficient=0.05` setting can land on a different (if similarly short) expression. A follow-up multi-seed robustness check quantifies exactly how much this varies:

**Multi-seed robustness check (5 random seeds each):**

| Setting | Length (mean ± std, range) | R² (mean ± std) |
|---|---|---|
| Unconstrained (pc=0.0) | 106.2 ± 72.4 (range: 59–233) | 0.9275 ± 0.0017 |
| Parsimony-constrained (pc=0.05) | 3.8 ± 1.8 (range: 1–5) | 0.9266 ± 0.0018 |

Accuracy is stable across seeds either way (R² varies by <0.002). But **complexity is wildly seed-dependent without parsimony pressure** — the unconstrained run swings from 59 to 233 nodes purely based on random seed, meaning any single unconstrained expression (this run's 233-node one included) isn't a meaningful, reproducible result on its own — it's one point in a wide range. With `parsimony_coefficient=0.05`, complexity reliably stays tiny (1–5 nodes) across every seed tested, at a cost of under 0.001 mean R².

**This is the project's strongest methodological finding**: symbolic regression's much-advertised interpretability is not automatic. Left unconstrained, GP's output complexity is essentially arbitrary and unreproducible even though its accuracy is stable. Interpretability had to be actively enforced via parsimony pressure — and once enforced, it was achieved reliably, at a small and consistent accuracy cost.

---

## Extension: Junction-Level Traffic Prediction

The original project aim was framed around traffic "into and out of Birmingham" generally. This extension takes that idea further: predicting traffic on the links connected to **specific named junctions**, using the trained model rather than historical averages — closer to the eventual goal of a junction-level traffic simulation tool.

### Scope

Only **31.9% of rows** (23,268 of 72,948) have both `start_junction_road_name` and `end_junction_road_name` populated — DfT only records junction endpoints for classified roads (A-roads and motorways), not minor streets. Within that subset:

- **170 distinct links** (count points), forming exactly **170 (road, start-junction, end-junction) combinations** once road names are canonicalized — this is the identity `junction_network.py` uses. 10 of those 170 count points are individually ambiguous: at the raw `(start, end, count_point_id)` level `junction_predictor.py`/`junction_cli.py` use instead, they carry **two** distinct junction-endpoint pairs each, so that tool's graph has 180 edges rather than 170. Both counts are correct for their own tool — see "Honest limitations" below.
- Covers **22 major named roads** (23 raw `road_name` spellings, merged to 22 after canonicalizing `A38M`/`A38(M)`).

This means the junction tool covers Birmingham's major road network specifically, not every street — a real and honest scope limitation, not a shortcoming to hide.

### Two data-quality fixes found while building the graph

1. **Inconsistent road naming fragmented a single road into two identities** — e.g. `A38M` and `A38(M)` (the Aston Expressway) appeared as separate `road_name` values, which would have split one continuous road into two disconnected graph identities. Fixed by canonicalizing road names (stripping punctuation/case before comparison) and merging variants.
2. **A pervasive, structural ambiguity in junction labels — not just the one-off "LA Boundary" case.** Birmingham's A4040 (the Middle Ring Road) is crossed by nearly every major radial route (A38, A34, A41, A452, etc.) at *different* points around the ring, but DfT's junction label just names the crossing road ("meets the A4040") without saying where. A systematic check (`check_generic_junctions.py`, which measures the geographic spread of count points sharing each junction label) found this pattern affected **24 different labels**, not one — including A38, A34, A41, A4040, A452, A453, A441, A47, A4540, and more, with spreads up to 24km across as many as 11 different roads. The fix generalizes the original "LA Boundary" patch: **any junction label matching a known road name in the dataset is treated as generic and qualified by the road using it as an endpoint** (e.g. `A4040 (via A34)`, `A4040 (via A38)`), rather than hardcoding one label. After the fix, the graph grew from 126 to **193 junction nodes** (170 links), with the top-traffic ranking now dominated by genuinely specific locations (`Park Circus B4132 Waterlinks Boulevard`, `B4217`, `A4040 Wheelwright Road`, `A456/A457 roundabout`) instead of ambiguous bare road names absorbing traffic from many unrelated physical points. (A separate, one-off data-entry typo in the raw junction labels — `"A456/A457 rooundabout"` vs `"A456/A457 roundabout"` — was also found and corrected, merging what would otherwise have been two nodes for the same physical roundabout.)

**A third fix, found while verifying the above**: the graph was originally built as a plain `networkx.Graph`, which only allows one edge between any two nodes. Since distinct roads can share the same two junction endpoints (e.g. both A38 and A4400 link `A456/A457 roundabout` to `B4100`), a plain `Graph` silently overwrote one link's data whenever this happened — one specific case, discovered by cross-checking `170` link rows against `169` graph edges after a rebuild. Switched to `networkx.MultiGraph`, which represents both links as parallel edges instead of dropping one; the graph now has exactly 170 edges, matching the 170 link rows.

**Known residual limitation**: qualifying a label by the road referencing it only resolves ambiguity when *multiple different roads* share that label — it does not resolve the same road reusing a bare number at more than one distinct point. The M6's bare junction numbers ("5", "6") were relabeled for consistency (`5 (via M6)`, `6 (via M6)`), but the underlying case found by the systematic check — "6" spanning 7.17km even with only the M6 referencing it — remains genuinely unresolved. This is a small, low-priority edge case affecting one motorway's numbering, documented here rather than pursued further given diminishing returns. Notably, after all fixes, the network's top three nodes by total connected traffic are all M6 junctions — a sensible result, since the M6 is Birmingham's dominant motorway corridor.

### The scripts

- **`check_generic_junctions.py`** — the systematic diagnostic described above: computes the geographic spread of count points sharing each junction label and flags any label that's likely generic/ambiguous, rather than relying on spotting issues manually one at a time.
- **`junction_network.py`** — builds the graph (junctions as nodes, links as edges) and computes a baseline network using **historical average traffic** per link. Produces `outputs/junction_network.png`, `junction_link_summary.csv`, and `junction_totals.csv`.
- **`junction_predictor.py`** — extends this by predicting traffic with the trained Random Forest model for **any chosen year and hour**, instead of a historical average, then aggregates predictions across every link touching a chosen junction. Exposes a reusable `predict_junction_traffic(junction_name, year, hour, ...)` function.
- **`junction_cli.py`** — an interactive command-line front-end for the above: trains the model once, then lets you repeatedly type a junction name, year, and hour and see the predicted breakdown immediately, without editing any code. This is the recommended way to actually explore the tool.

### Honest limitations

- **Not a geographic map.** Junction node positions use a force-directed layout, not real coordinates — the data identifies junctions by name, not by their own lat/long.
- **No turning-movement split.** The tool predicts total volume on each link touching a junction; it cannot say what proportion of that traffic turns left/right/continues straight, since DfT's raw counts don't include turning-movement data. A true simulation with that level of detail would need additional geometry and signal-timing data this dataset doesn't provide.
- **`flow_direction` (inbound/outbound/lateral) is relative to Birmingham's centre, not the specific junction being queried** — an approximate directional signal, not a precise "traffic arriving from the north at this exact junction" figure.
- **The two tools' totals are not directly comparable.** `junction_network.py`'s baseline averages both directions of travel *together* into one number per link; `junction_predictor.py` predicts each direction separately and sums them, and represents one specific hour rather than an all-hours average. Report these as distinct metrics, not the same thing measured twice.
- **The two tools also define "a link" slightly differently**, so they report different edge counts for the same underlying data (170 vs. 180 — see Scope above): `junction_network.py` identifies a link by `(road, start-junction, end-junction)`; `junction_predictor.py`/`junction_cli.py` identify it by `(start-junction, end-junction, count_point_id)`. The 10-edge gap is exactly the 10 count points that carry two distinct junction-endpoint pairs in the raw data.

### Example result — a genuine model validation, not just a number

At the A41 junction, one link (count point 7927, the A4540) showed westbound/inbound traffic exceeding eastbound/outbound at 8am (675.9 vs 594.8), which **flipped by 5pm** (outbound 678.8 vs inbound 641.4) — the classic AM-commute-in / PM-commute-out pattern, emerging from the model itself rather than being hand-coded. This isn't universal across every link (e.g. the A4400 near A38 stayed outbound-dominant at both times, which is also plausible — not every road carries a symmetric commuter flow), but it's a meaningful qualitative sanity check that the model captures real directional time-of-day structure.

---

## Repository Structure

```
FYP v.2/
├── outputs/                                    generated charts, CSVs, gp_expression.txt
└── Traffic Intelligence/
    ├── dft_rawcount_local_authority_id_141.csv
    ├── docs/                                    architecture, model design/results, feature docs
    ├── Findings/                                EDA_Report1.md — the project's first-pass EDA,
    │                                             preserved as the "before" half of the project's
    │                                             before/after methodology narrative
    └── Code/
        ├── traffic_common.py                    shared paths, feature engineering, encoding
        │                                         pipeline, and junction-graph cleaning — imported
        │                                         by every script below
        ├── traffic_flow_prediction_v2.py         main pipeline (see note below on the filename)
        ├── junction_network.py                   historical junction network (baseline)
        ├── junction_predictor.py                 model-based junction traffic predictor
        ├── junction_cli.py                       interactive CLI front-end for the above
        ├── check_generic_junctions.py            systematic check for generic/ambiguous junction labels
        ├── check_junctions.py                    one-off diagnostic (junction field coverage)
        └── check_m6_node.py                      one-off diagnostic (verifying the M6 junction case)
```

All `DATA_PATH`/`OUTPUT_DIR` values are now resolved relative to this repository layout (via `traffic_common.py`), not hardcoded to one machine — every script can be run from any working directory.

`docs/` has more detail than fits here: `Model_Design.md` and `Model_Results.md` for the modelling approach and full results tables, `Feature_Engineering.md`/`Feature_Rationale.md`/`Feature_Analysis.md` for what each feature is and why, and `Architecture.md` for how the scripts fit together.

**Optional rename, purely cosmetic:** `traffic_flow_prediction_v2.py` could be renamed to drop the `_v2` suffix, since — despite the name — it's the only, final version of the pipeline (the file has accumulated GroupKFold evaluation, target encoding, the feature-importance ablation, GP parsimony sweep, multi-seed robustness, and the `flow_direction` feature over the course of the project). Not done here since every script and this README reference the current filename; rename and update those references together if you want to do it.

---

## Setup

```bash
pip3 install -r requirements.txt
```

The raw CSV should already be at `Traffic Intelligence/dft_rawcount_local_authority_id_141.csv` (downloaded from the [DfT road traffic statistics site](https://roadtraffic.dft.gov.uk/local-authorities/141)). If it moves, update `DATA_PATH` in `Traffic Intelligence/Code/traffic_common.py` — every script resolves it from there, so there's a single place to change.

Paths are resolved relative to the repository layout, not the current working directory, so these can be run from anywhere:

```bash
python3 "Traffic Intelligence/Code/traffic_flow_prediction_v2.py"   # main pipeline
python3 "Traffic Intelligence/Code/junction_network.py"             # junction baseline network
python3 "Traffic Intelligence/Code/junction_predictor.py"           # junction traffic predictor
python3 "Traffic Intelligence/Code/junction_cli.py"                 # interactive junction query tool
```

---

## Future Work

- ~~Ablation: permutation importance with `count_point_id` excluded~~ — done; see Key Findings (4).
- ~~Try a parsimony-constrained GP for a shorter, more interpretable expression~~ — done; see Key Findings (5).
- ~~Extend to directional in/out-of-Birmingham modelling~~ — done via `flow_direction`; see `traffic_common.add_engineered_features()`, shared by `traffic_flow_prediction_v2.py` and the junction scripts.
- ~~Junction-level traffic prediction~~ — done; see the Extension section above.
- ~~A true turning-movement breakdown at junctions~~ — not currently feasible; DfT's raw counts don't include the turning-movement or signal-timing data this would require. Documented as a permanent scope limitation rather than a to-do.
- ~~Package the junction predictor as a small interactive tool~~ — done via `junction_cli.py`.
- Re-run the GP parsimony sweep and robustness check with more random seeds per setting, for a smoother, more statistically robust tradeoff curve.
- Resolve the remaining M6 junction-number edge case (the same road, "6", reusing a bare number at two points 7.17km apart) — low priority given its small traffic footprint relative to the fix already applied.
- Integrate `check_generic_junctions.py` as a validation step that runs automatically before `junction_network.py` builds the graph, rather than as a separate manual script — would catch this class of issue automatically on any future data update.