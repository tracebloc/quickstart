# Family training templates

One template per **family**, not per `(category, framework)` pair. There are
six families here and twenty legal pairs; the pairs are covered by conditional
cells inside a family's template rather than by a file each. Twenty
hand-maintained notebooks rot within a quarter — six families do not.

`notebook.render` still resolves by `(category, framework)`. It is the *files*
that are per family, not the lookup.

These are **not** the user-facing guide. They carry no Colab lines and no
install cell: the pod they render in already has the SDK and the model zoo. They
also carry no login prompt — though that one is a *target*, not yet true; see
"Two dependencies" below, because environment login is merged and unreleased.
For the guide, see
[`../traceblocTrainingGuide.ipynb`](../traceblocTrainingGuide.ipynb).

## The families and their pre-fills

| family | categories | `cycles` | `epochs` |
|---|---|---|---|
| [Vision, from scratch](vision_from_scratch.ipynb) | `image_classification`, `object_detection`, `semantic_segmentation`, `keypoint_detection` | 20 | 1 |
| [NLP fine-tune (incl. LoRA)](nlp_finetune.ipynb) | `text_classification`, `sentence_pair_classification`, `token_classification` | 8 | 1 |
| [NLP generative](nlp_generative.ipynb) | `causal_language_modeling`, `seq2seq`, `masked_language_modeling` | 5 | 1 |
| [Embeddings](embeddings.ipynb) | `embeddings` | 8 | 1 |
| [Tabular / time series](tabular_timeseries.ipynb) | `tabular_classification`, `tabular_regression`, `time_series_classification`, `time_series_forecasting` | 15 | 1 |
| [Survival](survival.ipynb) | `time_to_event_prediction` | 1 | 1 |

> **What actually enforces this.** `scripts/check_templates.py` is run by
> `.github/workflows/template-rules.yml` on every PR. Before that workflow
> existed nothing ran it, so every "enforced" below meant "enforced if the
> author remembers" — the defect this whole file is otherwise about
> (start-training#89, 5/10). It is **not yet a required status check**;
> until an admin adds it to develop's contexts, a red here is visible but
> not blocking.

The numbers live in [`families.json`](families.json), and
[`../../scripts/check_templates.py`](../../scripts/check_templates.py) asserts
that each notebook's settings cell really sets them — so this table, that file
and the notebooks cannot drift apart.

**`cycles` is what federates; `epochs` is what drifts.** The SDK's own
`epochs=10, cycles=1` is one round with ten local epochs, which is the textbook
client-drift setup and the worst corner of the trade-off; it is deliberately not
any template's pre-fill.

Two rules bind every row, both enforced by the checker:

1. **`cycles × epochs ≤ 20` for any pre-fill** — so a peer's first Start can
   never be the reason the team's budget is spent.
2. **No template pre-fills `epochs > 1` unless its aggregation strategy carries
   a drift correction** (`fedprox`, `fedadam`, `fedyogi`, `fedadagrad`). Plain
   FedAvg with many local epochs *is* the drift setup, so the cell comment and
   the number sitting next to it have to agree.

The checker also holds the rule that a family template must never offer a
setter to a category that refuses it. It carries the setter-to-category map
read off the SDK's own gates, and audits every cell against the categories that
cell is offered to — the whole family for a shared cell, only the gated subset
for a fragment. Three real instances of that mistake were found by hand in the
Tabular / time series template before the check existed (`feature_points`,
which only `tabular_classification` takes; `encoding_strategy` and
`normalize_features`, which the forecasting path does not read; and a `scaler`
default that was right for tabular and wrong for time series).

**A `cycles` pre-fill cannot reach a single-pass framework.** The SDK forces
`cycles` and `epochs` to 1 for `sklearn`, `lifelines` and `scikit_survival` —
by *framework*, for every category — and it only warns, so an ungated
`cycles(15)` on a sklearn tabular model prints "cycles cannot be updated" and
quietly trains one round. D9's table gives one `cycles` per family keyed on
category, which cannot be honoured for those pairs. Tabular / time series
(`sklearn`) and Survival (`lifelines`, `scikit_survival`) therefore gate
`cycles`/`epochs` on framework: pytorch pairs get the pre-fill, single-pass
pairs get a comment saying both are forced to 1. `families.json` records each
family's `frameworks` and the `single_pass_frameworks` set, and the checker
refuses an ungated pre-fill on a family with such a pair.

The `single_pass_frameworks` list is a **mirror** of the SDK's
`tracebloc.training.plan._SURVIVAL_FRAMEWORKS`, and a mirror goes stale
silently — the same objection this repo's CLAUDE.md raises about restating the
SDK's Python bound. The checker therefore *derives* the set when the SDK is
importable and fails on drift. The SDK is not a dependency here (it pulls
torch), so CI cannot import it — and in that case the checker **prints** that
the cross-check was skipped rather than passing quietly, because an
unverifiable mirror that says nothing is exactly the can't-fail shape this
checker exists to catch.

No family currently pre-fills `epochs > 1`, so none needs a drift correction.
The vision row buys its twenty effective passes with rounds (`cycles = 20`)
rather than local epochs, which is rule 2's whole point: from-scratch vision is
exactly where drift bites hardest.

### What the checker renders

Rule 12 renders every `(category, framework)` pair from the family's **own**
`categories` and `frameworks` lists. It used to hardcode `pytorch` and
`sklearn`, which both skipped real pairs — Survival's `lifelines` and
`scikit_survival` were never rendered, so a fragment gated only on those could
carry a syntax error while the check reported all rules holding — and rendered
illegal ones like vision + `sklearn`. Found by Cursor Bugbot.

`frameworks` is per family, so this still over-approximates: it renders
`(time_series_classification, sklearn)` though the zoo ships no sklearn model
there. That is the safe direction for a validator — it can only demand more
validity, never less — and the precise legal-pair matrix belongs to the engine
registry rather than being restated here.

## Verification status

D9: *a template ships only once a live experiment on dev has reached COMPLETED
using its own defaults* — a pre-fill nobody has run is a guess with a Start
button attached.

The records are in [`verification-dev.json`](verification-dev.json), committed
rather than left on the dev cluster: dev gets swept, and a verification that
depends on state we do not control is a claim, not a proof. The checker's
rule 13 binds that file to `families.json`, so **editing a pre-fill here fails
the check** until the dev run is redone and the evidence refreshed. A family
that cannot be verified must instead carry `unverified` in `families.json`
naming the ticket that explains why; a family that is neither is refused.

Verified on the complete settings cell:

| family | `cycles` | experiment | minutes |
|---|---|---|---|
| [Vision, from scratch](vision_from_scratch.ipynb) | 20 | `eaxx647p` | 40.3 |
| [NLP fine-tune](nlp_finetune.ipynb) | 8 | `enpm5bls` | 16.9 |
| [NLP generative](nlp_generative.ipynb) | 5 | `e6lymzry` | 11.4 |
| [Embeddings](embeddings.ipynb) | 8 | `erl4u2pk` | 18.0 |
| [Tabular / time series](tabular_timeseries.ipynb) | 15 | `eeofzvuj` | 28.9 |

**[Survival](survival.ipynb) is UNVERIFIED** — tracked internally. Both dev
`time_to_event_prediction` datasets carry a `label` of time-like values instead
of the 0/1 event indicator, so the engine's TTE validator refuses every
experiment on them, and there is no other TTE dataset on dev. Its pre-fills are
not in question; the evidence is missing, and by D9's own rule that means the
template has not shipped.

An earlier round applied only `cycles`/`epochs`/`training_classes`. Every other
field *coincided* with the template except `callbacks`, which came back `'[]'`
because `terminate_on_nan_callback()` was never called — a coincidence reads
exactly like a verification until someone reads the record. Both rounds are
kept in the evidence file, scoped for what each actually shows.

Those durations are also, as far as this epic has measured, the only dev-edge
timing baseline for federated runs by family. The second round was *faster*
than the first on every family despite six runs sharing the edge, so they are a
usable baseline rather than a contended outlier. `estimatedflops` is **not** a
progress meter, incidentally: completed runs land at 80-95% of it.

## The render contract

The backend stores no template and renders nothing. It passes the context as
environment; the SDK **in the pod** renders with
`notebook.render(category, framework, context)` from the templates in the image.
One manifest pins SDK, zoo and templates together, so a template cannot be
rendered by an SDK that does not match it.

Rendering does two things, and only these two:

**1. Substitute `{{ key }}` placeholders** from the context. Keys used here:

| key | what it is |
|---|---|
| `use_case` | use-case name, for the facts table |
| `dataset_id` | dataset key to link against |
| `category`, `framework` | the pair being rendered |
| `edge_count`, `records_per_edge` | facts-table figures |
| `experiment_name` | pre-filled *‹model› on ‹use case› #‹n›* |
| `model_path` | path the picker chose |
| `validation_split` | dataset-derived, not a family constant — the SDK computes it from the smallest edge's record count and the class count |
| `training_classes` | per-class subsample map |
| `data_type` | `rgb` or `grayscale`; a 1-channel dataset fails the channel check at the rgb default |
| `feature_points` | column count; must agree with the dataset or the link is refused |
| `sequence_length`, `forecast_horizon` | sequence shape |
| `scaler` | category-derived, not a family constant — `MinMaxScaler` for time series, `StandardScaler` for tabular and time-to-event |
| `tokenizer_path` | contributor `tokenizer.json`, when one is not resolvable by name |

**2. Drop cells that do not apply.** A cell carrying
`metadata.tracebloc.applies_to` is kept only when the pair being rendered
matches it:

- `{"category": [...]}` — keep only for these categories. Used for the Tabular
  / time series settings fragments, whose settings genuinely differ by
  category: `feature_points` (only `tabular_classification` of that family
  takes it), `sequence_length` (time series only), `forecast_horizon`
  (forecasting only), `missingness_indicators` (time-series classification
  only), and the `encoding_strategy` / `normalize_features` pair (everything
  but forecasting).

  > An earlier draft cited the custom-loss cell here as an
  > `object_detection`-only gate. That is no longer true and was the defect
  > described under *Verification* below: the cell is ungated in the settings
  > cell of every family that accepts a custom loss, because all of them do
  > except embeddings.
- `{"framework": [...]}` — keep only for these frameworks. Used for the vision
  augmentation group: those ten setters are gated on the *framework*, not the
  category, and they **refuse** rather than warn, so calling one on a
  non-pytorch model poisons the plan and Start then blocks. `shuffle`, the
  eleventh, is not framework-gated and stays in the shared cell.
- `{"dataset_flag": "allow_feature_modification"}` — keep only when the
  dataset sets that flag. Used for the feature-interaction cell, which is also
  a settings fragment: its calls have to land *in* the settings cell to be
  applied at all (see the `start()` note under *What is deliberately absent*).

Cells without that metadata are always kept.

The checker **executes** this contract rather than describing it: for every
family, for each category it owns, it renders the template and compiles the
resulting settings cell. That is what catches a placeholder sitting in a
position where no real value parses — a `{{ key }}` inside a string literal and
one standing as a bare argument are not interchangeable, and nothing else would
notice.

**Settings fragments.** A family can span categories that do not take the same
settings — Tabular / time series is the case that forces this: `sequence_length`
is meaningless for `tabular_classification`, `forecast_horizon` applies only to
forecasting, and `missingness_indicators` only to time-series classification.
Those live in cells marked `metadata.tracebloc.settings_fragment: true`
alongside their `applies_to` gate. A fragment that survives the gate is
**concatenated into the single settings cell**, in document order, after the
main block; one that does not is dropped.

That keeps both halves of the design true at once: the peer still sees *one*
settings cell with the complete applicable settings and no inapplicable group,
and one file still covers four categories. A fragment with no `applies_to`
would apply to the whole family and belongs in the main block instead — the
checker rejects it, because left as a fragment it quietly becomes a second
settings cell.

## Two dependencies that are not satisfied yet

**Environment login is merged but NOT RELEASED.** The connect cell's premise —
the pod is already authenticated and there is no password prompt — needs an SDK
*release* carrying environment login. It is on the SDK's `develop`
(pyproject 1.0.9) and **absent from the latest tag v1.0.7**, which is what
`pip install tracebloc` resolves; v1.0.7 has no `env_login` module and no
`TRACEBLOC_TOKEN` path at all (verified 2026-09-09). Until a release ships it,
`User()` prompts interactively, which is wrong for a pod. The cell says so in
the cell. Keying this on the change *merging* would have looked satisfied the
moment it did.

**`notebook.render` does not exist yet.** These files are templates and the
render contract above is a specification for the SDK side, not a description of
shipped behaviour.

## What is deliberately absent

- **`start()`.** No cell calls it. Start is a button, and there is no Run All:
  it re-links, executes the settings cell, then starts — in that order, because
  `start()` is one-shot and resets the plan.

  **A corollary that cost this PR a round of review:** because Start executes
  *only* the settings cell, a `training.*` call in any other cell is **inert**,
  however inviting it looks. An earlier draft put the custom-loss and
  feature-interaction calls in standalone cells sitting *before* the link, so
  uncommenting one applied nothing — and for a YOLO model, which cannot start
  without a custom loss, that made the run unstartable. Every setter now lives
  in the settings cell or in a gated fragment concatenated into it, and the
  checker refuses any that does not (found by Cursor Bugbot).
- **The five dead setters** — `horizontal_flip`, `vertical_flip`,
  `samplewise_center`, `samplewise_std_normalization`, `layers_freeze`. They are
  no-ops on every surviving framework and `start()` refuses some of them. They
  are still public API, so their removal needs its own major bump; a template
  pre-filling them would be pre-filling a failure. The checker rejects them.
- **A custom-loss cell in the Embeddings template.** The contrastive objective
  is intrinsic (InfoNCE over in-batch negatives) and the SDK rejects a supplied
  loss for that category, so offering the cell would offer a rejection. It is
  the *only* such family: the SDK's NLP base hook documents itself as "Base:
  every family supports one — no-op" and embeddings alone overrides it to
  raise, so every other template carries the cell. An earlier draft gated it to
  `object_detection` only, which hid a supported feature from eleven
  categories; `families.json` now records the exception list and the checker
  enforces both directions — by *coverage*, not presence: the offering must
  reach every category in the family. A first version of that rule asked only
  whether `training.loss_function` appeared *somewhere* in the file, and so
  passed the very defect it was written to stop.
- **Augmentation outside the vision template.** The eleven live augmentation
  setters apply to pytorch image models only; the sklearn branches are refused
  outright.
- **`data_shape` in the vision template.** Image size is fixed per model for
  pytorch and is not user-settable — `start()` refuses the call.

## Ownership

Per *family*, not per pair — six owners, in Data Science. Defaults recorded
2026-09-08. The conditional-cell mechanism above is what makes the smaller
number work.

> **Reaching users.** These templates render inside the pod, so they reach a
> peer through the notebook image, not through the Colab link. They are
> unrelated to the Drive-hosted copy of the *guide* that the web app's "Start
> training" button still points at — nothing in this
> repo can change that copy, and nothing here needs to.
