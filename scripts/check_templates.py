#!/usr/bin/env python3
"""Enforce the D9 template rules on the family templates.

Run from the repo root:

    python3 scripts/check_templates.py

Exits non-zero and prints every violation. The rules this encodes are the ones
prose cannot hold: they are about numbers inside notebook cells, and a reviewer
reading a diff of `.ipynb` JSON will not catch a `cycles` that stopped matching
the table.

Checked:

1. `cycles * epochs <= 20` for every family pre-fill.
2. No family pre-fills `epochs > 1` unless its aggregation strategy carries a
   drift correction.
3. Each notebook's settings cell really calls `cycles(...)` / `epochs(...)` /
   `aggregation_strategy(...)` with the values `families.json` declares, so the
   table and the notebooks cannot drift apart.
4. Every category appears in exactly one family, and every family names a
   notebook that exists and is valid JSON.
5. No template calls `start()` — Start is a button, and a template that starts
   a run has stopped being a template.
6. No template mentions one of the five dead setters, which are no-ops on every
   surviving framework.
7. No template carries a Colab-specific line; these render in the pod.
8. Every `metadata.tracebloc` cell uses keys the render contract defines, and
   no cell is gated on a category OR framework outside its own family — a typo must not
   silently make a conditional cell unconditional, or a gated cell unrenderable.
9. No `settings_fragment` is ungated: one that applies to the whole family
   belongs in the settings cell, not beside it.
10. Every `{{ key }}` placeholder names a context key the render contract
    defines, and no template carries a half-written `{ key }` -- which is what
    an f-string silently turns `{{ key }}` into.
17. No cell carries an internal reference — a private tracker id, an RFC id,
    an internal hostname. The templates render into a peer's notebook, so
    anything in a cell is shown outside the org -- and this repository is
    public, so the same holds for its prose.
16. No `training.*` setter call sits outside the settings cell or a settings
    fragment, and the settings cell comes after the cell that assigns
    `training`. Start re-links and then executes ONLY the settings cell, so a
    setter in a standalone cell is inert however inviting it looks — and one
    placed before the link would `NameError` if a peer uncommented it.
15. The custom-loss offering REACHES EVERY category that accepts one — not
    merely appears somewhere in the file — and reaches none in the families
    listed in `no_custom_loss_families`, whose objective is intrinsic. Checking
    presence rather than coverage let the exact defect this rule was written to
    stop pass it.
14b. `single_pass_frameworks` in `families.json` mirrors the SDK's
    `_SURVIVAL_FRAMEWORKS`; where the SDK is importable the two are compared
    and drift fails. Where it is not, the skip is PRINTED rather than silent —
    an unverifiable mirror that says nothing is the same can't-fail shape this
    checker exists to catch.
14. `cycles` / `epochs` are never offered to a single-pass framework. The SDK
    forces both to 1 for sklearn, lifelines and scikit_survival -- by FRAMEWORK,
    for every category -- and only WARNS, so an ungated pre-fill on a family
    with such a pair is both silently unhonoured and noisy on exactly the pairs
    it cannot serve.
13. Every family is either backed by a COMPLETED full-settings-cell run in
    `verification-dev.json` whose cycles/epochs match `families.json`, or
    explicitly marked `unverified` with the ticket that explains why. D9 ships
    a template only once a live run has COMPLETED on its own defaults, so a
    pre-fill edited without a fresh run must fail rather than inherit the old
    run's credibility.
12. Every family renders, for every (category, framework) pair it owns — read
    from the family's own lists, never hardcoded — to a settings cell that
    is valid Python and carries no unsubstituted placeholder -- the one check
    that actually executes the render contract rather than describing it.
11. No setter is offered to a category that refuses it: one in the shared
    settings cell must be accepted by EVERY category in the family, and one in
    a gated fragment by every category its gate admits. This is the check that
    closes the class -- three separate instances of it were found by hand
    before it existed.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "notebooks", "templates")
TABLE = os.path.join(TEMPLATES, "families.json")
EVIDENCE = os.path.join(TEMPLATES, "verification-dev.json")

MAX_EFFECTIVE_PASSES = 20

DEAD_SETTERS = (
    "horizontal_flip",
    "vertical_flip",
    "samplewise_center",
    "samplewise_std_normalization",
    "layers_freeze",
)

COLAB_MARKERS = ("colab.research.google.com", "google.colab", "drive.mount")

# Internal references must not appear in a TEMPLATE CELL. The templates render
# into a peer's notebook, so anything here is shown to people outside the org
# — and this repository is public, so its prose is too. The survival banner shipped with a private
# tracker id AND an RFC id rendered to strangers; grep-expressible, so it is a
# rule rather than something to stay vigilant about.
# Match the FORM, not a list of spellings. The first version enumerated the
# two leaks that had already happened (one `<repo>#N` reference, an RFC id) and so would
# have missed `client#12`, `model-zoo#7`, or a different internal host — the
# same "instrument shaped by what I already held" trap this checker keeps
# finding elsewhere.
INTERNAL_REF = re.compile(
    r"""(
          \b[a-z0-9][a-z0-9._-]*\#\d+          # any <repo>#N cross-reference
        | \bRFC[- ]?[A-Z]*-?\d+                # RFC-NNNN, RFC-AREA-NNNN
        # Any tracebloc host EXCEPT the public ones the templates legitimately
        # link (docs. and ai.). Written as a negative lookahead rather than a
        # list of internal hosts, so a new internal host is caught by default
        # instead of needing to be enumerated — which is the whole point of
        # matching the form.
        | \b(?!docs\.|ai\.)[a-z0-9-]+\.tracebloc\.io\b
        | \b(?:dev|staging)-api\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# The public docs/ai links the templates legitimately carry are URLs, and a
# heading anchor on those pages can be a bare number — `hyperparameters#1-optimizer`,
# `hyperparameters#3-loss-function` (the rendered GitBook/Mintlify ids for
# "1. Optimizer" / "3. Loss Function"). INTERNAL_REF's `<repo>#N` branch reads
# the `#1` / `#3` fragment of such a URL as a cross-reference, so strip the
# allowed `docs.`/`ai.` URLs (the SAME two hosts the host branch above exempts)
# BEFORE scanning. A real leak like `example-repo#4242` is not inside one
# of these URLs, so it still survives the strip and is still caught.
ALLOWED_DOC_URL = re.compile(r"https?://(?:docs|ai)\.tracebloc\.io/\S*", re.IGNORECASE)

APPLIES_TO_KEYS = {"category", "framework", "dataset_flag"}
#: What each axis's value must BE. `category`/`framework` are lists the renderer
#: membership-tests; `dataset_flag` is a single string (rule 9b enforces that
#: separately, and `render()` only tests the key's presence). Keyed off
#: APPLIES_TO_KEYS so a new axis cannot be added without declaring its shape.
AXIS_VALUE_TYPES = {"category": list, "framework": list, "dataset_flag": str}
assert set(AXIS_VALUE_TYPES) == APPLIES_TO_KEYS, "declare a shape for every axis"

TRACEBLOC_CELL_KEYS = {"applies_to", "settings_fragment"}

# The context keys the render contract defines (notebooks/templates/README.md).
# A `{{ key }}` outside this set would render as literal text in the peer's
# notebook, so it is a typo, not an extension point.
CONTEXT_KEYS = {
    "use_case",
    "dataset_id",
    "category",
    "framework",
    "edge_count",
    "records_per_edge",
    "experiment_name",
    "model_path",
    "validation_split",
    "training_classes",
    "data_type",
    "feature_points",
    "sequence_length",
    "forecast_horizon",
    "scaler",
    "tokenizer_path",
}

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
# A lone `{ name }` is a placeholder someone lost a brace on -- exactly what an
# f-string does to `{{ name }}`. Deliberately requires a bare identifier so it
# cannot match a real dict literal like {"type": "constant"}.
HALF_PLACEHOLDER = re.compile(
    r"(?<!\{)\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}(?!\})"
)

# Every category the six families must partition between them. Kept explicit so
# adding a category to the platform without giving it a family fails here rather
# than silently rendering nothing.
ALL_CATEGORIES = {
    "image_classification",
    "object_detection",
    "semantic_segmentation",
    "keypoint_detection",
    "text_classification",
    "sentence_pair_classification",
    "token_classification",
    "causal_language_modeling",
    "seq2seq",
    "masked_language_modeling",
    "embeddings",
    "tabular_classification",
    "tabular_regression",
    "time_series_classification",
    "time_series_forecasting",
    "time_to_event_prediction",
}


_NLP = (
    "text_classification",
    "token_classification",
    "sentence_pair_classification",
    "masked_language_modeling",
    "causal_language_modeling",
    "seq2seq",
    "embeddings",
)
_TS = ("time_series_classification", "time_series_forecasting")
_TABULARISH = ("tabular_classification", "tabular_regression", "time_to_event_prediction")

# Which categories each setter actually accepts, read off the SDK's own gates
# (tracebloc/training/shape.py, preprocessing.py, link_model_dataset.py).
# Partial by design: a setter absent here is simply not checked. Offering a
# setter to a category that refuses it is at best a red "not supported" line on
# the peer's pre-filled cell, and at worst -- where the gate calls _on_invalid
# rather than _on_unsupported -- a poisoned plan whose Start then blocks.
SETTER_CATEGORIES = {
    "feature_points": ("tabular_classification", "time_to_event_prediction"),
    "sequence_length": _NLP + _TS,
    "forecast_horizon": ("time_series_forecasting",),
    "missingness_indicators": ("time_series_classification",),
    "scaler": _TABULARISH + _TS,
    "handle_missing_values": _TABULARISH + _TS,
    "imputation_strategy": _TABULARISH + _TS,
    "encoding_strategy": _TABULARISH + ("time_series_classification",),
    "normalize_features": _TABULARISH + ("time_series_classification",),
    "enable_lora": _NLP,
    "set_lora_parameters": _NLP,
}

#: The ONE spelling of "a `training.<setter>(` call", shared by the discovery scan and
#: by `called_with_all` so the two cannot disagree about what a call is.
#:
#: They did disagree, and it cost two holes (@LukasWodka in review). `SETTER_CALL`
#: was `training\.([a-z_]+)\s*\(` -- no `\s*` around the DOT -- and ran on the RAW
#: source, while `called_with_all` had `training\s*\.\s*` and ran on `code_only(src)`:
#:
#:   `training . optimizer("adamw")`   Start APPLIES it; discovery missed it, so rule 13
#:                                     never compared it to the record and the family
#:                                     stayed verified.
#:   a commented mapped setter         discovery FOUND it (comments survive a raw scan)
#:                                     but `called_with_all` returned nothing, so the
#:                                     comparison loop never ran and a stale COMPLETED
#:                                     run kept licensing the cell.
#:
#: One fragment, used by both. That is the derive-not-restate rule this checker exists to
#: enforce, applied to the checker itself.
SETTER_DOT = r"training\s*\.\s*"
SETTER_CALL = re.compile(SETTER_DOT + r"([a-z_]+)\s*\(")

# A syntactically representative value per context key, used only to render a
# template and compile the result. The point is not the values but that every
# placeholder sits in a position where a real value parses -- a `{{ key }}`
# inside a string literal and one standing as a bare argument are not
# interchangeable, and nothing else would catch the difference.
SAMPLE_CONTEXT = {
    "use_case": "Chest X-ray triage",
    "dataset_id": "d0gfu0c1",
    "category": "image_classification",
    "framework": "pytorch",
    "edge_count": "2",
    "records_per_edge": "30",
    "experiment_name": "resnet_18 on Chest X-ray triage #1",
    "model_path": "/models/resnet_18.py",
    "validation_split": "0.2",
    "training_classes": '{"cat": 15, "dog": 15}',
    "data_type": "rgb",
    "feature_points": "3",
    "sequence_length": "128",
    "forecast_horizon": "1",
    "scaler": "StandardScaler",
    "tokenizer_path": "tokenizer.json",
}



# --- one definition of each artefact question -----------------------------
#
# These existed seven times over, hand-rolled with different fallbacks, and the
# divergence was a finding in its own right: rule 16 skipped a cell as "the
# settings cell" that `render()` simultaneously treated as main, and a
# `"applies_to": null` crashed rule 14 with AttributeError instead of
# reporting. One definition each, used everywhere.

SETTINGS_MARKER = "# Settings — the complete plan for this run, as plain SDK calls."

# The only dataset flag the platform actually has (`_linking.py` is the source).
# RESTATED MIRRORS, acknowledged rather than implied. These three are copies
# of facts owned elsewhere -- ALL_CATEGORIES and SETTER_CATEGORIES of the
# engine registry and the SDK's per-setter gates, KNOWN_DATASET_FLAGS of the
# backend's dataset flags -- and only `single_pass_frameworks` is cross-checked
# against its source (rule 14b). The others go stale silently, which is the
# same objection this repo's CLAUDE.md raises about restating the SDK's Python
# bound. Deriving them needs an importable engine registry; until then the
# honest position is that they are unverified copies and this comment says so.
KNOWN_DATASET_FLAGS = {"allow_feature_modification"}



# --- rule 13's field map --------------------------------------------------
#
# Which record field each settings-cell literal must agree with. Binding only
# status/cycles/epochs was a finding: aggregation_strategy, optimizer, seed and
# the rest could all change and inherit the old run's credibility, while the
# README claimed "a new number cannot inherit the old run's credibility".
#
# `None` means "the record does not carry this, so do not claim it is
# verified" -- listed explicitly rather than omitted, so adding a setter to a
# template forces a decision here instead of silently going unchecked.
#
# Several settings land under a DIFFERENT record name than the setter: a text
# `sequence_length(N)` and a tabular `feature_points(N)` both arrive as
# `data_shape`, which is why this is a map and not a name match.
EVIDENCE_FIELD_MAP = {
    "cycles": "cycles",
    "epochs": "epochs",
    "aggregation_strategy": "aggregation_strategy",
    "optimizer": "optimizer",
    "seed": "seed",
    "data_type": "data_type",
    "shuffle": "shuffle",
    "sequence_length": "data_shape",
    "feature_points": "data_shape",
    "validation_split": "validation_split",
    "learning_rate": "learningRate",
    # These ARE recorded -- verification-dev.json exports them for the tabular
    # and survival families -- and marking them None meant a change to any of
    # them inherited the old COMPLETED run. `scaler` lands under
    # `tabular_scaler`, which is the same setter-name-vs-record-name skew that
    # makes this a map rather than a name match.
    "scaler": "tabular_scaler",
    "handle_missing_values": "handle_missing_values",
    "imputation_strategy": "imputation_strategy",
    "encoding_strategy": "encoding_strategy",
    "normalize_features": "normalize_features",
    # Deliberately unverifiable against the record, and named so:
    "training_classes": None,      # the record stores it as `subdataset`
    "terminate_on_nan_callback": "callbacks",
    "missingness_indicators": None,
    "forecast_horizon": None,
    "loss_function": None,         # every occurrence is commented out
    "enable_lora": None,
    "set_lora_parameters": None,
    "early_stop_callback": None,
    "model_checkpoint_callback": None,
    "reduce_lr_callback": None,
    "feature_interaction": None,
    "get_features": None,
    "experiment_name": None,       # pre-filled per run, not a family constant
    # The augmentation group: every line is COMMENTED OUT in the template, so
    # the run could not have exercised any of them. Listed rather than omitted
    # so the map stays a complete statement about what the evidence covers.
    "rotation_range": None,
    "width_shift_range": None,
    "height_shift_range": None,
    "brightness_range": None,
    "shear_range": None,
    "zoom_range": None,
    "channel_shift_range": None,
    "fill_mode": None,
    "cval": None,
    "rescale": None,
}


def _literal_matches(setter, literal, recorded):
    """Does a settings-cell literal agree with the recorded value?"""
    if setter == "terminate_on_nan_callback":
        # No argument; presence in the cell must mean presence on the record.
        return "terminateOnNaN" in str(recorded)
    text = str(recorded)
    lit = literal.strip()
    if lit.startswith(("'", '"')) and lit[-1:] in "'\"":
        return lit[1:-1] == text
    try:
        return json.loads(lit.replace("'", '"')) == json.loads(
            text.replace("'", '"')
        )
    except Exception:  # noqa: BLE001 - fall back to a text compare
        return lit == text


def cell_meta(cell):
    """The cell's `metadata.tracebloc` dict, never None and never non-dict.

    `tracebloc: true` used to reach `.get` and raise AttributeError from
    `settings_source`, and a list-valued `applies_to` did the same in rule 8 —
    a fail-closed traceback, but the header above promises these helpers end
    that class, so they have to actually end it.
    """
    meta = cell.get("metadata")
    tb = (meta if isinstance(meta, dict) else {}).get("tracebloc")
    return tb if isinstance(tb, dict) else {}


def cell_gate(cell):
    """The cell's applies_to, or {} when it does not gate anything.

    An `applies_to` carrying no key the renderer understands -- `{}`, or only
    unknown keys -- is UNGATED, because that is what the renderer does with it.
    Reading it as "gated" is how a fragment with `applies_to: {}` slipped past
    rule 9 and then appended itself to every render, silently overriding the
    family's own values: the exact second-settings-cell that rule 9's message
    claims to prevent.
    """
    applies = cell_meta(cell).get("applies_to")
    if not isinstance(applies, dict):
        return {}
    # Keys the renderer understands, INCLUDING ones whose list is empty.
    #
    # An earlier fix dropped empty lists here so the file would "agree", and
    # that INVERTED the bug: the pod treats `category: []` as admitting NOBODY,
    # while dropping the key made the checker treat it as admitting EVERYBODY.
    # The checker then kept such a cell for every pair and counted it as
    # offered to the whole family — the opposite of what ships. The gate is
    # reported as written; `gate_audience()` below applies the pod's meaning.
    # SHAPE, not just the key. An axis value of the wrong type used to reach every
    # reader: `category: null` raised TypeError in `gate_audience` and in rule 9d's
    # membership test, and `category: "vision"` -- a bare string -- did something worse
    # than raise. It ITERATED, so `gate_audience` returned
    # ['v','i','s','i','o','n'] and the cell was read as offered to categories named
    # "v", "i", "s": a silent wrong answer rather than a crash (Bugbot reported the
    # TypeError; the string case was found while reproducing it).
    #
    # Dropped here and REPORTED by rule 8 off the raw dict -- the same division of
    # labour this function already uses for unknown keys. `render()` is no kinder to a
    # null gate than these rules were (`category not in None` raises there too), so the
    # checker's job is to name it, not to emulate it.
    return {
        k: v
        for k, v in applies.items()
        if k in APPLIES_TO_KEYS and isinstance(v, AXIS_VALUE_TYPES[k])
    }


def gates_nothing(cell):
    """True when applies_to carries no key the renderer understands.

    This is rule 9's question — `{}`, `null`, absent, or unknown-keys-only —
    and it is NOT the same as a key present with an empty list, which gates
    everything OUT rather than nothing.
    """
    return not cell_gate(cell)


def gate_audience(cell, axis, whole):
    """Who this cell is actually offered to on `axis`, the pod's way.

    * key absent  -> the whole family
    * key present -> exactly its values, EVEN IF EMPTY (nobody)

    Rules 11/14/15 previously used `gate or whole`, so an empty list fell
    through to `whole` and a `category: []` loss fragment satisfied rule 15
    while no pair could ever see it.
    """
    gate = cell_gate(cell)
    return list(gate[axis]) if axis in gate else list(whole)


def is_fragment(cell):
    return bool(cell_meta(cell).get("settings_fragment"))


def is_settings_cell(cell):
    """The single settings cell. One test, used by every rule and by render().

    Keyed on the generated marker rather than on prose, so a comment merely
    mentioning `training.experiment_name` cannot impersonate it.
    """
    if cell.get("cell_type") != "code":
        return False
    src = "".join(cell["source"])
    return SETTINGS_MARKER in src and not is_fragment(cell)



def render(nb, category, framework, with_flags=False):
    """Apply the render contract: drop what does not apply, substitute
    placeholders, concatenate surviving settings fragments into the settings
    cell. Returns the settings cell source."""
    main = ""
    fragments = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        applies = cell_gate(cell)
        # An empty list admits nobody, so `category not in []` drops the cell —
        # which is exactly what the pod does. Reported here rather than
        # normalised away.
        if "category" in applies and category not in applies["category"]:
            continue
        if "framework" in applies and framework not in applies["framework"]:
            continue
        if "dataset_flag" in applies and not with_flags:
            continue
        src = "".join(cell["source"])
        if is_fragment(cell):
            fragments.append(src)
        elif is_settings_cell(cell):
            main = src
    text = main + "\n" + "\n".join(fragments)
    for k, v in SAMPLE_CONTEXT.items():
        text = re.sub(r"\{\{\s*" + k + r"\s*\}\}", v.replace("\\", "\\\\"), text)
    return text


def derive_single_pass_frameworks(floor):
    """Read the single-pass framework set from the SDK, with its version.

    Returns (frameworks, version, problem). Exactly one of `frameworks` /
    `problem` is meaningful.

    Three things this used to get wrong, all found in the review that motivated this checker:

    * It imported whatever `tracebloc` happened to be on `sys.path` and
      reported "matches the installed SDK" with no version. Measured: `pip
      install tracebloc` under Python 3.9 silently backtracks to **0.8.1** — a
      release predating these templates — whose `_SURVIVAL_FRAMEWORKS` happens
      to match, so an ancient SDK "confirmed" the mirror. Hence the floor.
    * A bare `except Exception` reported a present-but-BROKEN SDK as "not
      importable", which is a different fact and hides a real problem.
    * "Cannot tell" printed a note and exited 0 — in CI, always. A
      cross-check that never runs where it gates is the defect this whole file
      exists to catch, so `TRACEBLOC_CHECK_STRICT` turns every
      cannot-tell into a failure.
    """
    try:
        import importlib.metadata as md

        version = md.version("tracebloc")
    except ModuleNotFoundError:
        return None, None, "the tracebloc distribution is not installed"
    except Exception as exc:  # noqa: BLE001 - metadata present but unreadable
        return None, None, f"tracebloc metadata unreadable: {exc!r}"

    if _version_tuple(version) < _version_tuple(floor):
        return (
            None,
            version,
            f"installed tracebloc {version} is below the floor {floor} recorded "
            f"in families.json, so it cannot confirm this mirror — an older "
            f"release can agree by coincidence",
        )
    try:
        from tracebloc.training.plan import _SURVIVAL_FRAMEWORKS
    except ImportError as exc:
        return None, version, f"tracebloc {version} is installed but not importable: {exc}"
    except Exception as exc:  # noqa: BLE001 - importable but raising
        return None, version, f"tracebloc {version} raised on import: {exc!r}"
    return (
        {getattr(f, "value", f) for f in _SURVIVAL_FRAMEWORKS},
        version,
        None,
    )


def _version_tuple(v):
    """The numeric release segment, for a floor comparison.

    Concatenating the digits of a chunk put a PRE-RELEASE ABOVE the floor:
    "1.0.7rc1" became (1, 0, 71), so a release candidate of the floor version
    read as newer than the floor. Split on the first non-digit instead, which
    makes 1.0.7rc1 -> (1, 0, 7) — equal to the floor, and accepted, which is
    the conservative reading for a floor rather than the flattering one.
    """
    parts = []
    for chunk in str(v).split(".")[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


STRICT = os.environ.get("TRACEBLOC_CHECK_STRICT") == "1"


def settings_source(nb):
    """Return the settings cell plus its fragments, or None if there is no cell.

    Fragments are concatenated into the settings cell at render time, so a
    value that lives in a gated fragment is still part of the settings the peer
    sees. Reading only the main cell would false-fail a template that gates a
    setting per framework — which is exactly what `cycles`/`epochs` need in the
    families with single-pass (sklearn / lifelines / scikit_survival) pairs.
    """
    main = None
    fragments = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if is_fragment(cell):
            fragments.append(src)
        elif is_settings_cell(cell):
            main = src
    if main is None:
        return None
    return "\n".join([main] + fragments)


def code_only(src):
    """`src` with comment text removed, so only LIVE calls remain.

    Two different questions were sharing one matcher:

      * "is this pre-fill APPLIED?"  — rules 3 and 13. Only live code counts.
      * "does this template OFFER this?" — rules 5, 6, 16. A commented
        `# training.start()` or a commented dead setter still invites a peer to
        uncomment it, so comments MUST count there.

    Anchoring `called_with_all` at line start used to exclude comments as a
    side effect. Widening it to catch `training . cycles (99)` lost that, and
    the regression was worse than the bug it fixed: commenting out the real
    `training.cycles(20)` and leaving the comment made rule 3 agree with
    families.json while Start applied the SDK default. Now the two questions
    use two matchers, on purpose.

    Deliberately naive about `#` inside string literals: the settings cells
    contain none, and a false "this is a comment" makes rule 3 report a
    MISSING pre-fill, which fails loudly rather than passing quietly.
    """
    out = []
    for line in src.split("\n"):
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


def called_with_all(src, method):
    """Every literal argument of `training.<method>(...)`, in order.

    Returning only the FIRST match was a finding: a second
    `training.cycles(99)` after the pre-fill left rule 3 reading 20 and passing
    while Python applied 99, bypassing the 20-pass ceiling the rule exists to
    hold. Callers must decide what more than one means; for a pre-fill it is
    always an error.
    """
    # Match every form `SETTER_CALL` does — `training . cycles ( 99 )`, an
    # indented call, an assignment — not just a line-start `training.m(`.
    # Widening to "all matches" while keeping the NARROW pattern left the hole
    # open: a second call Python would apply was still invisible to rule 3,
    # so the checker could print OK while Start ran a different value.
    # SETTER_DOT, not a second copy of it. The two spellings drifting is exactly the
    # defect this function's own comment describes one paragraph up.
    return re.findall(
        rf"{SETTER_DOT}{method}\s*\(\s*([^)]*?)\s*\)", code_only(src)
    )


def main() -> int:
    errors = []
    notes = []

    with open(TABLE) as fh:
        table = json.load(fh)
    families = table["families"]
    drift_ok = set(table["drift_correcting_strategies"])

    seen_categories = {}
    # Each notebook is parsed once here and reused by the later rules; the
    # families whose file is missing or unparseable are simply absent, which is
    # what lets those rules drop their own existence guards.
    loaded = {}

    for fam in families:
        key = fam["key"]
        cycles, epochs = fam["cycles"], fam["epochs"]
        strategy = fam["aggregation_strategy"]

        # Rule 1 — the budget ceiling.
        if cycles * epochs > MAX_EFFECTIVE_PASSES:
            errors.append(
                f"{key}: cycles*epochs = {cycles}*{epochs} = {cycles * epochs} "
                f"> {MAX_EFFECTIVE_PASSES}"
            )

        # Rule 2 — epochs > 1 needs a drift correction.
        if epochs > 1 and strategy not in drift_ok:
            errors.append(
                f"{key}: pre-fills epochs={epochs} with aggregation_strategy "
                f"'{strategy}', which carries no drift correction. Use one of "
                f"{sorted(drift_ok)}, or keep epochs at 1."
            )

        # Rule 4 — categories partition, notebook exists.
        for cat in fam["categories"]:
            if cat in seen_categories:
                errors.append(
                    f"category '{cat}' claimed by both '{seen_categories[cat]}' "
                    f"and '{key}'"
                )
            seen_categories[cat] = key
            if cat not in ALL_CATEGORIES:
                errors.append(f"{key}: unknown category '{cat}'")

        path = os.path.join(TEMPLATES, fam["template"])
        if not os.path.isfile(path):
            errors.append(f"{key}: template not found: {fam['template']}")
            continue
        try:
            with open(path) as fh:
                nb = json.load(fh)
        except json.JSONDecodeError as exc:
            errors.append(f"{key}: {fam['template']} is not valid JSON: {exc}")
            continue
        loaded[key] = nb

        # Join with a newline, not "": a cell's last source line carries no
        # trailing newline, so concatenating directly welds the last line of
        # one cell onto the first line of the next. That both hides
        # line-anchored matches and invents text that is in no cell.
        whole = "\n".join("".join(c["source"]) for c in nb["cells"])

        # Rule 3 moved INTO rule 12's per-pair loop. It used to read a
        # gate-blind CONCATENATION of the settings cell plus every fragment,
        # which made it count one `cycles` call per framework-gated fragment:
        # adding a second framework to a family failed the check, so the rule
        # actively BLOCKED a family from ever gaining a framework. Per pair is
        # the only frame where "single-pass means zero calls, everything else
        # exactly one equal to the table" is expressible at all.
        # Rule 5 — no template starts a run. Deliberately not anchored to the
        # start of a line: a commented-out `# training.start()` is still a
        # template telling a peer to bypass the button.
        if re.search(r"training\s*\.\s*start\s*\(", whole):
            errors.append(
                f"{key}: {fam['template']} calls training.start(). Start is a "
                f"button; no cell may start a run."
            )

        # Rule 6 — no dead setters.
        for dead in DEAD_SETTERS:
            if f"training.{dead}" in whole:
                errors.append(
                    f"{key}: {fam['template']} mentions dead setter "
                    f"'{dead}' (a no-op on every surviving framework)"
                )

        # Rule 7 — no Colab lines.
        for marker in COLAB_MARKERS:
            if marker in whole:
                errors.append(
                    f"{key}: {fam['template']} carries the Colab-specific "
                    f"marker '{marker}'; templates render in the pod"
                )

        # Rule 10 — placeholders are real context keys, and none is
        # half-written. `{ scaler }` instead of `{{ scaler }}` renders as
        # literal text in the peer's notebook and no other check sees it.
        for name in sorted(set(PLACEHOLDER.findall(whole))):
            if name not in CONTEXT_KEYS:
                errors.append(
                    f"{key}: {fam['template']} uses placeholder "
                    f"{{{{ {name} }}}}, which is not a context key "
                    f"({sorted(CONTEXT_KEYS)})"
                )
        for name in sorted(set(HALF_PLACEHOLDER.findall(whole))):
            if name in CONTEXT_KEYS:
                errors.append(
                    f"{key}: {fam['template']} has a half-written placeholder "
                    f"{{ {name} }} — it needs double braces to be substituted"
                )

        # Rules 8 and 9 — the render metadata is well formed.
        for i, cell in enumerate(nb["cells"]):
            # BEFORE the falsy-guard: cell_meta() normalises a non-dict to {},
            # so a `tracebloc: true` would `continue` here and the cell would
            # be silently treated as ungated. Name it first.
            raw_tb = (cell.get("metadata") or {}).get("tracebloc")
            if raw_tb is not None and not isinstance(raw_tb, dict):
                errors.append(
                    f"{key}: {fam['template']} cell {i} has a "
                    f"metadata.tracebloc that is {type(raw_tb).__name__}, not "
                    f"an object. Nothing crashes, because cell_meta() "
                    f"normalises it — but silence would turn a gated cell into "
                    f"an UNGATED one, trading a loud failure for a quiet one."
                )
            tb = cell_meta(cell)
            if not tb:
                continue

            unknown_meta = set(tb) - TRACEBLOC_CELL_KEYS
            if unknown_meta:
                errors.append(
                    f"{key}: {fam['template']} cell {i} has unknown "
                    f"metadata.tracebloc keys {sorted(unknown_meta)}; known "
                    f"keys are {sorted(TRACEBLOC_CELL_KEYS)}"
                )

            raw_applies = tb.get("applies_to")
            if raw_applies is not None and not isinstance(raw_applies, dict):
                errors.append(
                    f"{key}: {fam['template']} cell {i} has an applies_to that "
                    f"is {type(raw_applies).__name__}, not an object — the "
                    f"renderer reads keys off it. Reported rather than raising, "
                    f"which this file's shared helpers exist to guarantee."
                )
            applies = cell_gate(cell)
            if raw_applies is not None:
                # RAW dict: cell_gate() has already dropped unknown keys, so
                # checking its output could only ever catch the
                # typo-is-the-only-key case. A typo ALONGSIDE a valid key is
                # the case this rule was written for — it turns a conditional
                # cell unconditional on the axis that was misspelt.
                # `isinstance` FIRST: a non-dict `applies_to` is already reported
                # above, and `.items()` on the LIST form raises AttributeError --
                # which broke mutation 8c the moment this loop was added. The
                # sibling `set(raw_applies or {})` below tolerates a list by
                # accident; this one has to say so.
                axis_items = raw_applies.items() if isinstance(raw_applies, dict) else ()
                for axis, value in axis_items:
                    if axis not in APPLIES_TO_KEYS:
                        continue  # named by the unknown-keys check below
                    if not isinstance(value, AXIS_VALUE_TYPES[axis]):
                        errors.append(
                            f"{key}: {fam['template']} cell {i} has an "
                            f"applies_to.{axis} that is "
                            f"{type(value).__name__}, not "
                            f"{AXIS_VALUE_TYPES[axis].__name__} — the renderer "
                            f"membership-tests it. Reported rather than raising: "
                            f"a null crashed the checker and a bare string was "
                            f"read as its own characters."
                        )
                unknown = set(raw_applies or {}) - APPLIES_TO_KEYS
                if unknown:
                    errors.append(
                        f"{key}: {fam['template']} cell {i} has applies_to keys "
                        f"{sorted(unknown)}; known keys are "
                        f"{sorted(APPLIES_TO_KEYS)}"
                    )
                for cat in applies.get("category", []):
                    if cat not in fam["categories"]:
                        errors.append(
                            f"{key}: {fam['template']} cell {i} is gated on "
                            f"category '{cat}', which is not in this family "
                            f"({sorted(fam['categories'])}) — it would never "
                            f"render"
                        )
                # The same check on the framework axis, which had none. Rule 12
                # renders per (category, framework) from the family's own
                # lists, so a gate naming a framework the family does not ship
                # is a cell nothing can ever render — and nothing would say so.
                for fw in applies.get("framework", []):
                    if fw not in fam.get("frameworks", ()):
                        errors.append(
                            f"{key}: {fam['template']} cell {i} is gated on "
                            f"framework '{fw}', which is not in this family "
                            f"({sorted(fam.get('frameworks', ()))}) — it would "
                            f"never render"
                        )

            # Rule 9 — a settings fragment with no EFFECTIVE gate is not a
            # fragment. It applies to the whole family, so it belongs in the
            # settings cell; left as a fragment it appends itself after the
            # main block on every render and silently overrides the family's
            # own values — the second settings cell this rule claims to stop.
            #
            # "No effective gate" now means what the RENDERER means, via
            # cell_gate(): absent, null, `{}`, or only unknown keys. Testing
            # `applies_to is None` let `{}` through, and a `{}`-gated fragment
            # carrying optimizer/seed overrode sgd/0 on every pair.
            if is_fragment(cell) and gates_nothing(cell):
                raw = cell_meta(cell).get("applies_to", "<absent>")
                errors.append(
                    f"{key}: {fam['template']} cell {i} is a settings_fragment "
                    f"whose applies_to ({raw!r}) gates nothing, so the "
                    f"renderer keeps it for every pair. An ungated fragment "
                    f"belongs in the settings cell."
                )

            # Rule 9c — a gate key present with an EMPTY list renders for no
            # pair at all. cell_gate() drops it so nothing mistakes it for a
            # gate, but silence would leave a cell that can never appear.
            raw_gate = cell_meta(cell).get("applies_to")
            if isinstance(raw_gate, dict):
                for gk, gv in raw_gate.items():
                    if gk in APPLIES_TO_KEYS and gv == []:
                        errors.append(
                            f"{key}: {fam['template']} cell {i} is gated on "
                            f"an EMPTY {gk} list, which admits no pair — the "
                            f"cell can never render. Remove the key, or name "
                            f"the {gk} values it applies to."
                        )

            # Rule 9b — a dataset_flag must name a flag that exists, and a
            # gate that admits the whole family is not gating.
            flag = cell_gate(cell).get("dataset_flag")
            if flag is not None and not isinstance(flag, str):
                errors.append(
                    f"{key}: {fam['template']} cell {i} has a dataset_flag "
                    f"that is {type(flag).__name__}, not a string"
                )
            elif flag is not None and flag not in KNOWN_DATASET_FLAGS:
                errors.append(
                    f"{key}: {fam['template']} cell {i} is gated on "
                    f"dataset_flag '{flag}', which is not a flag the platform "
                    f"has ({sorted(KNOWN_DATASET_FLAGS)}). render() never "
                    f"reads the value, so a typo here renders the cell always "
                    f"or never with nothing to say so."
                )
            # Does the WHOLE gate drop any pair the family renders? Asking
            # only about the `category` key both missed a framework gate that
            # admits every framework the family ships, and FALSELY rejected a
            # framework-gated fragment that also listed every category —
            # render() does drop that one, on the framework axis.
            gate = cell_gate(cell)
            if gate and set(gate) & APPLIES_TO_KEYS:
                # EVERY axis render() drops on, not two of the three. Asking only
                # about `category` missed a framework gate admitting every framework
                # the family ships, and falsely rejected a framework-gated fragment
                # that also listed every category. Adding `framework` fixed those and
                # left the SAME defect one axis along: a gate carrying `dataset_flag`
                # was judged on category/framework alone and called a no-op, when
                # render() drops it on the flags-off pass (`render()`: `if
                # "dataset_flag" in applies and not with_flags: continue`).
                #
                # So the space is the render signature's own product --
                # (category, framework, with_flags) -- and the three drop conditions
                # below mirror render's three, in its order. APPLIES_TO_KEYS is the
                # authority for how many axes there are, so a fourth gate key added
                # there cannot silently leave this rule judging three.
                renders = [
                    (c, f, w)
                    for c in fam["categories"]
                    for f in fam.get("frameworks", ("pytorch",))
                    for w in (False, True)
                ]
                drops_some = any(
                    ("category" in gate and c not in gate["category"])
                    or ("framework" in gate and f not in gate["framework"])
                    or ("dataset_flag" in gate and not w)
                    for c, f, w in renders
                )
                if not drops_some:
                    errors.append(
                        f"{key}: {fam['template']} cell {i} is gated on "
                        f"{ {k: gate[k] for k in gate if k in APPLIES_TO_KEYS} } "
                        f"but that admits EVERY render the family produces, so "
                        f"the gate never drops it. Remove it, or narrow it to "
                        f"what actually differs."
                    )

    # Rule 11 — no setter offered to a category that refuses it.
    for fam in families:
        nb = loaded.get(fam["key"])
        if nb is None:
            continue
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            tb = cell_meta(cell)
            audience = tuple(gate_audience(cell, "category", fam["categories"]))
            src = "".join(cell["source"])
            for setter in sorted(set(SETTER_CALL.findall(src))):
                allowed = SETTER_CATEGORIES.get(setter)
                if allowed is None:
                    continue
                refused = [c for c in audience if c not in allowed]
                if refused:
                    errors.append(
                        f"{fam['key']}: {fam['template']} cell {i} offers "
                        f"training.{setter}() to {sorted(refused)}, which "
                        f"refuse it. Gate the cell, or move the call into a "
                        f"fragment for {sorted(set(audience) & set(allowed))}."
                    )

    # Rule 12 — the render contract, executed.
    for fam in families:
        nb = loaded.get(fam["key"])
        if nb is None:
            continue
        # The family's OWN frameworks. Hardcoding ("pytorch", "sklearn") both
        # skipped real pairs (survival's lifelines / scikit_survival were never
        # rendered, so a fragment gated only on those could carry a syntax
        # error and stay green) and rendered illegal ones (vision + sklearn).
        # `frameworks` is per FAMILY, so this over-approximates: it renders
        # e.g. (time_series_classification, sklearn) though the zoo ships no
        # sklearn model there. Over-approximating is the safe direction for a
        # validator — it can only demand more validity, never less — and the
        # precise per-category set is the engine registry's legal-pair matrix
        # in the engine registry, not something to restate here.
        for cat in fam["categories"]:
            for framework in fam.get("frameworks", ("pytorch",)):
                # Render twice and validate BOTH: flags off is what most
                # datasets get, flags on is the only way a flag-gated fragment
                # is ever compiled. An earlier version assigned `text` in the
                # loop but validated after it, so only the flags-on pass was
                # ever checked — it swapped coverage of the common path for the
                # rare one and no mutation caught it, because the one template
                # with a flag-gated fragment was not the one being mutated.
                for with_flags in (False, True):
                    mode = "flags on" if with_flags else "flags off"
                    where = f"({cat}, {framework}, {mode})"
                    text = render(nb, cat, framework, with_flags)
                    if not text.strip():
                        errors.append(f"{fam['key']}: renders empty for {where}")
                        continue
                    leftover = PLACEHOLDER.findall(text) + HALF_PLACEHOLDER.findall(
                        text
                    )
                    leftover = [n for n in leftover if n in CONTEXT_KEYS]
                    if leftover:
                        errors.append(
                            f"{fam['key']}: {where} renders with unsubstituted "
                            f"placeholder(s) {sorted(set(leftover))}"
                        )
                    try:
                        compile(
                            text,
                            f"{fam['template']}::{cat}/{framework}/{mode}",
                            "exec",
                        )
                    except SyntaxError as exc:
                        errors.append(
                            f"{fam['key']}: {where} renders to invalid Python: "
                            f"{exc.msg} at line {exc.lineno}"
                        )
                        continue

                    # Rule 3, per rendered pair. A single-pass framework gets
                    # ZERO calls (the SDK forces 1 and only warns, rule 14);
                    # every other pair gets exactly ONE, equal to the table.
                    # Audit BOTH modes. Skipping flags-on left a `cycles(99)`
                    # inside a flag-gated fragment invisible: verified families
                    # are backstopped by rule 13, but the UNVERIFIED one is
                    # not, so there the pre-fill silently changes.
                    single = framework in set(
                        table.get("single_pass_frameworks", ())
                    )
                    # `aggregation_strategy` was bound to the table by the
                    # old whole-file rule 3 and the per-pair move DROPPED it,
                    # so flipping it in families.json alone passed. Rule 13
                    # catches the notebook side; nothing bound the TABLE, which
                    # this file's own header calls the single source.
                    for method, expected in (
                        ("cycles", str(fam["cycles"])),
                        ("epochs", str(fam["epochs"])),
                        (
                            "aggregation_strategy",
                            f'"{fam["aggregation_strategy"]}"',
                        ),
                    ):
                        got = called_with_all(text, method)
                        if single and method in ("cycles", "epochs"):
                            if got:
                                errors.append(
                                    f"{fam['key']}: {where} calls "
                                    f"training.{method}({got}) on a "
                                    f"single-pass framework, which forces it "
                                    f"to 1 and only warns — the value is "
                                    f"silently dropped"
                                )
                        elif len(got) != 1:
                            errors.append(
                                f"{fam['key']}: {where} calls "
                                f"training.{method}() {len(got)} times "
                                f"({got}); exactly one is required, because "
                                f"Python applies the last"
                            )
                        elif got[0] != expected:
                            errors.append(
                                f"{fam['key']}: {where} has "
                                f"training.{method}({got[0]}) but "
                                f"families.json says {expected}"
                            )

    # Rule 14b — the mirrored framework set must match the SDK's own.
    declared_single_pass = set(table.get("single_pass_frameworks", ()))
    floor = table.get("sdk_version_floor")
    if not floor:
        errors.append(
            "families.json has no `sdk_version_floor`, so the mirror "
            "cross-check cannot tell a current SDK from one that predates "
            "these templates."
        )
    else:
        derived, version, problem = derive_single_pass_frameworks(floor)
        if problem:
            msg = (
                f"single_pass_frameworks NOT cross-checked ({problem}). It "
                f"mirrors `tracebloc.training.plan._SURVIVAL_FRAMEWORKS`."
            )
            if STRICT:
                errors.append(
                    msg + " TRACEBLOC_CHECK_STRICT=1, and a mirror verified "
                    "nowhere that gates is the whole defect."
                )
            else:
                notes.append(msg + " Set TRACEBLOC_CHECK_STRICT=1 to make this fail.")
        elif derived != declared_single_pass:
            errors.append(
                f"single_pass_frameworks in families.json is "
                f"{sorted(declared_single_pass)} but tracebloc {version} has "
                f"_SURVIVAL_FRAMEWORKS = {sorted(derived)}. The mirror has "
                f"drifted — update families.json, and re-check which families "
                f"need cycles/epochs gated on framework."
            )
        else:
            notes.append(
                f"single_pass_frameworks matches tracebloc {version} "
                f"(floor {floor}): {sorted(derived)}."
            )

    # Rule 16 — every setter call reachable by Start, and after the link.
    for fam in families:
        nb = loaded.get(fam["key"])
        if nb is None:
            continue
        link_at = settings_at = None
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if "link_model_dataset" in src:
                link_at = i if link_at is None else link_at
            if is_settings_cell(cell):
                settings_at = i
                continue
            if is_fragment(cell):
                continue
            # Match a CALL, not the substring "training." — the tokenizer cell's
            # prose ends "not at training." and a substring test flags it.
            calls = sorted(set(SETTER_CALL.findall(src)))
            if calls:
                errors.append(
                    f"{fam['key']}: {fam['template']} cell {i} calls "
                    f"training.{'/'.join(calls)}() outside the settings cell "
                    f"and outside any settings fragment. Start executes only "
                    f"the settings cell, so these are inert. Move them into "
                    f"the settings cell, or into a gated fragment."
                )
        if settings_at is None:
            continue
        if link_at is None:
            errors.append(
                f"{fam['key']}: {fam['template']} never assigns `training` via "
                f"link_model_dataset()"
            )
        elif link_at > settings_at:
            errors.append(
                f"{fam['key']}: {fam['template']} has its settings cell "
                f"(cell {settings_at}) BEFORE the link (cell {link_at}), so "
                f"`training` is unassigned when the settings run."
            )

    # Rule 17 — no internal reference in a cell a peer will read.
    for fam in families:
        nb = loaded.get(fam["key"])
        if nb is None:
            continue
        for i, cell in enumerate(nb["cells"]):
            text = ALLOWED_DOC_URL.sub("", "".join(cell["source"]))
            found = sorted(set(INTERNAL_REF.findall(text)))
            if found:
                errors.append(
                    f"{fam['key']}: {fam['template']} cell {i} carries "
                    f"internal reference(s) {found}. These templates render "
                    f"into a peer's notebook, so this is shown outside the "
                    f"org — describe the situation instead of citing a tracker."
                )

    # Rule 3b — EXACTLY ONE settings cell per template.
    #
    # Nothing asserted this. `render()`, `settings_source()` and rule 16 each
    # take the LAST match silently, so a second marker cell inserted BEFORE the
    # real one carrying `cycles(99)` passed with exit 0 while Start applied 99.
    for fam in families:
        nb = loaded.get(fam["key"])
        if nb is None:
            continue
        found = [i for i, c in enumerate(nb["cells"]) if is_settings_cell(c)]
        if len(found) != 1:
            errors.append(
                f"{fam['key']}: {fam['template']} has {len(found)} settings "
                f"cells (indices {found}); exactly one is required. Every rule "
                f"and render() silently take the last, so a second one is a "
                f"pre-fill nobody audits."
            )

    # Rule 15 — the custom-loss offering must reach EVERY category that takes
    # one, not merely appear somewhere in the file.
    #
    # The first version of this rule asked `any("training.loss_function" in
    # src)`, which is presence, not coverage — so re-introducing the very
    # defect it was written to stop (gating vision's loss cell to
    # object_detection alone, hiding it from the other three) PASSED. A rule
    # that cannot fail on its own motivating case is the shape this whole
    # checker exists to catch, and it was in the checker.
    no_loss = set(table.get("no_custom_loss_families", ()))
    for fam in families:
        nb = loaded.get(fam["key"])
        if nb is None:
            continue
        offered_to = set()
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            if "training.loss_function" not in "".join(cell["source"]):
                continue
            offered_to |= set(gate_audience(cell, "category", fam["categories"]))

        if fam["key"] in no_loss:
            if offered_to:
                errors.append(
                    f"{fam['key']}: {fam['template']} offers a custom loss to "
                    f"{sorted(offered_to)}, but this family's objective is "
                    f"intrinsic — the SDK refuses a loss.py at upload, so the "
                    f"cell advertises a rejection."
                )
            continue

        missing = set(fam["categories"]) - offered_to
        if missing:
            where = "nowhere in the template" if not offered_to else (
                f"only to {sorted(offered_to)}"
            )
            errors.append(
                f"{fam['key']}: {fam['template']} offers a custom loss "
                f"{where}, so {sorted(missing)} never see it although they "
                f"accept one. Put the call in the settings cell, or in "
                f"fragments whose gates cover every category in the family."
            )

    # Rule 14 — cycles/epochs must not reach a single-pass framework.
    single_pass = set(table.get("single_pass_frameworks", ()))
    for fam in families:
        nb = loaded.get(fam["key"])
        if nb is None:
            continue
        fam_forced = set(fam.get("frameworks", ())) & single_pass
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            calls = [
                m for m in ("cycles", "epochs")
                if re.search(rf"^\s*training\.{m}\(", src, flags=re.MULTILINE)
            ]
            if not calls:
                continue
            audience = set(gate_audience(cell, "framework", fam.get("frameworks", ())))
            reached = audience & single_pass
            if reached:
                errors.append(
                    f"{fam['key']}: {fam['template']} cell {i} offers "
                    f"training.{'/'.join(calls)}() to {sorted(reached)}, which "
                    f"force both to 1 and only warn — the pre-fill would be "
                    f"silently unhonoured there. Gate the cell on "
                    f"framework, as the family's own pairs include "
                    f"{sorted(fam_forced)}."
                )

    # Rule 13 — D9's "ships only after a COMPLETED run" made mechanical.
    try:
        with open(EVIDENCE) as fh:
            evidence = json.load(fh)["runs"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"verification-dev.json unreadable: {exc}")
        evidence = {}
    by_family = {v["family"]: v for v in evidence.values()}
    for fam in families:
        key = fam["key"]
        rec = by_family.get(key)
        run = (rec or {}).get("rounds", {}).get("2_full_settings_cell") or {}
        # Compare EVERY literal the rendered settings cell sets against the
        # record, through EVIDENCE_FIELD_MAP, and treat a missing field as a
        # failure rather than as agreement.
        mismatches = []
        nb_fam = loaded.get(key)
        if run.get("status") != "COMPLETED":
            mismatches.append(f"status={run.get('status')!r}")
        elif nb_fam is not None:
            src = settings_source(nb_fam) or ""
            # `code_only`, matching `called_with_all` below. Discovering on the RAW
            # source found commented setters that `called_with_all` then returned nothing
            # for, so the comparison loop never ran and the stale COMPLETED run kept
            # licensing the cell. Rule 13 asks "is this pre-fill APPLIED?", which
            # `code_only`'s docstring names as the live-code-only question.
            for setter in sorted(set(SETTER_CALL.findall(code_only(src)))):
                if setter not in EVIDENCE_FIELD_MAP:
                    mismatches.append(
                        f"{setter}: not in EVIDENCE_FIELD_MAP, so nothing "
                        f"says whether the run verified it"
                    )
                    continue
                field = EVIDENCE_FIELD_MAP[setter]
                if field is None:
                    continue
                if field not in run:
                    mismatches.append(
                        f"{setter}: record has no '{field}' field"
                    )
                    continue
                for literal in called_with_all(src, setter):
                    # A `{{ placeholder }}` is CONTEXT-derived -- the SDK
                    # computes validation_split from the dataset, the picker
                    # supplies sequence_length -- so one run's value cannot
                    # confirm or refute it as a pre-fill. Only literals the
                    # template fixes are the template's claim to verify.
                    if PLACEHOLDER.search(literal):
                        continue
                    if not _literal_matches(setter, literal, run[field]):
                        mismatches.append(
                            f"{setter}({literal}) vs recorded "
                            f"{field}={run[field]!r}"
                        )
        if not run.get("experiment"):
            mismatches.append("record names no experiment")
        verified = not mismatches
        declared = fam.get("unverified")
        if verified and declared:
            errors.append(
                f"{key}: marked unverified ('{declared}') but "
                f"verification-dev.json has a COMPLETED run "
                f"({run.get('experiment')}) matching its pre-fills. Drop the "
                f"marker."
            )
        elif not verified and not declared:
            why = "no record at all" if not run else (
                "; ".join(mismatches[:6]) + (
                    f" (+{len(mismatches) - 6} more)" if len(mismatches) > 6 else ""
                )
            ) if mismatches else (
                f"recorded run {run.get('experiment')} is "
                f"{run.get('status')} at cycles={run.get('cycles')}, "
                f"epochs={run.get('epochs')}"
            )
            errors.append(
                f"{key}: pre-fills cycles={fam['cycles']}, "
                f"epochs={fam['epochs']} are not backed by a COMPLETED "
                f"full-settings-cell run ({why}). Re-run on dev and refresh "
                f"verification-dev.json, or mark the family `unverified` with "
                f"the ticket that explains why."
            )

    # Rule 4, other half — nothing left unclaimed.
    missing = ALL_CATEGORIES - set(seen_categories)
    if missing:
        errors.append(
            f"categories with no family template: {sorted(missing)}"
        )

    for n in notes:
        print(f"note: {n}")
    if notes:
        print()

    if errors:
        print(f"{len(errors)} problem(s):\n")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(
        f"OK — {len(families)} families, "
        f"{len(seen_categories)} categories, all D9 rules hold."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
