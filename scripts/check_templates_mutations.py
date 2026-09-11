#!/usr/bin/env python3
"""Mutation harness for `check_templates.py`: every rule, seen to fail.

Run from the repo root:

    python3 scripts/check_templates_mutations.py

Each mutation copies the templates and the checker into a temp dir, applies one
change that realises a WRONG ANSWER, runs the real checker as a subprocess, and
asserts it exits non-zero — and, where the mutation is mode- or
identity-specific, that the message names the right thing rather than merely
that something fired.

## Why this file exists

Until now the suite was PROSE. Review notes said "all twelve spot-checked rules
fire", with no artefact anyone could run. That is the same defect the checker
itself exists to catch, one level up: a claim about enforcement with nothing
executing it.

The concrete cost, on this very code: a slice-based edit while rewriting
rule 14b **deleted rule 16 outright**. Nothing noticed, because only the
mutations for the rules being touched were re-run. It surfaced by luck — an
unrelated rule-16 mutation happened to come back green during a later spot
check. A committed harness turns that luck into a failing test, which is the
whole argument for this file (start-training#91, review).

Two disciplines it encodes, both learned the hard way here:

* **A mutation must sit where the code path it tests actually differs.** Rule
  12's mutations once lived in the vision template, which has no flag-gated
  fragments — so its flags-off and flags-on renders are byte-identical and no
  mutation there could tell the two modes apart. The mode-specific ones live in
  `tabular_timeseries` for that reason.
* **Assert the identity or count of what fires, not that something fired.** A
  flags-off break must be reported in BOTH modes; a flags-on break in one. Had
  these asserted only "non-zero", a single-mode loop would still have passed.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The checker this harness exercises. Imported for its PREDICATES (see
#: `_checker`) as well as run as a subprocess for its verdict -- one definition of
#: "which cell is the settings cell", not two.
CHECKER = os.path.join(ROOT, "scripts", "check_templates.py")
TEMPLATES = os.path.join("notebooks", "templates")


# --- mutation helpers ------------------------------------------------------

def _load(work, name):
    with open(os.path.join(work, TEMPLATES, name)) as fh:
        return json.load(fh)


def _save(work, name, doc):
    with open(os.path.join(work, TEMPLATES, name), "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")


def _sub(work, name, old, new):
    """Replace `old` with `new` in every source line of a notebook."""
    nb = _load(work, name)
    hit = False
    for cell in nb["cells"]:
        fixed = [line.replace(old, new) for line in cell["source"]]
        hit = hit or fixed != cell["source"]
        cell["source"] = fixed
    if not hit:
        raise AssertionError(f"mutation target not found in {name}: {old!r}")
    _save(work, name, nb)


@functools.lru_cache(maxsize=1)
def _checker():
    """The checker module itself, so its predicates are not re-implemented here.

    This harness ran the checker as a SUBPROCESS and separately re-implemented one of
    its predicates, which is the fake-proof generator Bugbot named: `_settings_cell`
    keyed on `training.experiment_name` appearing in any code cell, while the checker
    requires `SETTINGS_MARKER`. Two answers to "which cell is the settings cell" means
    a mutation can edit a DIFFERENT cell than the rule under test reads, and then pass
    for the wrong reason.

    Not hypothetical: the two already disagree on shipped content --
    `traceblocTrainingGuide.ipynb` cell 17 mentions that setter and is NOT a settings
    cell, so the legacy predicate matches it and `is_settings_cell` does not. The six
    family templates agree today, which is exactly why this was invisible.
    """
    spec = importlib.util.spec_from_file_location("_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_checker"] = module
    spec.loader.exec_module(module)
    return module


def _settings_cell(nb):
    checker = _checker()
    for cell in nb["cells"]:
        if checker.is_settings_cell(cell):
            return cell
    raise AssertionError("no settings cell")


def _append_cell(work, name, source, meta=None):
    nb = _load(work, name)
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {"tracebloc": meta} if meta else {},
            "execution_count": None,
            "outputs": [],
            "source": source,
        }
    )
    _save(work, name, nb)


def _table(work, fn):
    p = os.path.join(work, TEMPLATES, "families.json")
    with open(p) as fh:
        doc = json.load(fh)
    fn(doc)
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def _fam(doc, key):
    return next(f for f in doc["families"] if f["key"] == key)


# --- the mutations ---------------------------------------------------------
# (rule, description, mutate, expect_in_output)
#
# `expect_in_output` is a substring the failure MUST name. It is not decoration:
# asserting only "exited non-zero" would let a mutation pass because some other
# rule happened to fire, which is how a deleted rule hides.

MUTATIONS = [
    ("1", "cycles*epochs over the 20-pass ceiling",
     lambda w: _table(w, lambda d: _fam(d, "vision_from_scratch").__setitem__("cycles", 21)),
     "> 20"),
    ("2", "epochs>1 with a non-drift-correcting strategy",
     lambda w: _table(w, lambda d: _fam(d, "nlp_finetune").__setitem__("epochs", 2)),
     "no drift correction"),
    ("3a", "notebook cycles edited, table not",
     lambda w: _sub(w, "tabular_timeseries.ipynb", "training.cycles(15)", "training.cycles(3)"),
     "families.json says"),
    ("3b", "the pre-fill set TWICE (Python applies the last)",
     lambda w: _append_to_settings(w, "vision_from_scratch.ipynb", "\ntraining.cycles(99)"),
     "times"),
    ("4", "a category with no family template",
     lambda w: _table(w, lambda d: _fam(d, "vision_from_scratch")["categories"].remove("keypoint_detection")),
     "no family template"),
    ("5", "a template that calls start()",
     lambda w: _append_cell(w, "survival.ipynb", ["training.start()"]),
     "Start is a button"),
    ("6", "a dead setter reintroduced",
     lambda w: _append_to_settings(w, "vision_from_scratch.ipynb", "\ntraining.horizontal_flip(True)"),
     "dead setter"),
    ("7", "a Colab-specific line",
     lambda w: _append_cell(w, "embeddings.ipynb", ["# https://colab.research.google.com/x"]),
     "Colab-specific"),
    ("8a", "a cell gated on a category outside its family",
     lambda w: _append_cell(w, "nlp_finetune.ipynb", ["training.seed(0)"],
                            {"applies_to": {"category": ["image_classification"]}, "settings_fragment": True}),
     "not in this family"),
    ("8b", "a cell gated on a framework the family does not ship",
     lambda w: _append_cell(w, "nlp_finetune.ipynb", ["training.seed(0)"],
                            {"applies_to": {"framework": ["sklearn"]}, "settings_fragment": True}),
     "not in this family"),
    ("9a", "a settings_fragment whose applies_to is {} (gates nothing)",
     lambda w: _append_cell(w, "vision_from_scratch.ipynb", ['training.optimizer("adam")'],
                            {"applies_to": {}, "settings_fragment": True}),
     "gates nothing"),
    ("9c", "a gate key present with an EMPTY list — renders for no pair",
     lambda w: _empty_gate(w),
     "EMPTY"),
    # The discriminator for the SHARED settings-cell predicate. A decoy cell that
    # merely mentions the setter must not become the mutation target; under the legacy
    # predicate it did, and this case went quiet (Bugbot).
    ("3i", "a decoy cell mentioning the setter, then the real pre-fill edited",
     lambda w: _decoy_then_append(w, "vision_from_scratch.ipynb", "\ntraining.cycles(99)"),
     "times"),
    ("3c", "a second cycles() written in a form the narrow pattern missed",
     lambda w: _append_to_settings(w, "vision_from_scratch.ipynb", "\ntraining . cycles (99)"),
     "times"),
    ("3d", "the real cycles() COMMENTED OUT, comment left in place",
     lambda w: _sub(w, "vision_from_scratch.ipynb",
                    "training.cycles(20)", "# training.cycles(20)"),
     # Rule 3 is per-pair now, so a commented-out pre-fill reads as ZERO calls
     # for that pair rather than "never calls" for the file. Updated to the
     # message the rule actually emits, not loosened to match anything.
     "training.cycles() 0 times"),
    ("3f", "a family GAINS a framework with its own gated federation fragment",
     lambda w: _add_framework(w),
     None),  # must PASS: the old whole-file rule 3 blocked this outright
    # SYNTHETIC id on purpose: test data should not carry a real private
    # tracker reference just to prove the rule fires.
    ("17a", "a cross-repo reference in a peer-rendered cell",
     lambda w: _append_cell(w, "embeddings.ipynb", ["# see example-repo#4242 for context"]),
     "internal reference"),
    ("17b", "an RFC id in a peer-rendered cell",
     lambda w: _append_cell(w, "embeddings.ipynb", ["# per RFC-9999"]),
     "internal reference"),
    ("17c", "a non-public internal host",
     lambda w: _append_cell(w, "embeddings.ipynb", ["# https://internal.tracebloc.io/x"]),
     "internal reference"),
    ("17d", "the PUBLIC docs host must NOT be flagged",
     lambda w: _append_cell(w, "embeddings.ipynb", ["# https://docs.tracebloc.io/x"]),
     None),
    # 17e/17f pin the ALLOWED_DOC_URL strip that lets the real anchors through.
    # 17e is the case the strip exists for and 17d does not cover: a docs URL
    # whose fragment is a bare NUMBER (`#1-optimizer`), which INTERNAL_REF's
    # `<repo>#N` branch would otherwise read as a cross-reference. Delete the
    # strip and this legitimate link turns red — every template carries these.
    ("17e", "a docs URL with a NUMERIC fragment must NOT be flagged",
     lambda w: _append_cell(w, "embeddings.ipynb",
                            ["# https://docs.tracebloc.io/join-use-case/hyperparameters#1-optimizer"]),
     None),
    # 17f is the other edge: the strip must remove ONLY the allowed URL, not
    # swallow a real `<repo>#N` sitting beside one. A too-greedy strip would
    # pass this tree and let an actual leak render to a peer.
    ("17f", "a <repo>#N beside a docs URL is still flagged",
     lambda w: _append_cell(w, "embeddings.ipynb",
                            ["# https://docs.tracebloc.io/join-use-case/hyperparameters#1-optimizer"
                             " — but example-repo#4242 is internal"]),
     "internal reference"),
    ("8d-a", "metadata.tracebloc is a bool on a FRAGMENT cell",
     lambda w: _nondict_tracebloc(w, fragment=True),
     "not an object"),
    ("8d-b", "metadata.tracebloc is a bool on a PLAIN code cell",
     lambda w: _nondict_tracebloc(w, fragment=False),
     "not an object"),
    ("8e", "an unknown applies_to key ALONGSIDE a valid one",
     lambda w: _typo_alongside(w),
     "applies_to keys"),
    ("3g", "aggregation_strategy flipped in the TABLE only",
     lambda w: _table(w, lambda d: _fam(d, "tabular_timeseries").__setitem__("aggregation_strategy", "fedprox")),
     "aggregation_strategy"),
    ("3h", "cycles(99) hidden in a FLAG-gated fragment (flags-on only)",
     lambda w: _append_cell(w, "nlp_finetune.ipynb", ["training.cycles(99)"],
                            {"applies_to": {"dataset_flag": "allow_feature_modification"},
                             "settings_fragment": True}),
     "flags on"),
    ("3e", "a SECOND settings cell inserted BEFORE the real one",
     lambda w: _second_settings_cell(w),
     "settings cells"),
    ("9d", "a gate that admits every render the family produces",
     lambda w: _append_cell(w, "vision_from_scratch.ipynb", ["training.seed(0)"],
                            {"applies_to": {"framework": ["pytorch"]}, "settings_fragment": True}),
     "admits EVERY render"),
    # `expect=None` — must STAY OK. The same gate PLUS a dataset_flag, which render()
    # drops on the flags-off pass, so it is not a no-op. Rule 9d walked only
    # (category, framework) and called this dead (Bugbot), which is the defect rule 9d
    # was itself created to fix, one axis along.
    ("9d-b", "an all-framework gate that ALSO sets dataset_flag — render drops it",
     lambda w: _append_cell(w, "vision_from_scratch.ipynb", ["training.seed(0)"],
                            {"applies_to": {"framework": ["pytorch"],
                                            "dataset_flag": "allow_feature_modification"},
                             "settings_fragment": True}),
     None),
    ("8f-a", "a gate axis that is null (used to TypeError out of gate_audience)",
     lambda w: _malformed_axis(w, None),
     "not list"),
    ("8f-b", "a gate axis that is a bare STRING (used to iterate its characters)",
     lambda w: _malformed_axis(w, "vision"),
     "not list"),
    ("8f-c", "a null dataset_flag (its shape was unpinned by 8f-a/8f-b alone)",
     lambda w: _malformed_axis(w, None, axis="dataset_flag"),
     "not str"),
    ("8f-d", "a framework axis that is a bare STRING",
     lambda w: _malformed_axis(w, "pytorch", axis="framework"),
     "not list"),
    ("8c", "applies_to that is a LIST, not an object (used to AttributeError)",
     lambda w: _nondict_applies(w),
     "not an object"),
    ("15b", "a loss fragment gated on category: [] — offered to NOBODY",
     lambda w: _empty_gated_loss(w),
     "custom loss"),
    ("13e", "a recorded preprocessing pre-fill changed (handle_missing_values)",
     lambda w: _sub(w, "tabular_timeseries.ipynb",
                    "training.handle_missing_values(True)",
                    "training.handle_missing_values(False)"),
     "not backed by a COMPLETED"),
    ("13f", "a recorded preprocessing pre-fill changed (encoding_strategy)",
     lambda w: _sub(w, "tabular_timeseries.ipynb",
                    'training.encoding_strategy("label")',
                    'training.encoding_strategy("onehot")'),
     "not backed by a COMPLETED"),
    # NO 13g for `scaler`. Deliberately absent, with the reason, because a
    # mutation that cannot realise a wrong answer is worse than none:
    #   * survival sets a LITERAL scaler, but survival is declared
    #     `unverified` in the table, so rule 13 correctly reports nothing
    #     for it and the mutation would always "pass";
    #   * tabular_timeseries sets `{{ scaler }}`, a context-derived
    #     placeholder rule 13 skips by design.
    # So the `scaler -> tabular_scaler` map entry is currently unexercisable.
    # It is kept because it is correct the moment survival gains a COMPLETED
    # run, and this comment is here so nobody reads its absence as coverage.
    ("9b", "a typo'd dataset_flag",
     lambda w: _retag_flag(w, "tabular_timeseries.ipynb", "allow_feature_modificaton"),
     "not a flag the platform has"),
    ("10", "a half-written placeholder, as an f-string produces",
     lambda w: _sub(w, "embeddings.ipynb", "{{ sequence_length }}", "{ sequence_length }"),
     "half-written placeholder"),
    ("11", "a setter offered to a category that refuses it",
     lambda w: _append_to_settings(w, "vision_from_scratch.ipynb", "\ntraining.enable_lora(True)"),
     "which refuse it"),
    ("12a", "a break reachable only with dataset flags OFF",
     lambda w: _sub(w, "tabular_timeseries.ipynb", 'training.optimizer("sgd")', 'training.optimizer("sgd"'),
     "flags off"),
    ("12b", "a break reachable only with dataset flags ON",
     lambda w: _break_flag_fragment(w),
     "flags on"),
    ("13a", "evidence: callbacks blanked",
     lambda w: _evidence(w, lambda d: d["runs"]["vision"]["rounds"]["2_full_settings_cell"].__setitem__("callbacks", "[]")),
     "not backed by a COMPLETED"),
    ("13b", "evidence: the experiment id removed",
     lambda w: _evidence(w, lambda d: d["runs"]["vision"]["rounds"]["2_full_settings_cell"].pop("experiment")),
     "not backed by a COMPLETED"),
    ("13c", "a pre-fill changed in BOTH table and notebook, evidence untouched",
     lambda w: (_table(w, lambda d: _fam(d, "vision_from_scratch").__setitem__("aggregation_strategy", "fedprox")),
                _sub(w, "vision_from_scratch.ipynb", 'aggregation_strategy("fedavg")', 'aggregation_strategy("fedprox")')),
     "not backed by a COMPLETED"),
    ("13d", "optimizer changed in the settings cell only",
     lambda w: _sub(w, "vision_from_scratch.ipynb", 'training.optimizer("sgd")', 'training.optimizer("adamw")'),
     "not backed by a COMPLETED"),
    # The two forms the discovery scan and `called_with_all` used to disagree about
    # (@LukasWodka in review). Both mutate the SETTINGS CELL, so if rule 13 stops
    # comparing a live setter to the record, or starts comparing a commented one it
    # cannot read, exactly one of these goes quiet.
    ("13x-a", "a live setter written with spaces around the dot — Start applies it",
     lambda w: _sub(w, "vision_from_scratch.ipynb", 'training.optimizer("sgd")',
                    'training . optimizer("adamw")'),
     "not backed by a COMPLETED"),
    # `expect=None` — this one asserts the checker STAYS OK, like 17d.
    #
    # An UNMAPPED setter, commented out. The discovery scan used to read the RAW source,
    # so it found this comment, looked the name up in EVIDENCE_FIELD_MAP, missed, and
    # reported "not in EVIDENCE_FIELD_MAP, so nothing says whether the run verified it"
    # about a line Python never executes. Scanning `code_only` there makes it invisible,
    # which is correct: rule 13 asks whether a pre-fill is APPLIED.
    #
    # An earlier version of this case commented out a MAPPED setter and also changed the
    # live value, and it was a FAKE PROOF — caught by the value mismatch even with the
    # old two-pattern code, so it said nothing about the comment behaviour. Verified by
    # reverting: this form goes from a spurious failure to OK, the old form was caught
    # either way.
    ("13x-b", "an UNMAPPED setter, commented out — must not be read as a live call",
     lambda w: _sub(w, "vision_from_scratch.ipynb", 'training.optimizer("sgd")',
                    'training.optimizer("sgd")\n    # training.warmup_ratio(0.1)'),
     None),
    ("14", "cycles offered to a single-pass framework",
     lambda w: _widen_gate(w),
     "force both to 1"),
    ("15", "the custom-loss offering removed from a family that accepts one",
     lambda w: _strip_loss(w, "survival.ipynb"),
     "custom loss"),
    ("14b-a", "the SDK version floor removed from families.json",
     lambda w: _table(w, lambda d: d.pop("sdk_version_floor")),
     "no `sdk_version_floor`"),
    # Needs an importable SDK: with none, rule 14b legitimately skips the
    # cross-check, so drift is undetectable and this mutation would report a
    # false NOT CAUGHT. Marked NEEDS_SDK rather than dropped, and SKIPPED
    # LOUDLY when unavailable — a mutation quietly not run is the same defect
    # as a rule quietly not checked. CI installs the SDK, so it runs there.
    ("14b-b!", "single_pass_frameworks drifted from the SDK's own set",
     lambda w: _table(w, lambda d: d.__setitem__("single_pass_frameworks", ["sklearn"])),
     "has _SURVIVAL_FRAMEWORKS"),
    ("16", "a setter in a standalone cell, inert under Start",
     lambda w: _append_cell(w, "embeddings.ipynb", ['training.optimizer("adam")']),
     "outside the settings cell"),
]


def _decoy_then_append(work, name, text):
    """Insert a decoy cell that MENTIONS the settings setter, then mutate the real one.

    The discriminator for using the checker's own `is_settings_cell`. Under the legacy
    predicate ("`training.experiment_name` in any code cell") the decoy is found FIRST,
    so `_append_to_settings` edits a cell the rule under test never reads and the
    mutation goes quiet -- a mutation that cannot realise a wrong answer. Under the
    shared predicate the decoy has no SETTINGS_MARKER, so the real settings cell is
    still the target.
    """
    nb = _load(work, name)
    settings_at = next(
        i for i, c in enumerate(nb["cells"]) if _checker().is_settings_cell(c)
    )
    nb["cells"].insert(
        settings_at,
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["# prose about training.experiment_name, not a settings cell\n"],
        },
    )
    _save(work, name, nb)
    _append_to_settings(work, name, text)


def _append_to_settings(work, name, text):
    nb = _load(work, name)
    _settings_cell(nb)["source"].append(text)
    _save(work, name, nb)


def _evidence(work, fn):
    p = os.path.join(work, TEMPLATES, "verification-dev.json")
    with open(p) as fh:
        doc = json.load(fh)
    fn(doc)
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2)


def _retag_flag(work, name, value):
    nb = _load(work, name)
    for cell in nb["cells"]:
        applies = ((cell.get("metadata") or {}).get("tracebloc") or {}).get("applies_to") or {}
        if "dataset_flag" in applies:
            applies["dataset_flag"] = value
            _save(work, name, nb)
            return
    raise AssertionError("no dataset_flag gate to retag")


def _break_flag_fragment(work):
    nb = _load(work, "tabular_timeseries.ipynb")
    for cell in nb["cells"]:
        applies = ((cell.get("metadata") or {}).get("tracebloc") or {}).get("applies_to") or {}
        if "dataset_flag" in applies:
            cell["source"].append("\ntraining.seed(0")
            _save(work, "tabular_timeseries.ipynb", nb)
            return
    raise AssertionError("no flag-gated fragment to break")


def _add_framework(work):
    """Give tabular a second framework with its own federation fragment.

    Expected to PASS. The whole-file rule 3 counted one `cycles` call per
    framework-gated fragment and failed, so the rule BLOCKED a family from
    ever gaining a framework — a check preventing a correct change. Kept as a
    NEGATIVE mutation so the restructure cannot regress.
    """
    _table(work, lambda d: _fam(d, "tabular_timeseries")["frameworks"].append("xgboost"))
    _append_cell(
        work, "tabular_timeseries.ipynb",
        ["training.cycles(15)\n", "training.epochs(1)"],
        {"applies_to": {"framework": ["xgboost"]}, "settings_fragment": True},
    )


def _second_settings_cell(work):
    nb = _load(work, "vision_from_scratch.ipynb")
    marker = (
        "# ======================================================================\n"
        "# Settings \u2014 the complete plan for this run, as plain SDK calls.\n"
    )
    nb["cells"].insert(3, {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": [marker, 'training.experiment_name("x")\n',
                                  "training.cycles(99)"],
    })
    _save(work, "vision_from_scratch.ipynb", nb)


def _malformed_axis(work, value, axis="category"):
    """Put a wrong-typed value on a gate axis, on a real template cell.

    `category: null` used to raise TypeError out of `gate_audience` and rule 9d --
    a crash instead of a rule error, which the checker's own comment says its shared
    helpers exist to prevent. `category: "vision"` was worse: a bare string ITERATES,
    so the audience became ['v','i','s','i','o','n'] and the cell read as offered to
    categories that do not exist -- a silent wrong answer (Bugbot reported the null;
    the string turned up while reproducing it).

    PARAMETRISED ON THE AXIS, because the first version of this helper hardcoded
    `category` and both its cases varied only that one. @LukasWodka mutation-proved the
    gap: change `"dataset_flag": str` to `object` in AXIS_VALUE_TYPES and the clean tree
    still passes with both category cases green, so `dataset_flag: null` was a silent OK
    again -- the exact false green this fix removes. An instrument has to vary along
    every axis it claims to cover.
    """
    nb = _load(work, "vision_from_scratch.ipynb")
    cell = nb["cells"][0]
    cell.setdefault("metadata", {}).setdefault("tracebloc", {})
    cell["metadata"]["tracebloc"]["applies_to"] = {axis: value}
    _save(work, "vision_from_scratch.ipynb", nb)


def _nondict_tracebloc(work, fragment):
    """Set metadata.tracebloc to a bool, on a fragment or a plain cell.

    Both placements matter: `settings_source()` reached it on one and the
    rules 8/9 loop on the other, and the loop's falsy-guard meant a fixed
    accessor would have turned the traceback into a SILENTLY UNGATED cell.
    """
    nb = _load(work, "tabular_timeseries.ipynb")
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tb = (cell.get("metadata") or {}).get("tracebloc") or {}
        if bool(tb.get("settings_fragment")) == fragment:
            cell.setdefault("metadata", {})["tracebloc"] = True
            _save(work, "tabular_timeseries.ipynb", nb)
            return
    raise AssertionError(f"no {'fragment' if fragment else 'plain'} cell found")


def _typo_alongside(work):
    nb = _load(work, "tabular_timeseries.ipynb")
    for cell in nb["cells"]:
        applies = ((cell.get("metadata") or {}).get("tracebloc") or {}).get("applies_to")
        if isinstance(applies, dict) and "framework" in applies:
            applies["categor"] = ["tabular_classification"]
            _save(work, "tabular_timeseries.ipynb", nb)
            return
    raise AssertionError("no framework gate to add a typo beside")


def _nondict_applies(work):
    nb = _load(work, "tabular_timeseries.ipynb")
    for cell in nb["cells"]:
        tb = (cell.get("metadata") or {}).get("tracebloc") or {}
        if "applies_to" in tb:
            tb["applies_to"] = ["pytorch"]
            _save(work, "tabular_timeseries.ipynb", nb)
            return
    raise AssertionError("no applies_to to corrupt")


def _empty_gated_loss(work):
    """Strip vision's loss offering and re-add it gated on an EMPTY category
    list. The pod renders that for nobody, so rule 15's coverage must FAIL --
    an earlier fix made the checker read it as offered to the whole family,
    which is the pod's meaning inverted."""
    _strip_loss(work, "vision_from_scratch.ipynb")
    _append_cell(
        work, "vision_from_scratch.ipynb",
        ['# training.loss_function({"type": "custom", "value": "loss.py"})'],
        {"applies_to": {"category": []}, "settings_fragment": True},
    )


def _empty_gate(work):
    nb = _load(work, "tabular_timeseries.ipynb")
    for cell in nb["cells"]:
        applies = ((cell.get("metadata") or {}).get("tracebloc") or {}).get("applies_to") or {}
        if "category" in applies:
            applies["category"] = []
            _save(work, "tabular_timeseries.ipynb", nb)
            return
    raise AssertionError("no category gate to empty")


def _widen_gate(work):
    nb = _load(work, "tabular_timeseries.ipynb")
    for cell in nb["cells"]:
        if "training.cycles(15)" in "".join(cell["source"]):
            cell["metadata"]["tracebloc"]["applies_to"]["framework"] = ["pytorch", "sklearn"]
            _save(work, "tabular_timeseries.ipynb", nb)
            return
    raise AssertionError("no framework-gated federation fragment")


def _strip_loss(work, name):
    nb = _load(work, name)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "training.loss_function" in src:
            head = src.partition("# --- Custom loss")[0]
            cell["source"] = [f"{line}\n" for line in head.rstrip().split("\n")]
            _save(work, name, nb)
            return
    raise AssertionError("no custom-loss offering to strip")


# --- runner ----------------------------------------------------------------

SDK_AVAILABLE = (
    subprocess.run(
        [sys.executable, "-c", "import tracebloc.training.plan"],
        capture_output=True,
    ).returncode
    == 0
)


def run_checker(work, strict=False):
    env = dict(os.environ)
    if strict:
        env["TRACEBLOC_CHECK_STRICT"] = "1"
    else:
        env.pop("TRACEBLOC_CHECK_STRICT", None)
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "check_templates.py")],
        cwd=work, capture_output=True, text=True, env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    failures = []

    with tempfile.TemporaryDirectory() as base:
        clean = os.path.join(base, "clean")
        shutil.copytree(os.path.join(ROOT, TEMPLATES), os.path.join(clean, TEMPLATES))
        shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(clean, "scripts"))

        # Control: the tree as committed must PASS. A suite whose baseline is
        # already red proves nothing about any mutation.
        code, out = run_checker(clean)
        if code != 0:
            print("CONTROL FAILED — the committed tree does not pass:\n" + out)
            return 1
        print(f"control: clean tree passes (exit {code})")

        skipped = []
        for rule, desc, mutate, expect in MUTATIONS:
            needs_sdk = rule.endswith("!")
            if needs_sdk and not SDK_AVAILABLE:
                skipped.append(f"rule {rule}: {desc} (needs an importable SDK)")
                print(f"  rule {rule:<4} SKIPPED     {desc} — no importable SDK")
                continue
            work = os.path.join(base, f"m{rule.rstrip('!')}")
            shutil.copytree(clean, work)
            try:
                mutate(work)
            except AssertionError as exc:
                failures.append(f"rule {rule}: mutation could not be applied — {exc}")
                print(f"  rule {rule:<4} UNAPPLIED  {desc}")
                continue
            code, out = run_checker(work, strict=needs_sdk)
            if expect is None:
                # A NEGATIVE mutation: a legitimate change the checker must
                # ACCEPT. Without these, tightening a rule until it rejects
                # everything looks like progress.
                if code == 0:
                    print(f"  rule {rule:<4} accepted    {desc}")
                else:
                    failures.append(
                        f"rule {rule}: checker REJECTED a legitimate change "
                        f"({desc})\n{out.strip()[:400]}"
                    )
                    print(f"  rule {rule:<4} FALSE FAIL  {desc}")
                continue
            if code == 0:
                failures.append(
                    f"rule {rule}: checker PASSED a tree mutated to be wrong "
                    f"({desc})"
                )
                print(f"  rule {rule:<4} NOT CAUGHT  {desc}")
            elif expect not in out:
                failures.append(
                    f"rule {rule}: checker failed, but no message named "
                    f"{expect!r} — something else fired ({desc})"
                )
                print(f"  rule {rule:<4} WRONG RULE  {desc}")
            else:
                print(f"  rule {rule:<4} caught      {desc}")

    if skipped:
        print(
            f"\n{len(skipped)} mutation(s) SKIPPED — they need an importable "
            f"tracebloc, which CI installs:"
        )
        for sk in skipped:
            print(f"  - {sk}")

    if failures:
        print(f"\n{len(failures)} mutation(s) not caught:\n")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        f"\nOK — {len(MUTATIONS) - len(skipped)} of {len(MUTATIONS)} mutations "
        f"run, every one caught by the rule that owns it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
