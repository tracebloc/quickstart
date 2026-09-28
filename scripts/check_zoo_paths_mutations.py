#!/usr/bin/env python3
"""Mutation harness for `scripts/check_zoo_paths.py`: every rule, seen to fail.

Run from the repo root, with a `model-zoo` checkout beside it:

    python3 scripts/check_zoo_paths_mutations.py [--zoo ../model-zoo]

Each mutation copies `notebooks/` and `README.md` into a temp dir, applies one
change that realises a WRONG ANSWER, runs the real checker's `check()` on the
copy, and asserts it reports EXACTLY the violation that change should cause --
not merely that something fired. Before it does, it asserts the mutation
landed: the checker's own `collect_references()` must find the planted path in
the copy (or, for the strip case, find nothing), so a mutation that silently
missed reads as a failure here rather than as a rule that still works.

## Why this is its own file

It used to be `check_zoo_paths.py --self-test`, inside the file it tests. A
mutation harness that lives in its checker is invisible to anything that asks
"which file proves this one can fail", and it can be deleted in the same edit
that deletes the rule. `check_templates_mutations.py` already had this shape;
now both checkers do.

## Why exact assertions

The first version asserted only that `check()` returned something. A rule that
started reporting the wrong kind of violation -- a missing FILE read as a
missing DIRECTORY, a README path attributed to a notebook -- would still have
been "caught". Each case below names the one line it expects.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The checker this harness exercises, imported rather than copied: a harness
#: that re-implements the rule goes on passing after the real one breaks.
CHECKER = os.path.join(ROOT, "scripts", "check_zoo_paths.py")

_spec = importlib.util.spec_from_file_location("check_zoo_paths", CHECKER)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

MUTANT_NOTEBOOK = "notebooks/mutant.ipynb"
MUTANT_LABEL = f"{MUTANT_NOTEBOOK} cell 0 (markdown)"
NO_ZOO = "/nonexistent-zoo-checkout"

#: (description, edits, zoo override, the planted reference, the one violation
#: `check()` must return). `edits` maps a path in the copy to the text planted
#: there: a notebook is written as a one-cell notebook, anything else is
#: appended to. `__strip__` empties the copy's notebooks and README instead.
_MUTATIONS = (
    (
        "a model file the zoo does not ship (the quickstart#11 defect itself)",
        {MUTANT_NOTEBOOK: "model_zoo/image_classification/pytorch/densenet.py"},
        None,
        "model_zoo/image_classification/pytorch/densenet.py",
        f"{MUTANT_LABEL}: model_zoo/image_classification/pytorch/densenet.py "
        "is not a file in the zoo",
    ),
    (
        "a directory the zoo does not have (a retired framework)",
        {MUTANT_NOTEBOOK: "`model_zoo/image_classification/tensorflow/`"},
        None,
        "model_zoo/image_classification/tensorflow/",
        f"{MUTANT_LABEL}: model_zoo/image_classification/tensorflow/ "
        "is not a directory in the zoo",
    ),
    (
        "a family that never existed",
        {MUTANT_NOTEBOOK: "`model_zoo/telepathy/pytorch/`"},
        None,
        "model_zoo/telepathy/pytorch/",
        f"{MUTANT_LABEL}: model_zoo/telepathy/pytorch/ is not a directory in the zoo",
    ),
    (
        # The third branch of the rule: no trailing `/` and no file extension,
        # so the path may be either -- and is neither.
        "a bare path that is neither a file nor a directory",
        {MUTANT_NOTEBOOK: "`model_zoo/telepathy`"},
        None,
        "model_zoo/telepathy",
        f"{MUTANT_LABEL}: model_zoo/telepathy does not exist in the zoo",
    ),
    (
        "a zoo path named in README.md rather than in a notebook",
        {"README.md": "see `model_zoo/image_classification/pytorch/densenet.py`"},
        None,
        "model_zoo/image_classification/pytorch/densenet.py",
        "README.md: model_zoo/image_classification/pytorch/densenet.py "
        "is not a file in the zoo",
    ),
    (
        "no zoo checkout at all",
        {},
        NO_ZOO,
        None,
        f"no model zoo at {NO_ZOO!r} (expected a checkout containing "
        f"model_zoo/). This is a FAILURE, not a skip: without the zoo "
        f"this script can only report that it checked nothing. Clone it "
        f"with: git clone https://github.com/tracebloc/model-zoo.git {NO_ZOO}",
    ),
    # Not a bad path but an absent one: the shape where the checker keeps
    # passing while checking nothing. If ZOO_PATH stops matching how the
    # notebooks write these paths, every real reference vanishes and the
    # loop in `check()` runs zero times -- which without this rule reads
    # exactly like a clean tree.
    (
        "a tree that names no zoo paths at all",
        {"__strip__": True},
        None,
        None,
        "found no model_zoo/... references at all. Either the notebooks "
        "stopped naming zoo paths — in which case delete this check — or "
        "ZOO_PATH no longer matches how they are written, in which case "
        "the check is silently vacuous.",
    ),
)


def _write_notebook(target: str, text: str) -> None:
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cells": [{"cell_type": "markdown", "metadata": {}, "source": [text]}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            handle,
        )


def _plant(workspace: str, edits: dict) -> None:
    shutil.copytree(os.path.join(ROOT, "notebooks"), os.path.join(workspace, "notebooks"))
    shutil.copy(os.path.join(ROOT, "README.md"), workspace)
    edits = dict(edits)
    if edits.pop("__strip__", False):
        shutil.rmtree(os.path.join(workspace, "notebooks"))
        os.makedirs(os.path.join(workspace, "notebooks"))
        open(os.path.join(workspace, "README.md"), "w").close()
    for relative_path, text in edits.items():
        target = os.path.join(workspace, relative_path)
        if target.endswith(".ipynb"):
            _write_notebook(target, text)
        else:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write("\n" + text + "\n")


def _anchor_problem(workspace: str, edits: dict, planted) -> str | None:
    """Why the mutation did NOT land, or None when it did."""
    references = checker.collect_references(workspace)
    if edits.get("__strip__"):
        return None if not references else f"the strip left {len(references)} reference(s)"
    if planted is None:
        return None
    if planted not in {reference for _, reference in references}:
        return f"the checker never read the planted {planted!r}"
    return None


def run(zoo: str) -> int:
    print("mutation harness: every rule of check_zoo_paths.py, seen to fail\n")
    failures = []

    # The tree as committed must pass first, or the mutations below prove
    # nothing about the rules -- only that the checker always fails.
    clean = checker.check(zoo)
    print(f"  [{'ok' if not clean else 'BROKEN'}] the committed tree passes")
    if clean:
        failures.append(f"the committed tree does not pass: {clean[0]}")

    for description, edits, zoo_override, planted, expected in _MUTATIONS:
        workspace = tempfile.mkdtemp(prefix="zoo-paths-mutation-")
        try:
            _plant(workspace, edits)
            problem = _anchor_problem(workspace, edits, planted)
            if problem:
                print(f"  [INERT ] {description}\n           {problem}")
                failures.append(f"{description}: the mutation did not apply ({problem})")
                continue
            violations = checker.check(zoo_override or zoo, root=workspace)
            if violations == [expected]:
                print(f"  [caught] {description}\n           {expected}")
            else:
                verdict = "MISSED" if not violations else "WRONG "
                print(f"  [{verdict}] {description}")
                print(f"           expected: {expected}")
                print(f"           got:      {violations}")
                failures.append(description)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    if failures:
        print("\nmutation harness FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nOK — {len(_MUTATIONS)} mutations, each caught with the violation it causes.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zoo",
        default=os.environ.get("MODEL_ZOO_PATH", checker.DEFAULT_ZOO),
        help="path to a model-zoo checkout (default: ../model-zoo)",
    )
    return run(parser.parse_args().zoo)


if __name__ == "__main__":
    sys.exit(main())
