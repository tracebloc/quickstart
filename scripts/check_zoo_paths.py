#!/usr/bin/env python3
"""Every model-zoo path this repo names must exist in the model zoo.

Run from the repo root, with a `model-zoo` checkout beside it — the same
place the guide's own clone cell puts one:

    python3 scripts/check_zoo_paths.py [--zoo ../model-zoo]

Its mutation harness is `scripts/check_zoo_paths_mutations.py`.

Why this exists
---------------

quickstart#11: the guide's upload cell shipped

    MODEL_PATH = ".../image_classification/pytorch/densenet.py"

and the zoo had not contained `densenet.py` for some time. A peer running
the notebook top to bottom — the flow the "Run All" wording invites — got
a `FileNotFoundError` on a line the guide itself supplied.

Fixing that one path fixes one instance. **Nothing bound the path to the
zoo**, so the same edit rots again the next time the zoo is curated, and
nobody finds out until a peer does. Two repos, one contract, and no check
across it: that is the class, and this script is the fix for the class.

What is checked
---------------

Every `model_zoo/...` path appearing anywhere in `notebooks/` (both
markdown and code cells, plus the templates) or in `README.md` must
resolve inside the zoo checkout:

* a path ending in `/` must be a DIRECTORY,
* a path ending in a file extension must be a FILE,
* a bare path must be either.

The distinction matters: `model_zoo/image_classification/` resolves as a
directory whatever is inside it, which is exactly how the guide's table
managed to advertise a TensorFlow family that had been retired. The table
now names a directory per FRAMEWORK, so the framework claim is a path and
this check covers it.

What is NOT checked, on purpose
-------------------------------

The templates' `{{ model_path }}` placeholders. Those are filled by the
renderer at serve time, not by this repo, and a template carrying an
unsubstituted placeholder is already rule 10 in
`scripts/check_templates.py`.

Missing checkout is a FAILURE, not a skip
-----------------------------------------

This script refuses to pass when it cannot find the zoo. A check that
quietly skips is indistinguishable from one that always passes — the
exact defect `.github/workflows/template-rules.yml` was created to end
after nothing at all executed `check_templates.py`. The mutation harness
registers that behaviour as a case, so the refusal is itself seen to
fail rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ZOO = os.path.join(os.path.dirname(ROOT), "model-zoo")

#: Where a zoo path may be named. Anything a reader can copy out of.
SEARCH_ROOTS = ("notebooks", "README.md")

#: A zoo path as it appears in prose or in code: the `model_zoo/` segment
#: and everything path-ish after it. Deliberately anchored on the segment
#: rather than on the `../model-zoo/` prefix, because the guide writes it
#: both ways — `model_zoo/image_classification/pytorch/` in the table and
#: `../model-zoo/model_zoo/.../resnet_18.py` in the upload cell — and a
#: pattern that only knew one of them would have missed the defect that
#: prompted this script.
ZOO_PATH = re.compile(r"model_zoo/[A-Za-z0-9_./-]*")

#: Trailing characters the surrounding prose contributes: a backtick, a
#: closing quote, a comma, a full stop. Stripped from the right so
#: "`model_zoo/x/`." does not become a lookup for a directory named `.`.
TRAILING_PUNCTUATION = "`'\",.;:)]}"

FILE_EXTENSIONS = (".py", ".json", ".md", ".txt", ".pkl", ".yaml", ".yml")


def _iter_text(path: str):
    """Yield (label, text) for every place a zoo path could hide in ``path``."""
    if path.endswith(".ipynb"):
        with open(path, encoding="utf-8") as handle:
            notebook = json.load(handle)
        for index, cell in enumerate(notebook.get("cells", [])):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            yield f"{path} cell {index} ({cell.get('cell_type')})", source
        return
    with open(path, encoding="utf-8") as handle:
        yield path, handle.read()


def _iter_files(root: str = ROOT):
    """Every file under ``root`` that could name a zoo path.

    ``root`` is a parameter and not the module constant because the
    mutation harness runs the checker against a mutated COPY of the tree. The
    first version of this function read ``ROOT`` directly while
    ``check()`` accepted a ``root``: every mutation then landed in the
    copy and the checker read the real tree, so four of the five
    mutations came back "MISSED" — the harness caught its own harness.
    """
    for search_root in SEARCH_ROOTS:
        target = os.path.join(root, search_root)
        if os.path.isfile(target):
            yield target
            continue
        for directory, _, filenames in os.walk(target):
            for filename in sorted(filenames):
                if filename.endswith((".ipynb", ".md", ".json")):
                    yield os.path.join(directory, filename)


def collect_references(root: str = ROOT):
    """Every (label, zoo path) this repo names, in file order."""
    references = []
    for path in _iter_files(root):
        for label, text in _iter_text(path):
            for match in ZOO_PATH.finditer(text):
                reference = match.group(0).rstrip(TRAILING_PUNCTUATION)
                # `model_zoo/` alone is the `!ls` cell listing the zoo's
                # own top level, not a claim about any one family.
                if reference.strip("/") == "model_zoo":
                    continue
                references.append((label.replace(root + os.sep, ""), reference))
    return references


def check(zoo: str, root: str = ROOT) -> list:
    """Return a list of human-readable violations (empty when clean)."""
    violations = []
    zoo_inner = os.path.join(zoo, "model_zoo")
    if not os.path.isdir(zoo_inner):
        return [
            f"no model zoo at {zoo!r} (expected a checkout containing "
            f"model_zoo/). This is a FAILURE, not a skip: without the zoo "
            f"this script can only report that it checked nothing. Clone it "
            f"with: git clone https://github.com/tracebloc/model-zoo.git {zoo}"
        ]

    references = collect_references(root)
    if not references:
        return [
            "found no model_zoo/... references at all. Either the notebooks "
            "stopped naming zoo paths — in which case delete this check — or "
            "ZOO_PATH no longer matches how they are written, in which case "
            "the check is silently vacuous."
        ]

    for label, reference in references:
        resolved = os.path.join(zoo, reference)
        wants_directory = reference.endswith("/")
        wants_file = reference.endswith(FILE_EXTENSIONS)
        if wants_directory and not os.path.isdir(resolved):
            violations.append(f"{label}: {reference} is not a directory in the zoo")
        elif wants_file and not os.path.isfile(resolved):
            violations.append(f"{label}: {reference} is not a file in the zoo")
        elif not wants_directory and not wants_file and not os.path.exists(resolved):
            violations.append(f"{label}: {reference} does not exist in the zoo")
    return violations


def _clone_zoo(destination: str) -> str:
    print(f"cloning the model zoo into {destination} ...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/tracebloc/model-zoo.git",
            destination,
        ],
        check=True,
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zoo",
        default=os.environ.get("MODEL_ZOO_PATH", DEFAULT_ZOO),
        help="path to a model-zoo checkout (default: ../model-zoo)",
    )
    parser.add_argument(
        "--clone",
        action="store_true",
        help="clone the zoo's DEFAULT branch if it is not already there — the "
        "same ref the guide's own `git clone` cell gets, so this check "
        "follows that cell rather than a branch name written down here",
    )
    arguments = parser.parse_args()

    zoo = arguments.zoo
    if arguments.clone and not os.path.isdir(os.path.join(zoo, "model_zoo")):
        _clone_zoo(zoo)

    violations = check(zoo)
    if violations:
        print("model-zoo path check FAILED:\n")
        for violation in violations:
            print(f"  {violation}")
        print(
            "\nEvery model_zoo/... path this repo prints is a promise to a "
            "peer running the notebook top to bottom. Point it at something "
            "the zoo ships, or remove it."
        )
        return 1

    references = collect_references()
    print(
        f"model-zoo paths OK — {len(references)} references, all resolved "
        f"against {zoo}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
