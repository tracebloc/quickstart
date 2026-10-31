#!/usr/bin/env python3
# mutates: scripts/check_mutation_wiring.py
"""Mutation harness for `scripts/check_mutation_wiring.py`: every rule, seen to fail.

Run from the repo root:

    python3 scripts/check_mutation_wiring_mutations.py

Each mutation copies the tracked files the checker reads (`.github/`, `scripts/`,
`Makefile`, `.gitignore`) into a temp dir, applies ONE change that realises a wrong
answer, makes the copy a git checkout, asserts the change landed, and asserts that
`check()` reports exactly the violation that change causes -- not merely that
something fired. The copy as committed must pass first.

The runner cases call the real `run()` and `leg_share()`. A run is only as honest
as its exit code, so a declaration of planted rows must pass when every row passes
and fail, naming the row, when one fails; a declaration with no rows, and a leg
outside its count, must refuse to run at all.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The checker this harness exercises, imported rather than copied: a harness that
#: re-implements the rule goes on passing after the real one breaks.
CHECKER = os.path.join(ROOT, "scripts", "check_mutation_wiring.py")

_spec = importlib.util.spec_from_file_location("check_mutation_wiring", CHECKER)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

DECL = ".github/mutation-coverage.yml"
WEEKLY = ".github/workflows/mutation-nightly.yml"
TEMPLATE_RULES = ".github/workflows/template-rules.yml"
ZOO_PATHS = ".github/workflows/zoo-paths.yml"
ALONE = "mutation rows run in .github/workflows/mutation-nightly.yml alone"
COPIED = (".github/", "scripts/")
COPIED_FILES = ("Makefile", ".gitignore")

GREEN_ROW = "scripts/check_green_mutations.py"
RED_ROW = "scripts/check_red_mutations.py"
EXIT_WITH = 'import sys\n\nif __name__ == "__main__":\n    sys.exit({code})\n'


# --- planting -------------------------------------------------------------


def _path(work, relative):
    return os.path.join(work, relative)


def _text(work, relative):
    with open(_path(work, relative), encoding="utf-8") as handle:
        return handle.read()


def _write(work, relative, text):
    os.makedirs(os.path.dirname(_path(work, relative)) or work, exist_ok=True)
    with open(_path(work, relative), "w", encoding="utf-8") as handle:
        handle.write(text)


def _replace(work, relative, old, new):
    """Replace the one occurrence of `old`; a target that is absent or repeated
    means the mutation would not land where it claims."""
    text = _text(work, relative)
    if text.count(old) != 1:
        raise AssertionError(f"{relative}: {old!r} occurs {text.count(old)} times, not once")
    _write(work, relative, text.replace(old, new))


def _rows_block(work, new):
    """Replace the declaration's whole `rows:` block with `new`."""
    text = _text(work, DECL)
    blocks = re.findall(r"^rows:\n(?:  - .*\n)+", text, flags=re.MULTILINE)
    if len(blocks) != 1:
        raise AssertionError(f"{DECL}: {len(blocks)} `rows:` block(s), not one")
    _write(work, DECL, text.replace(blocks[0], new))


def _line_of(work, relative, line):
    """The 1-based number of `line` in the file, found here, not by the checker."""
    lines = _text(work, relative).split("\n")
    if lines.count(line) != 1:
        raise AssertionError(f"{relative}: {line!r} occurs {lines.count(line)} times, not once")
    return lines.index(line) + 1


def _step_runs(work, relative, needle):
    """The planted text is inside a step's `run:`, read by the checker's own parser."""
    doc = checker.load_yaml(_text(work, relative), relative)
    for _job, step in checker._workflow_steps(doc, relative):
        if needle in str(step.get("run", "")) or needle in str(step.get("uses", "")):
            return None
    return f"no step of {relative} carries {needle!r}"


def _plant_step(work, relative, anchor, step):
    _replace(work, relative, anchor, step + anchor)


def _copy_tree(work):
    for path in checker.tracked_files(ROOT):
        if path.startswith(COPIED) or path in COPIED_FILES:
            target = _path(work, path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(_path(ROOT, path), target)


def _track(work):
    for argv in (["git", "init", "--quiet"], ["git", "add", "--all"]):
        subprocess.run(argv, cwd=work, env=checker.git_env(), check=True,
                       stdin=subprocess.DEVNULL, capture_output=True)


# --- the tree mutations ---------------------------------------------------
# (description, mutate, landed, expected). `landed(work)` returns why the mutation
# did NOT apply, or None; `expected(work)` is every line `check()` must return.

D9_ANCHOR = "      - name: D9 rule checker\n"
ZOO_ANCHOR = "      - name: Every model-zoo path in the notebooks resolves\n"
NIGHTLY_RECIPE = 'mutation-nightly:\n\t$(WIRING) --run --zoo "$(ZOO)"\n'
PHONY = ".PHONY: check templates zoo-paths wiring mutations mutation-nightly setup\n"
WEEKLY_ENTRY = "          ENTRY: ${{ 'make mutation-nightly' }}\n"


def _make_prints(work, target, needle):
    text = checker.make_dry_run(work, [target])
    return None if needle in text else f"`make -n {target}` does not print {needle!r}"


TREE_MUTATIONS = (
    (
        "`rows:` given through a YAML alias, a shape the reader refuses",
        lambda w: _replace(w, DECL, "  - scripts/check_templates_mutations.py\n", "  - *templates\n"),
        lambda w: None if "  - *templates" in _text(w, DECL) else "the alias is not in the declaration",
        lambda w: [f"{DECL}:{_line_of(w, DECL, '  - *templates')}: anchors, aliases and tags are not read"],
    ),
    (
        "`rows:` a single string instead of a list",
        lambda w: _rows_block(w, "rows: scripts/check_templates_mutations.py\n"),
        lambda w: None if "rows: scripts/" in _text(w, DECL) else "the scalar is not in the declaration",
        lambda w: [f"{DECL}: `rows` must be a list of globs, got 'scripts/check_templates_mutations.py'"],
    ),
    (
        "`rows:` empty: a weekly run of nothing",
        lambda w: _rows_block(w, "rows: []\n"),
        lambda w: None if "rows: []" in _text(w, DECL) else "the empty list is not in the declaration",
        lambda w: [f"{DECL}: `rows` is empty, and running no rows is not a pass"],
    ),
    (
        "a row glob that matches no tracked file",
        lambda w: _replace(w, DECL, "  - scripts/check_zoo_paths_mutations.py\n",
                           "  - scripts/check_zoo_paths_mutations.py\n  - scripts/check_nothing_mutations.py\n"),
        lambda w: None if "check_nothing_mutations.py" in _text(w, DECL) else "the glob is not in the declaration",
        lambda w: [f"{DECL}: row glob 'scripts/check_nothing_mutations.py' matches no tracked file"],
    ),
    (
        "a rows file with no __main__ block, which `--run` cannot run",
        lambda w: (
            _write(w, "scripts/check_inert_mutations.py", 'print("no rows here")\n'),
            _replace(w, DECL, "  - scripts/check_zoo_paths_mutations.py\n",
                     "  - scripts/check_zoo_paths_mutations.py\n  - scripts/check_inert_mutations.py\n"),
        ),
        lambda w: None if "scripts/check_inert_mutations.py" in checker.read_rows(w) else "the row is not read",
        lambda w: ['scripts/check_inert_mutations.py: no `if __name__ == "__main__":` block, so running it runs no row'],
    ),
    (
        "`make mutation-nightly` keeps a list of its own instead of reading the declaration",
        lambda w: _replace(w, "Makefile", NIGHTLY_RECIPE,
                           "mutation-nightly:\n\t$(PYTHON) scripts/check_templates_mutations.py\n"
                           "\t$(PYTHON) scripts/check_zoo_paths_mutations.py --zoo $(ZOO)\n"),
        lambda w: _make_prints(w, "mutation-nightly", "scripts/check_zoo_paths_mutations.py"),
        lambda w: [
            "`make mutation-nightly` runs `scripts/check_mutation_wiring.py --run` 0 times; once is what "
            "makes its rows the ones .github/mutation-coverage.yml declares",
            "`make mutation-nightly` names scripts/check_templates_mutations.py itself; the rows come from "
            ".github/mutation-coverage.yml alone",
            "`make mutation-nightly` names scripts/check_zoo_paths_mutations.py itself; the rows come from "
            ".github/mutation-coverage.yml alone",
        ],
    ),
    (
        "`make mutation-nightly` runs the rows without the SDK",
        lambda w: _replace(w, "Makefile", NIGHTLY_RECIPE, NIGHTLY_RECIPE.replace("--run", "--run --local")),
        lambda w: _make_prints(w, "mutation-nightly", "--run --local"),
        lambda w: ["`make mutation-nightly` passes --local, so the weekly run has no SDK and skips the rows that need it"],
    ),
    (
        "`make mutation-nightly` is gone",
        lambda w: (_replace(w, "Makefile", NIGHTLY_RECIPE, ""),
                   _replace(w, "Makefile", PHONY, PHONY.replace(" mutation-nightly", ""))),
        lambda w: None if "mutation-nightly:" not in _text(w, "Makefile") else "the target is still there",
        lambda w: ["`make -n mutation-nightly` exited 2, so the weekly entry point cannot be read"],
    ),
    (
        "a pull-request workflow runs a harness again",
        lambda w: _plant_step(w, TEMPLATE_RULES, D9_ANCHOR,
                              "      - name: Mutation harness\n"
                              "        run: python3 scripts/check_templates_mutations.py\n\n"),
        lambda w: _step_runs(w, TEMPLATE_RULES, "scripts/check_templates_mutations.py"),
        lambda w: [f"{TEMPLATE_RULES}: job `d9-rules` runs scripts/check_templates_mutations.py; {ALONE}"],
    ),
    (
        "a pull-request workflow runs `make check`, which runs every row",
        lambda w: _plant_step(w, TEMPLATE_RULES, D9_ANCHOR, "      - run: make check\n\n"),
        lambda w: _step_runs(w, TEMPLATE_RULES, "make check"),
        lambda w: [f"{TEMPLATE_RULES}: job `d9-rules` runs `make check`, whose dry run runs the mutation rows; {ALONE}"],
    ),
    (
        "a pull-request workflow runs the weekly entry, hidden in env and bash -c",
        lambda w: _plant_step(w, TEMPLATE_RULES, D9_ANCHOR,
                              "      - env:\n"
                              "          ENTRY: ${{ 'make mutation-nightly' }}\n"
                              '        run: bash -euo pipefail -c "$ENTRY"\n\n'),
        lambda w: _step_runs(w, TEMPLATE_RULES, '-c "$ENTRY"'),
        lambda w: [
            f"{TEMPLATE_RULES}: job `d9-rules` runs `make mutation-nightly`, whose dry run runs the mutation "
            f"rows; {ALONE}"
        ],
    ),
    (
        "a scheduled workflow runs every row through the runner",
        lambda w: _plant_step(w, ZOO_PATHS, ZOO_ANCHOR,
                              "      - run: python3 scripts/check_mutation_wiring.py --run --zoo model-zoo\n\n"),
        lambda w: _step_runs(w, ZOO_PATHS, "check_mutation_wiring.py --run"),
        lambda w: [
            f"{ZOO_PATHS}: job `zoo-paths` runs `python3 scripts/check_mutation_wiring.py --run --zoo "
            f"model-zoo`; {ALONE}"
        ],
    ),
    (
        "a workflow step the guard cannot follow (a local action)",
        lambda w: _plant_step(w, ZOO_PATHS, ZOO_ANCHOR, "      - uses: ./.github/actions/rows\n\n"),
        lambda w: _step_runs(w, ZOO_PATHS, "./.github/actions/rows"),
        lambda w: [
            f"{ZOO_PATHS}: job `zoo-paths` cannot be read: it uses ./.github/actions/rows, which this "
            "guard does not read"
        ],
    ),
    (
        "a workflow the guard cannot read",
        lambda w: _replace(w, ZOO_PATHS, "    timeout-minutes: 10\n", "    timeout-minutes: *ten\n"),
        lambda w: None if "*ten" in _text(w, ZOO_PATHS) else "the alias is not in the workflow",
        lambda w: [f"{ZOO_PATHS}:{_line_of(w, ZOO_PATHS, '    timeout-minutes: *ten')}: anchors, aliases and tags are not read"],
    ),
    (
        "the weekly workflow also triggers on pull requests",
        lambda w: _replace(w, WEEKLY, "\non:\n  schedule:\n", "\non:\n  pull_request:\n  schedule:\n"),
        lambda w: None if "pull_request" in checker.load_yaml(_text(w, WEEKLY), WEEKLY)["on"] else "no trigger planted",
        lambda w: [
            f"{WEEKLY} triggers on ['pull_request', 'schedule', 'workflow_dispatch']; the weekly run "
            "triggers on ['schedule', 'workflow_dispatch'] only"
        ],
    ),
    (
        "the weekly workflow runs another command",
        lambda w: _replace(w, WEEKLY, WEEKLY_ENTRY, WEEKLY_ENTRY.replace("mutation-nightly", "check")),
        lambda w: _step_runs(w, WEEKLY, "$ENTRY") or (
            None if "'make check'" in _text(w, WEEKLY) else "the entry is unchanged"),
        lambda w: [f"{WEEKLY} does not run `make mutation-nightly`"],
    ),
    (
        "the weekly workflow is gone",
        lambda w: os.remove(_path(w, WEEKLY)),
        lambda w: None if not os.path.exists(_path(w, WEEKLY)) else "the workflow is still there",
        lambda w: [f"{WEEKLY}: missing, so nothing runs the mutation rows weekly"],
    ),
)


# --- the runner cases -----------------------------------------------------


def _planted_rows(work, rows):
    for row, code in rows:
        _write(work, row, EXIT_WITH.format(code=code))
    _rows_block(work, "rows:\n" + "".join(f"  - {row}\n" for row, _ in rows))


def _run(work):
    lines = []
    code = checker.run(work, local=True, zoo=work, out=lines.append)
    return code, lines


RUN_CASES = (
    # (description, the planted rows, exit code, the last line, a line that must appear)
    ("every planted row passes, so the run passes",
     [(GREEN_ROW, 0)], 0, "mutation rows: all 1 passed", f"  ok      {GREEN_ROW}  ("),
    ("one planted row fails, so the run fails and names it",
     [(GREEN_ROW, 0), (RED_ROW, 3)], 1, f"mutation rows FAILED: {RED_ROW}", f"  FAILED  {RED_ROW}  ("),
    ("no rows at all, so the run refuses",
     [], 2, f"mutation rows: cannot run -- {DECL}: `rows` is empty, and running no rows is not a pass", None),
)


def _check_run_case(base, description, rows, want_code, want_last, want_line):
    work = os.path.join(base, f"run-{len(os.listdir(base))}")
    shutil.copytree(os.path.join(base, "clean"), work, ignore=shutil.ignore_patterns(".git"))
    if rows:
        _planted_rows(work, rows)
    else:
        _rows_block(work, "rows: []\n")
    _track(work)
    if rows and [r for r, _ in rows] != checker.read_rows(work):
        return f"{description}: the planted rows did not land ({checker.read_rows(work)})"
    code, lines = _run(work)
    if code != want_code or not lines or lines[-1] != want_last:
        return (f"{description}: expected exit {want_code} ending {want_last!r}, "
                f"got exit {code} ending {lines[-1] if lines else None!r}")
    if want_line and not any(line.startswith(want_line) for line in lines):
        return f"{description}: no line starts with {want_line!r} in {lines}"
    return None


def _check_leg_refusal():
    env = {"MUTATION_LEG": "2", "MUTATION_LEGS": "1"}
    want = "MUTATION_LEG='2' and MUTATION_LEGS='1' are not a leg 1..N of N"
    try:
        share = checker.leg_share(["a", "b"], env)
    except checker.CannotRead as exc:
        return None if str(exc) == want else f"a leg outside its count: refused with {exc}, not {want!r}"
    return f"a leg outside its count ran {share}"


# --- the run --------------------------------------------------------------


def main() -> int:
    print("mutation harness: every rule of check_mutation_wiring.py, seen to fail\n")
    failures = []
    with tempfile.TemporaryDirectory(prefix="mutation-wiring-") as base:
        clean = os.path.join(base, "clean")
        _copy_tree(clean)
        _track(clean)
        control = checker.check(clean)
        print(f"  [{'ok' if not control else 'BROKEN'}] the committed tree passes")
        if control:
            for line in control:
                print(f"           {line}")
            failures.append("the committed tree does not pass")

        for n, (description, mutate, landed, expected) in enumerate(TREE_MUTATIONS):
            work = os.path.join(base, f"m{n}")
            shutil.copytree(clean, work, ignore=shutil.ignore_patterns(".git"))
            try:
                mutate(work)
                _track(work)
                problem = landed(work)
                want = expected(work)
            except (AssertionError, checker.CannotRead, KeyError, TypeError) as exc:
                problem = f"{type(exc).__name__}: {exc}"
            if problem:
                print(f"  [INERT ] {description}\n           {problem}")
                failures.append(f"{description}: the mutation did not apply ({problem})")
                continue
            got = checker.check(work)
            if got == want:
                print(f"  [caught] {description}")
                for line in got:
                    print(f"           {line}")
            else:
                print(f"  [{'MISSED' if not got else 'WRONG '}] {description}")
                print(f"           expected: {want}")
                print(f"           got:      {got}")
                failures.append(description)

        for description, rows, want_code, want_last, want_line in RUN_CASES:
            problem = _check_run_case(base, description, rows, want_code, want_last, want_line)
            verdict = "WRONG " if problem else "ok" if want_code == 0 else "caught"
            print(f"  [{verdict}] {description}")
            if problem:
                print(f"           {problem}")
                failures.append(problem)

    problem = _check_leg_refusal()
    print(f"  [{'caught' if not problem else 'WRONG '}] a leg outside its count refuses to run")
    if problem:
        failures.append(problem)

    if failures:
        print("\nmutation harness FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    total = len(TREE_MUTATIONS) + len(RUN_CASES) + 1
    print(f"\nOK -- {total} mutations, each caught with the violation it causes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
