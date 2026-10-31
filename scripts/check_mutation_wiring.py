#!/usr/bin/env python3
"""The mutation rows run on the weekly schedule, in one place, never on a pull request.

Run from the repo root:

    python3 scripts/check_mutation_wiring.py          the wiring check
    python3 scripts/check_mutation_wiring.py --rows   the rows files, one per line
    python3 scripts/check_mutation_wiring.py --run    run them all (`make mutation-nightly`)

Its mutation harness is `scripts/check_mutation_wiring_mutations.py`.

What is checked
---------------

A mutation row proves a checker can still fail; `rows:` in
`.github/mutation-coverage.yml` names the files that hold them. They run in
`.github/workflows/mutation-nightly.yml`, whose one command is `make mutation-nightly`,
and in no other workflow: a pull request does not wait on them, and a checker it
breaks is caught by the next weekly run. This file holds that wiring:

1. `rows:` reads cleanly: a non-empty list of globs, each matching tracked Python
   files that have a `__main__` block, so `--run` can run every one.
2. `make mutation-nightly` runs `--run` once, which reads `rows:` itself, and names
   no rows file of its own: the set is the declaration's, never a second list.
3. No workflow but the weekly one runs a rows file: by name, through `--run`, or
   through a make target whose dry run does either.
4. The weekly workflow triggers on `schedule` and `workflow_dispatch` only, and
   runs `make mutation-nightly`.

Every input is parsed, never grepped: YAML by `load_yaml` (the block subset
workflows use; anything else is refused), shell by `commands`, make by `make -n`.
An input it cannot read is a finding, never a pass.

Running the rows
----------------

`--run` runs each rows file as `<python> <file>` from the repo root, with
`MODEL_ZOO_PATH` set to the zoo, and exits 1 when any of them fails, 2 when it
cannot run them. Without `--local` it is the weekly run: the SDK goes into a venv
of its own (the rows of `check_templates.py` need it), the public model zoo is
cloned when `--zoo` does not exist, and only this leg's share runs when
`MUTATION_LEG` and `MUTATION_LEGS` are set. With `--local` (`make check`) it runs
under this interpreter, against the zoo it is given.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, "/")

DECLARATION = ".github/mutation-coverage.yml"
#: The declaration's keys, as `mutation-coverage / check` reads them.
DECLARATION_KEYS = ("exclude", "exempt", "guards", "rows")
WORKFLOWS = ".github/workflows"
WEEKLY = ".github/workflows/mutation-nightly.yml"
WEEKLY_EVENTS = frozenset({"schedule", "workflow_dispatch"})
WEEKLY_TARGET = "mutation-nightly"

ZOO_URL = "https://github.com/tracebloc/model-zoo.git"
#: The SDK the rows of check_templates.py import, installed as template-rules.yml's
#: "Install the SDK" step installs it; that step records why each package is there.
SDK_INSTALLS = (("--no-deps", "tracebloc"), ("numpy", "psutil", "requests", "rich", "termcolor", "tqdm"))
SDK_PROBE = (
    "import importlib.metadata as m\n"
    "from tracebloc.training.plan import _SURVIVAL_FRAMEWORKS\n"
    "print('tracebloc', m.version('tracebloc'))\n"
)
#: A row that has not finished by then is a failure, inside the weekly job's cap.
ROW_TIMEOUT = 30 * 60
STEP_TIMEOUT = 10 * 60


class CannotRead(Exception):
    """An input the guard cannot read. A finding, never a pass."""


# --------------------------------------------------------------------- YAML


class _Yaml:
    """Block YAML as workflows write it: mappings, sequences, plain, quoted, literal
    and folded scalars, and flow collections. Scalars stay strings. Anchors, aliases,
    tags, complex keys, indentation indicators and multiple documents are refused."""

    def __init__(self, text, where):
        self.where = where
        self.lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def fail(self, i, why):
        raise CannotRead(f"{self.where}:{i + 1}: {why}")

    def indent(self, i):
        line = self.lines[i]
        n = len(line) - len(line.lstrip(" "))
        if line[n : n + 1] == "\t":
            self.fail(i, "a tab in the indentation")
        return n

    def sig(self, i):
        """The next line at or after `i` that is neither blank nor a comment."""
        while i < len(self.lines):
            text = self.lines[i].strip()
            if text and not text.startswith("#"):
                return i
            i += 1
        return i

    @staticmethod
    def is_item(content):
        return content == "-" or content.startswith(("- ", "-\t"))

    def document(self):
        i = self.sig(0)
        if i < len(self.lines) and self.lines[i].startswith("%"):
            self.fail(i, "a YAML directive")
        if i < len(self.lines) and self.lines[i].rstrip() == "---":
            i = self.sig(i + 1)
        if i == len(self.lines):
            raise CannotRead(f"{self.where}: an empty document")
        if self.indent(i) != 0:
            self.fail(i, "the document does not start at column 0")
        value, i = self.block(i, 0, -1)
        i = self.sig(i)
        if i < len(self.lines):
            self.fail(i, "text after the document")
        return value

    def block(self, i, n, parent):
        content = self.lines[i][n:]
        if self.is_item(content):
            return self.sequence(i, n)
        if self.key_split(i, content) is not None:
            return self.mapping(i, n)
        return self.value(i, content.strip(), parent)

    def sequence(self, i, n):
        items = []
        while True:
            i = self.sig(i)
            if i == len(self.lines) or self.indent(i) < n:
                return items, i
            if self.indent(i) > n:
                self.fail(i, "unexpected indentation")
            content = self.lines[i][n:]
            if not self.is_item(content):
                return items, i
            rest = content[1:].lstrip(" ")
            if not rest.strip() or rest.startswith("#"):
                j = self.sig(i + 1)
                if j < len(self.lines) and self.indent(j) > n:
                    item, i = self.block(j, self.indent(j), n)
                else:
                    item, i = None, i + 1
            else:
                column = len(self.lines[i]) - len(rest)
                self.lines[i] = " " * column + rest
                if self.is_item(rest) or self.key_split(i, rest) is not None:
                    item, i = self.block(i, column, n)
                else:
                    item, i = self.value(i, rest.strip(), n)
            items.append(item)

    def mapping(self, i, n):
        result = {}
        while True:
            i = self.sig(i)
            if i == len(self.lines) or self.indent(i) < n:
                return result, i
            if self.indent(i) > n:
                self.fail(i, "unexpected indentation")
            content = self.lines[i][n:]
            if self.is_item(content):
                self.fail(i, "a sequence item among mapping keys")
            split = self.key_split(i, content)
            if split is None:
                self.fail(i, "expected `key: value`")
            key, rest = split
            if key in result:
                self.fail(i, f"duplicate key {key!r}")
            rest = rest.strip()
            if not rest or rest.startswith("#"):
                j = self.sig(i + 1)
                if j < len(self.lines) and self.indent(j) > n:
                    value, i = self.block(j, self.indent(j), n)
                elif j < len(self.lines) and self.indent(j) == n and self.is_item(self.lines[j][n:]):
                    value, i = self.sequence(j, n)
                else:
                    value, i = None, i + 1
            else:
                value, i = self.value(i, rest, n)
            result[key] = value

    def key_split(self, i, content):
        """(key, the text after its colon), or None when `content` is not a key line."""
        if content[:1] == "?" and content[1:2] in ("", " ", "\t"):
            self.fail(i, "a complex key")
        if content[:1] in ("'", '"'):
            scanned = _scan_quoted(content, 0)
            if scanned is None:
                return None
            key, end = scanned
            match = re.match(r"[ \t]*:(?=[ \t]|$)", content[end:])
            return (key, content[end + match.end() :]) if match else None
        if not content or content[0] in "[{&*!|>%@`#":
            return None
        for match in re.finditer(r":(?=[ \t]|$)", content):
            key = content[: match.start()]
            if re.search(r"[ \t]#", key):
                return None
            key = key.rstrip()
            return (key, content[match.end() :]) if key else None
        return None

    def value(self, i, text, n):
        """The scalar or flow collection starting with `text` on line `i`; its
        continuation lines are indented past `n`."""
        head = text[:1]
        if head in "&*!":
            self.fail(i, "anchors, aliases and tags are not read")
        if head in "@`%" or self.is_item(text):
            self.fail(i, f"a value cannot start with {text[:2]!r}")
        if head in "|>":
            return self.block_scalar(i, text, n)
        if head in "[{":
            return self.flow(i, text)
        if head in ("'", '"'):
            return self.quoted(i, text)
        return self.plain(i, text, n)

    def plain(self, i, text, n):
        first = re.split(r"[ \t]#", text, maxsplit=1)
        if re.search(r":(?=[ \t]|$)", first[0]):
            self.fail(i, "a `key:` inside a plain scalar")
        result, closed, last = first[0].rstrip(), len(first) > 1, i
        j, blanks = i + 1, 0
        while j < len(self.lines) and not closed:
            stripped = self.lines[j].strip()
            if not stripped:
                blanks, j = blanks + 1, j + 1
                continue
            if self.indent(j) <= n or stripped.startswith("#"):
                break
            parts = re.split(r"[ \t]#", stripped, maxsplit=1)
            if re.search(r":(?=[ \t]|$)", parts[0]):
                self.fail(j, "a `key:` inside a multi-line plain scalar")
            result += ("\n" * blanks if blanks else " ") + parts[0].rstrip()
            closed, last, blanks, j = len(parts) > 1, j, 0, j + 1
        return result, last + 1

    def quoted(self, i, text):
        buffer, j = text, i
        while True:
            scanned = _scan_quoted(buffer, 0)
            if scanned is not None:
                value, end = scanned
                rest = buffer[end:].strip()
                if rest and not rest.startswith("#"):
                    self.fail(j, "text after a quoted scalar")
                return value, j + 1
            escaped_break = text[0] == '"' and (len(buffer) - len(buffer.rstrip("\\"))) % 2 == 1
            buffer = buffer[:-1] if escaped_break else buffer.rstrip(" \t")
            blanks = 0
            j += 1
            while j < len(self.lines) and not self.lines[j].strip():
                blanks, j = blanks + 1, j + 1
            if j == len(self.lines):
                self.fail(i, "an unclosed quoted scalar")
            line = self.lines[j].strip(" \t")
            joint = "\n" * blanks if blanks else ("" if escaped_break else " ")
            buffer += joint + line

    def block_scalar(self, i, text, n):
        match = re.fullmatch(r"([|>])([+-]?)(?:[ \t]+#.*)?[ \t]*", text)
        if not match:
            self.fail(i, f"a block scalar header this guard does not read: {text!r}")
        style, chomp = match.groups()
        j = i + 1
        while j < len(self.lines) and not self.lines[j].strip():
            j += 1
        if j == len(self.lines) or self.indent(j) <= n:
            return "", i + 1
        column = self.indent(j)
        body, j = [], i + 1
        while j < len(self.lines):
            line = self.lines[j]
            spaces = len(line) - len(line.lstrip(" "))
            if line.strip() and spaces < column:
                if line[spaces : spaces + 1] == "\t":
                    self.fail(j, "a tab in the indentation")
                break
            body.append(line[column:] if line.strip() else "")
            j += 1
        trailing = len(body) - len(_rstrip_blank(body))
        body = body[: len(body) - trailing]
        out = "\n".join(body) if style == "|" else _fold(body)
        if j == len(self.lines) and not trailing and self.lines[-1].strip():
            return out, j  # the file ends inside the scalar, with no line break to keep
        if chomp == "+":
            out += "\n" * (trailing + (1 if body else 0))
        elif chomp == "" and body:
            out += "\n"
        return out, j

    def flow(self, i, text):
        buffer, j = _strip_flow_comment(text), i
        while not _flow_balanced(buffer):
            j += 1
            if j == len(self.lines):
                self.fail(i, "an unclosed flow collection")
            buffer += " " + _strip_flow_comment(self.lines[j].strip())
        try:
            value, end = _flow_value(buffer, 0)
        except ValueError as exc:
            self.fail(i, f"a flow collection this guard does not read ({exc})")
        if buffer[end:].strip():
            self.fail(j, "text after a flow collection")
        return value, j + 1


def _rstrip_blank(lines):
    end = len(lines)
    while end and not lines[end - 1]:
        end -= 1
    return lines[:end]


def _fold(lines):
    """Folded-scalar line joining: a break between two plain lines becomes a space,
    and breaks around blank or more-indented lines are kept."""
    out, previous, blanks = "", None, 0
    for line in lines:
        if not line:
            blanks += 1
            continue
        kind = "more" if line[:1] in (" ", "\t") else "plain"
        if previous is None:
            out += "\n" * blanks
        elif previous == kind == "plain":
            out += "\n" * blanks if blanks else " "
        else:
            out += "\n" * (blanks + 1)
        out += line
        previous, blanks = kind, 0
    return out


_ESCAPES = {
    "0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t", "n": "\n", "v": "\v", "f": "\f",
    "r": "\r", "e": "\x1b", " ": " ", '"': '"', "/": "/", "\\": "\\", "N": "\x85",
    "_": "\xa0", "L": "\u2028", "P": "\u2029",
}
_HEX_ESCAPES = {"x": 2, "u": 4, "U": 8}


def _scan_quoted(text, start):
    """(value, the index after the closing quote) of the quoted scalar at `start`,
    or None when it does not close in `text`."""
    quote, out, k = text[start], [], start + 1
    while k < len(text):
        ch = text[k]
        if quote == "'" and ch == "'":
            if text[k + 1 : k + 2] == "'":
                out.append("'")
                k += 2
                continue
            return "".join(out), k + 1
        if quote == '"' and ch == '"':
            return "".join(out), k + 1
        if quote == '"' and ch == "\\":
            code = text[k + 1 : k + 2]
            if code in _ESCAPES:
                out.append(_ESCAPES[code])
                k += 2
                continue
            if code in _HEX_ESCAPES:
                digits = text[k + 2 : k + 2 + _HEX_ESCAPES[code]]
                if len(digits) != _HEX_ESCAPES[code] or not re.fullmatch(r"[0-9A-Fa-f]+", digits):
                    raise CannotRead(f"a bad \\{code} escape in {text!r}")
                out.append(chr(int(digits, 16)))
                k += 2 + len(digits)
                continue
            if not code:
                return None
            raise CannotRead(f"an unknown escape \\{code} in {text!r}")
        out.append(ch)
        k += 1
    return None


def _strip_flow_comment(text):
    quote = None
    for k, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (k == 0 or text[k - 1] in " \t"):
            return text[:k].rstrip()
    return text


def _flow_balanced(text):
    depth, quote = 0, None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
    return depth <= 0


def _flow_value(text, p):
    while p < len(text) and text[p] in " \t":
        p += 1
    if p == len(text):
        raise ValueError("a missing value")
    head = text[p]
    if head in "&*!":
        raise ValueError("anchors, aliases and tags are not read")
    if head in ("'", '"'):
        scanned = _scan_quoted(text, p)
        if scanned is None:
            raise ValueError("an unclosed quoted scalar")
        return scanned
    if head not in "[{":
        match = re.match(r"(?:[^,\[\]{}:#]|:(?![ \t,\]}]|$)|(?<![ \t])#)*", text[p:])
        return match.group(0).strip(), p + match.end()
    close, out = ("]", []) if head == "[" else ("}", {})
    p += 1
    while True:
        while p < len(text) and text[p] in " \t":
            p += 1
        if text[p : p + 1] == close:
            return out, p + 1
        if head == "[":
            item, p = _flow_value(text, p)
            if text[p : p + 1].strip() == ":" or re.match(r"[ \t]*:", text[p:]):
                raise ValueError("a mapping inside a flow sequence")
            out.append(item)
        else:
            key, p = _flow_value(text, p)
            match = re.match(r"[ \t]*:", text[p:])
            if not isinstance(key, str) or not match:
                raise ValueError("a flow mapping entry without `key:`")
            p += match.end()
            if re.match(r"[ \t]*[,}]", text[p:]):
                value = None
            else:
                value, p = _flow_value(text, p)
            if key in out:
                raise ValueError(f"duplicate key {key!r}")
            out[key] = value
        match = re.match(r"[ \t]*(,|\]|\})", text[p:])
        if not match or (match.group(1) != "," and match.group(1) != close):
            raise ValueError(f"expected `,` or `{close}`")
        p += match.end()
        if match.group(1) == close:
            return out, p


def load_yaml(text, where):
    """`text` as nested dicts, lists and strings; CannotRead for anything else."""
    return _Yaml(text, where).document()


# --------------------------------------------------------------------- the declaration


def glob_regex(pattern):
    """A `rows:` glob as `mutation-coverage / check` reads it: `*` and `?` inside one
    path segment, `**/` across whole segments, a bare `**` across anything."""
    if pattern.startswith(("/", "./")) or any(c in pattern for c in "[]{}\\"):
        raise CannotRead(
            f"{DECLARATION}: row glob {pattern!r} is not repo-relative or uses syntax "
            "beyond `*`, `?` and `**`"
        )
    out, k = [], 0
    while k < len(pattern):
        if pattern.startswith("**/", k):
            out.append("(?:[^/]+/)*")
            k += 3
        elif pattern.startswith("**", k):
            out.append(".*")
            k += 2
        elif pattern[k] == "*":
            out.append("[^/]*")
            k += 1
        elif pattern[k] == "?":
            out.append("[^/]")
            k += 1
        else:
            out.append(re.escape(pattern[k]))
            k += 1
    return re.compile("".join(out) + r"\Z")


def git_env():
    """This environment without the variables a git hook exports: with them set, git
    reads the hook's repository whatever directory it is pointed at."""
    hook_vars = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY")
    return {k: v for k, v in os.environ.items() if k not in hook_vars}


def tracked_files(root):
    try:
        proc = subprocess.run(
            ["git", "-C", root, "ls-files", "-z"], env=git_env(),
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CannotRead(f"`git ls-files` could not run: {exc}") from exc
    if proc.returncode != 0:
        raise CannotRead(f"`git ls-files` exited {proc.returncode}: {proc.stderr.strip()}")
    return sorted(p for p in proc.stdout.split("\0") if p)


def _read(root, relative):
    try:
        with open(os.path.join(root, relative), encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise CannotRead(f"{relative}: cannot be read ({exc.strerror})") from exc


def read_rows(root):
    """The rows files, sorted: the `rows:` globs of the declaration matched against
    `git ls-files`. CannotRead for any shape the declaration should not have."""
    doc = load_yaml(_read(root, DECLARATION), DECLARATION)
    if not isinstance(doc, dict):
        raise CannotRead(f"{DECLARATION}: the top level is not a mapping")
    unknown = sorted(set(doc) - set(DECLARATION_KEYS))
    if unknown:
        raise CannotRead(f"{DECLARATION}: unknown key(s) {unknown}; the keys are {list(DECLARATION_KEYS)}")
    globs = doc.get("rows")
    if not isinstance(globs, list) or not all(isinstance(g, str) and g for g in globs):
        raise CannotRead(f"{DECLARATION}: `rows` must be a list of globs, got {globs!r}")
    if not globs:
        raise CannotRead(f"{DECLARATION}: `rows` is empty, and running no rows is not a pass")
    tracked, rows = tracked_files(root), []
    for pattern in globs:
        regex = glob_regex(pattern)
        hits = [p for p in tracked if regex.match(p)]
        if not hits:
            raise CannotRead(f"{DECLARATION}: row glob {pattern!r} matches no tracked file")
        rows += [p for p in hits if p not in rows]
    return sorted(rows)


def _is_main_guard(node):
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left, test.comparators[0]]
    names = [s for s in sides if isinstance(s, ast.Name) and s.id == "__name__"]
    consts = [s for s in sides if isinstance(s, ast.Constant) and s.value == "__main__"]
    return len(names) == 1 and len(consts) == 1


def row_problems(root, rows):
    """Why `--run` could not run each rows file: `<python> <file>` needs a Python
    file with a `__main__` block."""
    problems = []
    for row in rows:
        if not row.endswith(".py"):
            problems.append(f"{row}: not a Python file, so `{SELF} --run` cannot run it")
            continue
        try:
            tree = ast.parse(_read(root, row), filename=row)
        except CannotRead as exc:
            problems.append(str(exc))
            continue
        except SyntaxError as exc:
            problems.append(f"{row}: does not parse (line {exc.lineno}: {exc.msg})")
            continue
        if not any(_is_main_guard(node) for node in tree.body):
            problems.append(f'{row}: no `if __name__ == "__main__":` block, so running it runs no row')
    return problems


# --------------------------------------------------------------------- shell and make

_HEREDOC = re.compile(r"(?<!<)<<(-?)[ \t]*(?:'([A-Za-z_]\w*)'|\"([A-Za-z_]\w*)\"|\\?([A-Za-z_]\w*))")
_PREFIXES = frozenset({"if", "then", "else", "elif", "do", "while", "until", "!", "{", "time", "exec", "command", "nohup"})
_SHELLS = frozenset({"bash", "sh", "dash", "zsh"})


def _drop_heredoc_bodies(script):
    out, lines, k = [], script.split("\n"), 0
    while k < len(lines):
        out.append(lines[k])
        found = _HEREDOC.search(lines[k])
        k += 1
        if found:
            word = found.group(2) or found.group(3) or found.group(4)
            while k < len(lines) and (lines[k].lstrip("\t") if found.group(1) else lines[k]) != word:
                k += 1
            k += 1
    return "\n".join(out)


def _shell_text(script):
    """`script` with comments dropped and unquoted line breaks as `;`, so shlex
    keeps command boundaries."""
    out, quote, k = [], None, 0
    while k < len(script):
        ch = script[k]
        if quote == "'":
            out.append(ch)
            quote = None if ch == "'" else quote
        elif quote == '"':
            out.append(ch)
            if ch == "\\" and k + 1 < len(script):
                out.append(script[k + 1])
                k += 1
            elif ch == '"':
                quote = None
        elif ch == "\\" and script[k + 1 : k + 2] == "\n":
            out.append(" ")
            k += 1
        elif ch == "\\" and k + 1 < len(script):
            out.append(ch + script[k + 1])
            k += 1
        elif ch in ("'", '"'):
            out.append(ch)
            quote = ch
        elif ch == "#" and (not out or out[-1][-1] in " \t;&|("):
            while k + 1 < len(script) and script[k + 1] != "\n":
                k += 1
        elif ch == "\n":
            out.append(" ; ")
        else:
            out.append(ch)
        k += 1
    return "".join(out)


def commands(script, depth=0):
    """Every simple command `script` runs, as argv lists: through `;`, `&&`, `|`,
    subshells and `$( )`, and into `bash -c`. CannotRead when shlex cannot."""
    lexer = shlex.shlex(_shell_text(_drop_heredoc_bodies(script)), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise CannotRead(f"shell this guard cannot read ({exc})") from exc
    found, current = [], []
    for token in tokens + [";"]:
        if token and set(token) <= set(";&|()"):
            if current:
                found.append(current)
            current = []
        elif token and set(token) <= set("<>&") | set("0123456789") and set(token) & set("<>"):
            continue
        else:
            current.append(token)
    out = []
    for argv in found:
        while argv and (argv[0] in _PREFIXES or re.match(r"[A-Za-z_]\w*=", argv[0])):
            argv = argv[1:]
        if argv and argv[0] == "env":
            argv = argv[1:]
            while argv and (argv[0].startswith("-") or re.match(r"[A-Za-z_]\w*=", argv[0])):
                argv = argv[1:]
        if not argv:
            continue
        out.append(argv)
        inner = _shell_script_arg(argv)
        if inner is not None and depth < 3:
            out += commands(inner, depth + 1)
    return out


def _shell_script_arg(argv):
    """The script of `bash -c <script>`, or None."""
    if os.path.basename(argv[0]) not in _SHELLS:
        return None
    k, has_c = 1, False
    while k < len(argv) and argv[k][:1] in ("-", "+") and argv[k] not in ("-", "--"):
        cluster = argv[k]
        has_c = has_c or "c" in cluster[1:]
        k += 2 if cluster[1:] and cluster[-1] == "o" else 1
    return argv[k] if has_c and k < len(argv) else None


def make_targets(argv):
    """The targets a `make` command builds ([] for the default goal), or None when
    `argv` is not make. CannotRead for a make pointed at another makefile."""
    if os.path.basename(argv[0]) not in ("make", "gmake"):
        return None
    targets, k = [], 1
    while k < len(argv):
        arg = argv[k]
        if arg.startswith(("-C", "-f", "--directory", "--file", "--makefile")):
            raise CannotRead(f"`{shlex.join(argv)}` reads another directory or makefile")
        if arg in ("-I", "-o", "-W"):
            k += 1
        elif arg in ("-j", "-l") and k + 1 < len(argv) and argv[k + 1].isdigit():
            k += 1
        elif not arg.startswith("-") and "=" not in arg:
            targets.append(arg)
        k += 1
    return targets


def make_dry_run(root, targets):
    """What `make -n <targets>` prints, from `root`."""
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES", "GNUMAKEFLAGS")}
    shown = " ".join(["make", "-n", *targets])
    try:
        proc = subprocess.run(
            ["make", "-n", *targets], cwd=root, env=env,
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CannotRead(f"`{shown}` could not run: {exc}") from exc
    if proc.returncode != 0:
        raise CannotRead(f"`{shown}` exited {proc.returncode}")
    return proc.stdout


def is_runner(argv):
    """`argv` runs every rows file: this file with `--run`."""
    return "--run" in argv and any(os.path.basename(a) == os.path.basename(SELF) for a in argv)


def named_rows(text, rows):
    """The rows files `text` names, by path or file name, outside its comment lines."""
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [
        row for row in rows
        if re.search(r"(?<![\w.-])" + re.escape(os.path.basename(row)) + r"(?![\w/-]|\.\w)", body)
    ]


# --------------------------------------------------------------------- the checks


def _check_entry(root, rows):
    """`make mutation-nightly` runs `--run` once, without `--local`, and names no rows file."""
    target = f"`make {WEEKLY_TARGET}`"
    try:
        text = make_dry_run(root, [WEEKLY_TARGET])
        runs = [argv for argv in commands(text) if is_runner(argv)]
    except CannotRead as exc:
        return [f"{exc}, so the weekly entry point cannot be read"]
    problems = []
    if len(runs) != 1:
        problems.append(
            f"{target} runs `{SELF} --run` {len(runs)} times; once is what makes its rows "
            f"the ones {DECLARATION} declares"
        )
    elif "--local" in runs[0]:
        problems.append(f"{target} passes --local, so the weekly run has no SDK and skips the rows that need it")
    for row in named_rows(text, rows or []):
        problems.append(f"{target} names {row} itself; the rows come from {DECLARATION} alone")
    return problems


_LITERAL = re.compile(r"\$\{\{\s*(?:'((?:[^']|'')*)'|(-?\d+(?:\.\d+)?|true|false))\s*\}\}")


def _literal_env(env):
    """The variables of an `env:` mapping whose values are known before the run."""
    out = {}
    for name, value in (env or {}).items() if isinstance(env, dict) else ():
        if not isinstance(value, str):
            continue
        match = _LITERAL.fullmatch(value.strip())
        if match:
            out[name] = match.group(1).replace("''", "'") if match.group(1) is not None else match.group(2)
        elif "${{" not in value:
            out[name] = value
    return out


def _substitute(script, env):
    return re.sub(
        r"\$(?:\{(\w+)\}|(\w+))",
        lambda m: env.get(m.group(1) or m.group(2), m.group(0)),
        script,
    )


def _workflow_steps(doc, relative):
    """(job, step) for every step, the step carrying its `env` merged with its job's
    and its workflow's. CannotRead for a shape a workflow does not have."""
    if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict) or not doc["jobs"]:
        raise CannotRead(f"{relative}: `jobs` is not a mapping of jobs")
    for job_id, job in doc["jobs"].items():
        if not isinstance(job, dict):
            raise CannotRead(f"{relative}: job `{job_id}` is not a mapping")
        if isinstance(job.get("uses"), str):
            yield job_id, {"uses": job["uses"], "env": {}}
            continue
        steps = job.get("steps")
        if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
            raise CannotRead(f"{relative}: job `{job_id}` has no list of steps")
        for step in steps:
            env = {**_literal_env(doc.get("env")), **_literal_env(job.get("env")), **_literal_env(step.get("env"))}
            yield job_id, {**step, "env": env}


def _shell_commands(step):
    """The commands a `run:` step's shell runs, or [] for a step whose shell is not one."""
    shell = step.get("shell")
    if isinstance(shell, str) and shell.split()[:1] != [] and shell.split()[0] not in _SHELLS:
        return []
    return commands(_substitute(step["run"], step["env"]))


def _runs_rows(root, step, rows, make_cache):
    """What in `step` runs mutation rows: rows files it names, the runner, make targets."""
    uses = step.get("uses")
    if isinstance(uses, str) and uses.startswith("./"):
        raise CannotRead(f"it uses {uses}, which this guard does not read")
    if not isinstance(step.get("run"), str):
        return []
    found = list(named_rows(_substitute(step["run"], step["env"]), rows or []))
    for argv in _shell_commands(step):
        if is_runner(argv):
            found.append(f"`{shlex.join(argv)}`")
        targets = make_targets(argv)
        if targets is None:
            continue
        key = tuple(targets)
        if key not in make_cache:
            text = make_dry_run(root, targets)
            make_cache[key] = any(is_runner(a) for a in commands(text)) or bool(named_rows(text, rows or []))
        if make_cache[key]:
            found.append(f"`{shlex.join(['make', *targets])}`, whose dry run runs the mutation rows")
    return found


def _weekly_problems(doc, relative):
    on = doc.get("on") if isinstance(doc, dict) else None
    if isinstance(on, str):
        events = {on}
    elif isinstance(on, (list, dict)):
        events = set(on)
    else:
        return [f"{relative}: `on:` is not an event, a list or a mapping, so its triggers cannot be read"]
    problems = []
    if events != WEEKLY_EVENTS:
        problems.append(
            f"{relative} triggers on {sorted(events)}; the weekly run triggers on "
            f"{sorted(WEEKLY_EVENTS)} only"
        )
    entries = 0
    for _job, step in _workflow_steps(doc, relative):
        if isinstance(step.get("run"), str):
            entries += sum(make_targets(argv) == [WEEKLY_TARGET] for argv in _shell_commands(step))
    if not entries:
        problems.append(f"{relative} does not run `make {WEEKLY_TARGET}`")
    return problems


def _check_workflows(root, rows):
    try:
        names = sorted(n for n in os.listdir(os.path.join(root, WORKFLOWS)) if n.endswith((".yml", ".yaml")))
    except OSError as exc:
        return [f"{WORKFLOWS}: cannot be listed ({exc.strerror})"]
    problems, make_cache = [], {}
    if os.path.basename(WEEKLY) not in names:
        problems.append(f"{WEEKLY}: missing, so nothing runs the mutation rows weekly")
    for name in names:
        relative = f"{WORKFLOWS}/{name}"
        try:
            doc = load_yaml(_read(root, relative), relative)
            if relative == WEEKLY:
                problems += _weekly_problems(doc, relative)
                continue
            for job, step in _workflow_steps(doc, relative):
                try:
                    found = _runs_rows(root, step, rows, make_cache)
                except CannotRead as exc:
                    problems.append(f"{relative}: job `{job}` cannot be read: {exc}")
                    continue
                for what in found:
                    problems.append(f"{relative}: job `{job}` runs {what}; mutation rows run in {WEEKLY} alone")
        except CannotRead as exc:
            problems.append(str(exc))
    return problems


def check(root):
    """Every way the wiring is wrong, as one line each; [] when it holds."""
    problems = []
    try:
        rows = read_rows(root)
    except CannotRead as exc:
        problems.append(str(exc))
        rows = None
    else:
        problems += row_problems(root, rows)
    return problems + _check_entry(root, rows) + _check_workflows(root, rows)


# --------------------------------------------------------------------- running the rows


def leg_share(rows, env):
    """This leg's rows: every MUTATION_LEGS-th, from the MUTATION_LEG-th. All of them
    when neither is set."""
    leg, legs = env.get("MUTATION_LEG"), env.get("MUTATION_LEGS")
    if leg is None and legs is None:
        return list(rows)
    if not (leg or "").isdigit() or not (legs or "").isdigit() or not 1 <= int(leg) <= int(legs):
        raise CannotRead(f"MUTATION_LEG={leg!r} and MUTATION_LEGS={legs!r} are not a leg 1..N of N")
    share = list(rows)[int(leg) - 1 :: int(legs)]
    if not share:
        raise CannotRead(f"leg {leg} of {legs} has no rows: there are {len(rows)} rows files")
    return share


def _step(argv, what, out):
    out(f"mutation rows: {what}")
    try:
        code = subprocess.run(argv, stdin=subprocess.DEVNULL, timeout=STEP_TIMEOUT).returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CannotRead(f"could not {what}: {exc}") from exc
    if code != 0:
        raise CannotRead(f"could not {what}: exit {code}")


def _sdk_python(workdir, out):
    venv = os.path.join(workdir, "sdk-venv")
    _step([sys.executable, "-m", "venv", venv], "create the SDK venv", out)
    python = os.path.join(venv, "bin", "python")
    for packages in SDK_INSTALLS:
        _step([python, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *packages],
              f"install {' '.join(packages)}", out)
    _step([python, "-c", SDK_PROBE], "import the SDK", out)
    return python


def _say(line):
    print(line, flush=True)


def run(root, *, local, zoo, out=_say):
    """Run every rows file (this leg's share, weekly); 0 green, 1 a row failed, 2 cannot run."""
    try:
        rows = read_rows(root)
        problems = row_problems(root, rows)
        if problems:
            raise CannotRead("; ".join(problems))
        share = list(rows) if local else leg_share(rows, os.environ)
    except CannotRead as exc:
        out(f"mutation rows: cannot run -- {exc}")
        return 2
    with tempfile.TemporaryDirectory(prefix="mutation-rows-") as workdir:
        try:
            zoo = os.path.abspath(zoo)
            if not local and not os.path.isdir(zoo):
                zoo = os.path.join(workdir, "model-zoo")
                _step(["git", "clone", "--depth", "1", "--quiet", ZOO_URL, zoo], f"clone {ZOO_URL}", out)
            python = sys.executable if local else _sdk_python(workdir, out)
        except CannotRead as exc:
            out(f"mutation rows: cannot run -- {exc}")
            return 2
        out(f"mutation rows: {len(share)} of {len(rows)} rows files, zoo {zoo}")
        results = []
        for row in share:
            out(f"\n== {row}")
            started = time.monotonic()
            try:
                code = subprocess.run(
                    [python, row], cwd=root, env={**os.environ, "MODEL_ZOO_PATH": zoo},
                    stdin=subprocess.DEVNULL, timeout=ROW_TIMEOUT,
                ).returncode
            except subprocess.TimeoutExpired:
                code = f"no result after {ROW_TIMEOUT}s"
            results.append((row, code, time.monotonic() - started))
    out("")
    for row, code, seconds in results:
        out(f"  {'ok    ' if code == 0 else 'FAILED'}  {row}  ({seconds:.1f}s, exit {code})")
    failed = [row for row, code, _ in results if code != 0]
    if failed:
        out(f"mutation rows FAILED: {', '.join(failed)}")
        return 1
    out(f"mutation rows: all {len(results)} passed")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rows", action="store_true", help="print the rows files and exit")
    mode.add_argument("--run", action="store_true", help="run every rows file")
    parser.add_argument("--local", action="store_true",
                        help="with --run: this interpreter, no SDK venv, no zoo clone, every leg")
    parser.add_argument("--zoo", default=os.environ.get("MODEL_ZOO_PATH", os.path.join(os.path.dirname(ROOT), "model-zoo")),
                        help="a model-zoo checkout (default: ../model-zoo)")
    args = parser.parse_args(argv)
    if args.local and not args.run:
        parser.error("--local goes with --run")
    if args.run:
        return run(ROOT, local=args.local, zoo=args.zoo)
    if args.rows:
        try:
            print("\n".join(read_rows(ROOT)))
        except CannotRead as exc:
            print(f"cannot tell -- {exc}")
            return 2
        return 0
    problems = check(ROOT)
    for problem in problems:
        print(f"FAIL  {problem}")
    if problems:
        print(f"\nmutation wiring: {len(problems)} problem(s)")
        return 1
    print("OK -- the mutation rows run weekly, in one place, and on no pull request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
