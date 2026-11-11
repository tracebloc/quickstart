---
paths:
  - "scripts/**"
  - "Makefile"
  - ".github/mutation-coverage.yml"
  - ".github/workflows/mutation-*.yml"
  - ".github/workflows/template-rules.yml"
  - ".github/workflows/zoo-paths.yml"
---
# Checkers and their mutation rows

Each has a `_mutations.py` sibling that mutates a copy
of the tree and asserts every rule is SEEN to fail. Run it after editing
a rule: a checker passing says the tree is clean, only a mutation says
the checker can still fail.

The harnesses are the `rows:` of `.github/mutation-coverage.yml`. CI runs
them only on the weekly schedule (`mutation-nightly.yml`, which runs `make
mutation-nightly` and files a bug when it goes red), never on a pull
request: a checker your pull request breaks is caught by `make mutations`
when you run it after editing a rule, or by the next weekly run. `make mutation-nightly`
installs the SDK into a venv of its own and clones the public zoo when
`ZOO` does not exist.
