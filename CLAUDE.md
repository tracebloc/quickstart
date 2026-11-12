<!-- claude-budget: armed -->
<!-- root = commands, the area map, rules every session needs; module rules in `<dir>/CLAUDE.md` or `.claude/rules/` with `paths:`; reference in `docs/`; history in commit messages -->
# CLAUDE.md

The public quickstart: the notebooks that walk a user through a first training
run on tracebloc (`notebooks/`), and the stdlib Python checkers that guard them
(`scripts/`).

## Checks

Three checkers, each with its own mutation harness. `make check` runs the
three checkers; the pre-push hook runs it before every push, and `make setup`
installs that hook in this clone. The mutation rows are the weekly tier, so
`make check` leaves them out; `make mutations` runs them locally:

```sh
make check       # the three checkers
make mutations   # every mutation row, under this interpreter
make setup       # this clone's pre-push hook, which runs `make check`
python3 scripts/check_templates.py                      # D9 template rules
python3 scripts/check_zoo_paths.py --zoo ../model-zoo   # zoo paths resolve
python3 scripts/check_mutation_wiring.py                # where the mutation rows run
make mutation-nightly                                   # the weekly run of every row
```

`check_zoo_paths.py` is the cross-repo one: every `model_zoo/...` path
this repo prints — the guide's table, its `MODEL_PATH` default, the
README — must exist in a `model-zoo` checkout. It needs one beside this
repo (`--clone` fetches it), and it **fails rather than skips** when it
cannot find one.
CI does not run it on your pull request, so run it yourself before you push.

## Map

- `notebooks/`: the training guide (`traceblocTrainingGuide.ipynb`) and the
  retired per-family templates; their rules are in `notebooks/templates/README.md`.
- `scripts/`, `Makefile`, `.github/mutation-coverage.yml`: the checkers and
  their mutation rows; rules in `.claude/rules/checks.md`.

`make mutations` and `make mutation-nightly` read the list from the declaration
through `scripts/check_mutation_wiring.py --run`, so a row file is added
there and nowhere else; the same script, run on every pull request by
mutation-wiring.yml, fails if any other workflow runs a row.
