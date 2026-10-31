# quickstart: the notebooks' checkers, their mutation rows, and the pre-push hook.
# CI runs template-rules.yml on a pull request; zoo-paths.yml runs on develop pushes
# and nightly, so `make check` is the only pre-merge run of the zoo check. The mutation
# rows (`rows:` in .github/mutation-coverage.yml) run in CI only on the weekly schedule,
# as mutation-nightly.yml's `make mutation-nightly`. `make check` runs them as well:
# they take seconds, and catch a checker you broke before you push.
#
#   make check             the three checkers, then every mutation row
#   make mutation-nightly  every mutation row, with the SDK installed: the weekly run
#   make setup             install this clone's pre-push hook, which runs `make check`
#   make install-hooks     the same hook alone; `make hooks-status` says whether it is current
#
# The checkers are stdlib Python. check_templates.py cross-checks the SDK when
# `tracebloc` is importable and prints a note when it is not; CI installs it and sets
# TRACEBLOC_CHECK_STRICT=1. check_zoo_paths.py needs a model-zoo checkout: ZOO is the
# sibling clone by default, and the check fails, never skips, without one;
# `make mutation-nightly` clones the public zoo when ZOO does not exist.
# check_mutation_wiring.py --run reads the rows from the declaration, so no list of
# them is kept here.

PYTHON ?= python3
ZOO ?= ../model-zoo
WIRING = $(PYTHON) scripts/check_mutation_wiring.py

.DEFAULT_GOAL := check
.PHONY: check templates zoo-paths wiring mutations mutation-nightly setup

check: templates zoo-paths wiring mutations

templates:
	$(PYTHON) scripts/check_templates.py

zoo-paths:
	$(PYTHON) scripts/check_zoo_paths.py --zoo $(ZOO)

wiring:
	$(WIRING)

# This interpreter as it is: without the SDK, the rows that need it skip, and say so.
mutations:
	$(WIRING) --run --local --zoo "$(ZOO)"

mutation-nightly:
	$(WIRING) --run --zoo "$(ZOO)"

# setup: nothing to install but the hook; the checkers need only python3.
setup:
	@$(MAKE) --no-print-directory install-hooks

# The one pre-push installer, generated: `make install-hooks` writes the hook that runs
# `make check`, `make hooks-status` says whether it is current. Never edit
# .github/install-hooks.mk here; it is rendered from the org's canonical copy.
include .github/install-hooks.mk
