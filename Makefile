# quickstart: the notebooks' two checkers and the pre-push hook that runs them.
# CI runs template-rules.yml on a pull request; zoo-paths.yml runs on develop pushes
# and nightly, so `make check` is the only pre-merge run of the zoo check.
#
#   make check          both checkers, each after its mutation harness
#   make setup          install this clone's pre-push hook, which runs `make check`
#   make install-hooks  the same hook alone; `make hooks-status` says whether it is current
#
# The checkers are stdlib Python. check_templates.py cross-checks the SDK when
# `tracebloc` is importable and prints a note when it is not; CI installs it and sets
# TRACEBLOC_CHECK_STRICT=1. check_zoo_paths.py needs a model-zoo checkout: ZOO is the
# sibling clone by default, and the check fails, never skips, without one.

PYTHON ?= python3
ZOO ?= ../model-zoo

.DEFAULT_GOAL := check
.PHONY: check templates zoo-paths setup

check: templates zoo-paths

templates:
	$(PYTHON) scripts/check_templates_mutations.py
	$(PYTHON) scripts/check_templates.py

zoo-paths:
	$(PYTHON) scripts/check_zoo_paths_mutations.py --zoo $(ZOO)
	$(PYTHON) scripts/check_zoo_paths.py --zoo $(ZOO)

# setup: nothing to install but the hook; the checkers need only python3.
setup:
	@$(MAKE) --no-print-directory install-hooks

# The one pre-push installer, generated: `make install-hooks` writes the hook that runs
# `make check`, `make hooks-status` says whether it is current. Never edit
# .github/install-hooks.mk here; it is rendered from the org's canonical copy.
include .github/install-hooks.mk
