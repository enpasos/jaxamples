#!/usr/bin/env bash

set -euo pipefail

export POETRY_NO_INTERACTION=1
export POETRY_VIRTUALENVS_IN_PROJECT=true

ruff_args=()
if [[ "${1:-}" == "--fix" ]]; then
    ruff_args+=(--fix)
    shift
fi

if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--fix]" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "[ci-local] poetry install"
poetry install

echo "[ci-local] poetry check"
poetry check

echo "[ci-local] compileall"
poetry run python -m compileall jaxamples tests

echo "[ci-local] ruff"
poetry run ruff check "${ruff_args[@]}" jaxamples tests

echo "[ci-local] pytest"
poetry run pytest -q
