#!/usr/bin/env bash
set -euo pipefail

# Fail on compiler and Python warnings; do not hide diagnostics.
export PYTHONWARNINGS=error
export CFLAGS="${CFLAGS:-} -Werror"
export CXXFLAGS="${CXXFLAGS:-} -Werror"

if [[ $# -eq 0 ]]; then
    exec python -m contextgru --help
fi

exec python -m contextgru "$@"
