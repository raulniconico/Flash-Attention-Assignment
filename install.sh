#!/usr/bin/env bash
# Create .venv and install requirements.txt for this project (Linux / macOS / Git Bash).
#
# Usage: ./install.sh              # create or update .venv
#        ./install.sh --recreate   # delete .venv first
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
MIN_MINOR=10

python_ok() {
    # Real Python >= 3.MIN_MINOR (skips the Microsoft Store stub, which exits non-zero).
    "$@" -c "import sys; sys.exit(0 if sys.version_info >= (3, $MIN_MINOR) else 1)" >/dev/null 2>&1
}

PYTHON=()
for cand in python3.12 python3.13 python3.11 python3.10 python3 python "py -3"; do
    read -r -a cmd <<<"$cand"
    if command -v "${cmd[0]}" >/dev/null 2>&1 && python_ok "${cmd[@]}"; then
        PYTHON=("${cmd[@]}")
        break
    fi
done
if [ ${#PYTHON[@]} -eq 0 ]; then
    echo "error: Python >= 3.$MIN_MINOR not found. Install it (https://www.python.org/downloads/," \
         "or on Windows run install.ps1, which can install it via winget), then re-run." >&2
    exit 1
fi
echo "==> using: ${PYTHON[*]} ($("${PYTHON[@]}" --version))"

if [ "${1:-}" = "--recreate" ] && [ -d "$VENV" ]; then
    echo "==> Removing existing .venv"
    rm -rf "$VENV"
fi

if [ -x "$VENV/bin/python" ]; then
    VENV_PY="$VENV/bin/python"
elif [ -x "$VENV/Scripts/python.exe" ]; then
    VENV_PY="$VENV/Scripts/python.exe"
else
    echo "==> Creating virtual environment in .venv"
    "${PYTHON[@]}" -m venv "$VENV"
    if [ -x "$VENV/bin/python" ]; then VENV_PY="$VENV/bin/python"; else VENV_PY="$VENV/Scripts/python.exe"; fi
fi

echo "==> Installing requirements"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r "$ROOT/requirements.txt"

echo "==> Verifying imports"
"$VENV_PY" -c "import numpy, matplotlib, PIL, IPython, ipykernel; print('    ok: numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"

echo
if [ -d "$VENV/bin" ]; then
    echo "Done. Activate with:  source .venv/bin/activate"
else
    echo "Done. Activate with:  source .venv/Scripts/activate"
fi
echo "Run notebooks with:   jupyter notebook   (after activating)"
