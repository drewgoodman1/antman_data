#!/usr/bin/env zsh
set -euo pipefail

# WASP runner for zsh
# - Activates the project's venv
# - Runs the trading bot (which loads .env via python-dotenv)

PROJECT_DIR="$HOME/workspace/antman_data"
VENV_DIR="$PROJECT_DIR/antman_env"

cd "$PROJECT_DIR"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
else
  echo "Virtualenv not found at $VENV_DIR" >&2
  exit 1
fi

# Unbuffered output so logs stream immediately
export PYTHONUNBUFFERED=1

exec python -u trading/wasp.py
