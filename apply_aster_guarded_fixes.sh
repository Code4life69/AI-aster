#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-.}"
MODE="${2:-}"

if [[ "$MODE" == "--dry-run" ]]; then
  python apply_aster_guarded_fixes.py "$REPO" --dry-run
else
  python apply_aster_guarded_fixes.py "$REPO"
fi
