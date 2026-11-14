#!/usr/bin/env bash
# Run the project's Python from the flake dev shell (nix develop), or directly if already inside it.
root="$(dirname "$(readlink -f "$0")")"
if [ -n "$IN_NIX_SHELL" ] && python -c "import pyarrow, scipy" 2>/dev/null; then
  exec python "$@"
fi
exec nix develop "$root" --command python "$@"
