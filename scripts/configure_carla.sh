#!/usr/bin/env bash
# Make a CARLA 0.9.15 install start on the task map.
#
# The packaged build is reliable when it starts on the layered Town03_Opt map,
# and CarDreamer's WorldManager then reuses it for Town03 tasks instead of
# calling load_world(). This sets the startup, default and transition maps in
# CarlaUE4/Config/DefaultEngine.ini. It is idempotent and keeps a one-time
# backup of the original file.
#
# Usage:
#   bash scripts/configure_carla.sh [/path/to/carla]   (default: $CARLA_ROOT)
#   CARLA_DEFAULT_MAP=Town04_Opt bash scripts/configure_carla.sh

set -euo pipefail

CARLA_DIR="${1:-${CARLA_ROOT:-}}"
MAP="${CARLA_DEFAULT_MAP:-Town03_Opt}"

if [[ -z "$CARLA_DIR" ]]; then
  echo "ERROR: pass the CARLA directory or set CARLA_ROOT." >&2
  exit 1
fi
INI="$CARLA_DIR/CarlaUE4/Config/DefaultEngine.ini"
if [[ ! -f "$INI" ]]; then
  echo "ERROR: not found: $INI" >&2
  exit 1
fi
if [[ ! -e "$CARLA_DIR/CarlaUE4/Content/Carla/Maps/$MAP.umap" ]]; then
  echo "ERROR: map $MAP is not part of this CARLA install." >&2
  exit 1
fi

[[ -f "$INI.orig" ]] || cp "$INI" "$INI.orig"

MAP_PATH="/Game/Carla/Maps/$MAP.$MAP"
for key in EditorStartupMap GameDefaultMap ServerDefaultMap TransitionMap; do
  if ! grep -q "^$key=" "$INI"; then
    echo "ERROR: $key is missing from $INI" >&2
    exit 1
  fi
  sed -i "s|^$key=.*|$key=$MAP_PATH|" "$INI"
done

echo "CARLA default maps set to $MAP in $INI:"
grep -E '^(EditorStartupMap|GameDefaultMap|ServerDefaultMap|TransitionMap)=' "$INI" | sed 's/^/  /'
