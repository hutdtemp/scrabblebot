#!/usr/bin/env bash
# Regenerate src/module_bindings from the upstream module schema.
#
# Preferred route (coordinator's isolated Docker runtime):
#   CONTAINER=scrabble-local-e99a53b4 ./scripts/generate-bindings.sh
#
# Fallback route (standalone official image; requires the upstream module's
# dist/bundle.js to exist):
#   UPSTREAM=/path/to/upstream ./scripts/generate-bindings.sh
set -euo pipefail

CONTAINER=${CONTAINER:-scrabble-local-e99a53b4}
UPSTREAM=${UPSTREAM:-/home/zlare/docs/src/clockworklabs_scrabblebot/main}
BOT_DIR=${BOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
IMAGE=${IMAGE:-clockworklabs/spacetime:v2.12.0}

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "generating via container $CONTAINER" >&2
  docker exec --user "$(id -u):$(id -g)" -e HOME=/task/home "$CONTAINER" \
    spacetime --config-path /task/cli.toml generate \
    --js-path /source/spacetimedb/dist/bundle.js --lang rust \
    --out-dir /bot/src/module_bindings --no-config -y
else
  echo "generating via standalone $IMAGE" >&2
  mkdir -p "$BOT_DIR/src/module_bindings"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$UPSTREAM":/upstream:ro \
    -v "$BOT_DIR":/bot \
    "$IMAGE" generate \
    --js-path /upstream/spacetimedb/dist/bundle.js --lang rust \
    --out-dir /bot/src/module_bindings --no-config -y
fi

# The official image may not include rustfmt. Use the project's edition and
# the host's existing Rust formatter; never hand-edit generated bindings.
rustfmt --edition 2024 "$BOT_DIR"/src/module_bindings/*.rs

echo "bindings written to $BOT_DIR/src/module_bindings" >&2
