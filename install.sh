#!/usr/bin/env bash
# Install cloud-qmd into a pi agent dir.
#
#   bash install.sh                 # -> ${PI_AGENT_DIR:-~/.pi/agent}/extensions/cloud-qmd
#   PI_AGENT_DIR=/tmp/x bash install.sh
#   bash install.sh --force         # replace an existing install (old copy is backed up)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="${PI_AGENT_DIR:-$HOME/.pi/agent}"
DEST="$AGENT_DIR/extensions/cloud-qmd"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if [[ -e "$DEST" && $FORCE -eq 0 ]]; then
	echo "error: $DEST already exists" >&2
	echo "       re-run with --force to replace it (the old copy is backed up)" >&2
	exit 1
fi

if [[ -e "$DEST" ]]; then
	BACKUP="$DEST.bak.$(date +%Y%m%d%H%M%S)"
	echo "==> backing up the existing install to $BACKUP"
	mv "$DEST" "$BACKUP"
fi

echo "==> installing into $DEST"
mkdir -p "$DEST/lib" "$DEST/test"
cp "$SRC/index.ts" "$DEST/index.ts"
cp "$SRC/lib/engine.mjs" "$DEST/lib/engine.mjs"
cp "$SRC/test/mock-provider.mjs" "$DEST/test/mock-provider.mjs"
cp "$SRC/test/smoke.mjs" "$DEST/test/smoke.mjs"
cp "$SRC/README.md" "$DEST/README.md"

echo "==> self-test"
node "$DEST/test/smoke.mjs" | tail -3

cat <<EOF

installed. next steps:
  1. restart pi (or run /reload) so the extension is loaded — it writes $AGENT_DIR/bin/qmd
     and a config template at $AGENT_DIR/cloud-qmd.json
  2. fill in provider.apiKey in $AGENT_DIR/cloud-qmd.json
  3. run: /cloud-qmd test      (or: qmd __cloud test)
EOF
