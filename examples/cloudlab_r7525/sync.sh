#!/bin/bash
set -euo pipefail

echo "Installing dependencies..."
if ! command -v rsync >/dev/null 2>&1; then
    echo "Installing rsync..."
    sudo apt-get update
    sudo apt-get install -y rsync
fi

REMOTE_HOST="node-2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_DIR="$REPO_ROOT/"
REMOTE_DIR="/mydata/R2CC/"

echo "Ensuring remote directory exists on $REMOTE_HOST:$REMOTE_DIR"
ssh "$REMOTE_HOST" "mkdir -p /mydata/R2CC/examples/cloudlab_r7525"

echo "Syncing $LOCAL_DIR to $REMOTE_HOST:$REMOTE_DIR"
rsync -avzp --delete --perms "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"

echo "Sync completed."
