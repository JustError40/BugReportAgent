#!/bin/sh
set -e

# Ensure the state directory exists and is writable by the app user (tgmcp, uid 1000)
mkdir -p /app/state
chown -R tgmcp:tgmcp /app/state

exec gosu tgmcp "$@"
