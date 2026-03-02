# Bug-Agent — Dynamic Tags Reference
#
# This file is loaded into the system prompt on every LLM call.
# Edit freely; changes take effect on the next message processed.
# The agent can also update this file via the `update_tags` tool.
#
# Format:
#   - One tag per line under its category.
#   - Tags are lowercase, hyphen-separated.
#   - The LLM picks the most relevant tags from this list.

## Priority Tags
- critical
- high-priority
- low-priority

## Type Tags
- bug
- crash
- regression
- performance
- ui-glitch
- data-loss
- security
- memory-leak

## Area Tags
- backend
- frontend
- mobile
- api
- database
- auth
- payments
- notifications
- infra
- ci-cd

## Environment Tags
- production
- staging
- dev
- ios
- android
- web

## Status Hint Tags
- needs-repro
- intermittent
- customer-reported
- blocker
