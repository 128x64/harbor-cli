---
name: release-notes
description: How to write release notes for this project from git history
---
1. Run `git log --oneline <last-tag>..HEAD` (find the last tag with `git describe --tags --abbrev=0`).
2. Group commits into: Features, Fixes, Internal. Drop merge commits and trivial chores.
3. Write one line per item in plain language, newest-first within each group, with the PR/issue number if present.
4. Put the result in CHANGELOG.md under a new `## <version> - <date>` heading. Do not rewrite older entries.
