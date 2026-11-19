# Contributing to Harbor CLI

Thanks for helping! Bug reports and pull requests are welcome at https://github.com/128x64/harbor-cli.

## Set up
```
git clone https://github.com/128x64/harbor-cli
cd harbor-cli
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --doctor
```

## Run the tests
```
python tests/test_harbor.py
```
The suite is fully offline: it starts mock NIM / Ollama / MCP / GitHub servers, and drives the real program in a pseudo-terminal
whose output is rendered by a tiny terminal emulator (`tests/vt.py`). `python tests/screen_probe.py` prints what the UI looks like.

## Guidelines
- Keep the terminal UI stable: change it only on purpose, and add or adjust a screen test when you do.
- Add a test for every behaviour change. Tests must not touch the network or your real `~/.harbor`.
- Python 3.9+ and the standard library only for the core (optional libraries must degrade gracefully).
- Anything that downloads or runs code needs a safety review (see `harbor/updater.py` for the bar).

## Releasing (maintainers)
Users receive updates from the default branch, so a release is just:
1. Bump `VERSION` in `harbor/__init__.py`.
2. Add a section for it at the top of `CHANGELOG.md` (the update window shows it).
3. Run the tests, then push to the default branch.

Set `"update_channel": "release"` in a user's config to follow GitHub Releases (tags like `v3.5.0`) instead of the default branch.
