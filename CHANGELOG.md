# Changelog

Harbor CLI shows the newest section of this file in its update window, so keep the latest version first.

## 3.4.0
- **Automatic updates.** On start Harbor CLI checks this repository for a newer version; if there is one it stops, opens an
  installer window, installs it and starts again. Settings: `auto_update` (`auto`/`ask`/`off`), `update_repo`, `update_channel`.
  Also `harbor --update`, `harbor --check-update` and `/update`.
- Installer window built with Tkinter; falls back to a console installer when Tk or a display is not available.
- Updates are verified (GitHub-only HTTPS, zip-slip checks, version match), back up what they replace, run a self-test and roll
  back automatically if it fails. Your models, `.venv` and settings are never touched.
- `harbor.bat` is now safe to replace while Harbor CLI is running.
- README rewritten for people visiting the repository; new `.gitignore`, `CONTRIBUTING.md`.

## 3.3.0
- Renamed from NimCode to **Harbor CLI** (command `harbor`). Old settings, folders and `NIMCODE_*` variables keep working.
- New repository scanner: `harbor --scan <folder | GitHub URL | owner/repo>`, `/scan` and the `scan_repo` tool.

## 3.2.1
- Model preload before the first message, live token counter, instant Shift+Tab mode switching (also on Windows).

## 3.2.0
- Provider presets (OpenAI-compatible services), model profiles, custom sub-agents, skills, permission rules, plan approval,
  `--output-format json`, `/context`, `/resume`, `/export`, built-in prompt commands.

## 3.1.0
- Ollama backend, visible permission modes, capability-aware system prompt.

## 3.0.0
- Local `model.gguf` support, VS Code-style MCP configuration, autonomous mode, hooks, undo.
