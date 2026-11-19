<h1 align="center">Harbor CLI</h1> <p align="center"><b>An agentic coding assistant for your terminal that works with any model.</b><br> NVIDIA NIM &middot; Ollama &middot; a local GGUF file &middot; any OpenAI-compatible API</p> <p align="center"> <a href="https://github.com/128x64/harbor-cli">github.com/128x64/harbor-cli</a> </p>

Harbor CLI reads your project, edits files, runs commands, checks its own work and reports back - like the AI coding agents you may know, but you choose the model and everything runs from a plain Python install: no Node, no Docker.

Why Harbor CLI
Any model. Free NVIDIA NIM models, a model running in Ollama, a .gguf file on your disk, or OpenAI, Anthropic (compat layer), Gemini, OpenRouter, Groq, Together, DeepSeek, Mistral, xAI, LM Studio, vLLM ... Tool calling works natively or, for models that lack it, through a built-in prompt-based protocol.
Agentic. Plans with a todo list, calls tools in parallel, delegates to sub-agents, verifies with your tests, and has an autonomous mode that keeps going until the job is done.
In your control. Manual, accept-edits, plan (read-only) and autonomous modes - always shown under the input box. Every edit and command can be reviewed, destructive commands always ask, and /undo reverts file edits.
Batteries included. About 50 tools (files, PDF/Office reading, shell, git, web search, Python, SQLite, vision, browser), MCP servers (including the ones you set up in VS Code), custom sub-agents, skills, hooks and permission rules.
Keeps itself up to date. On start it checks this repository and, if there is a newer version, installs it through a small installer window and restarts - with backups and automatic rollback. See Automatic updates.
Windows, macOS and Linux. Works with PowerShell, uses the git bundled with GitHub Desktop when Git itself is missing.
What it looks like
text

<!-- \<PLACEHOLDER  -->

An example session (illustrative):

text
> add a --verbose flag to the CLI and make sure the tests still pass

● I'll look at how the arguments are parsed first.

● Search(parser.add_argument)
  ⎿  Found 12 matches

● Read(src/cli.py)
  ⎿  Read 180 lines

● Update(src/cli.py)
  ⎿  Updated src/cli.py with 4 additions and 0 removals
      41  +    parser.add_argument("-v", "--verbose", action="store_true",
      42  +                        help="print extra diagnostic output")

● Bash(python -m pytest -q)
  ⎿  42 passed in 1.9s

● Added `--verbose` (src/cli.py:41) and verified: all 42 tests pass.

## Install

You need Python 3.9 or newer. Get the code with git clone https://github.com/128x64/harbor-cli or download the ZIP from the green Code button on GitHub.

### Windows

Install Python from `python.org` and tick Add python.exe to PATH.
Open a terminal in the `harbor-cli` folder and run `install.bat`.
Start it from any project folder with `path\to\harbor-cli\harbor.bat` (add the folder to your PATH to just type harbor).

### macOS / Linux

bash```
./install.sh
.venv/bin/python main.py          # run it from your project folder```
Check your setup at any time with `harbor --doctor`.

### Choose a model

| You want | Do this |
| :--- | :--- |
| **A free hosted model:** | Get a key at [https://build.nvidia.com](https://build.nvidia.com), then `setx NVIDIA_API_KEY "nvapi-..."` (Windows) or `export NVIDIA_API_KEY=...` |
| **A model on your machine (Ollama):** | `ollama pull qwen2.5-coder:7b`, then `harbor --backend ollama` |
| **A GGUF file:** | Put `model.gguf` next to `main.py` (plus `llama-server.exe` from the llama.cpp releases); it is used automatically |
| **OpenAI, OpenRouter, Groq, Gemini, LM Studio ...:** | `harbor --provider openrouter` (key in `OPENROUTER_API_KEY`), `harbor --provider lmstudio`, ... |
Local models are loaded into memory at start-up, before you type your first message.

### Using it

Type what you want in plain English: "find why the build fails and fix it", "add tests for utils.py", "explain how auth works".


| Key | Action |
| :--- | :--- |
| `Shift+Tab` / `Ctrl+T` | Switch mode: manual, accept edits, plan, autonomous |
| `Esc` | Interrupt the running response |
| `/` then `Tab` | Commands (`/help`, `/model`, `/scan`, `/undo`, `/review`, `/commit`, `/update` ...) |
| `@` then `Tab` | Mention a file |
| `!command` | Run a shell command directly |


### Useful flags:
1. `--auto` (autonomous)
2. `--plan` (read-only)
3. `--yolo` (approve everything except destructive commands)
4. `--verify "pytest -q"` (a command that must pass before the task counts as done)
5. `-p "task"` (one-shot, scriptable)
6. `-c` (continue the last session).

### What it can do
1. Code: read, search, edit, multi-edit; run shell commands and background processes; Python and SQLite; git (read-only commands run freely).
2. Documents and the web: read PDF, Word, Excel, PowerPoint; Bing web search, page fetch, HTTP requests, downloads.
3. Senses: look at images and screenshots with a vision model, OCR, clipboard, a headless browser (optional).
4. Teamwork: research sub-agents in parallel, worker sub-agents, your own agents and skills.
5. Repo scanner: harbor --scan <folder | GitHub URL | owner/repo> profiles a project - languages, entry points, dependencies, tests, CI, licence - and flags possible secrets (always redacted), risky code and missing basics. No model needed.
6. MCP: uses the MCP servers from VS Code (.vscode/mcp.json), Claude Desktop, Cursor or ~/.harbor/mcp.json. Servers started with npx need Node.js and docker ones need Docker; uvx / python -m servers work out of the box.
7. Automatic updates

Every time Harbor CLI starts in an interactive terminal it looks at this repository (harbor/__init__.py on the default branch). If the version there is newer than yours:

1. Harbor CLI stops and opens the installer window (a console installer if Tkinter or a display is missing).
2. The window shows the changelog and counts down; you can click Not now or Skip this version.
3. It downloads the new version, verifies it, backs up the old files, installs, runs a self-test, and starts Harbor CLI again with the same arguments.

### What makes this safe:

It talks to GitHub over HTTPS only, and the archive must contain Harbor CLI's own files and announce the version it carries.
Archives are size-limited and checked for path tricks and symlinks. Nothing is executed except the self-test of the new version.
Only Harbor CLI's files are replaced. Your models (*.gguf), llama-server, .venv, notes and settings are never touched.
The old files are backed up to ~/.harbor/backups (last three kept) and restored automatically if the self-test fails.
Git clones and pip installs are never overwritten - you get a hint (git pull) instead. Scripts and CI runs (-p, --scan) never check.

## Updates

Harbor checks for updates automatically by default. You can customize this behavior using configuration options, environment variables, or CLI flags.

### Configuration (`~/.harbor/config.json`)

```json
{
  "auto_update": "auto",
  "update_repo": "owner/repo",
  "update_channel": "release"
}
```

* **`auto_update`**: Controls update behavior. Accepts `"auto"` (default), `"ask"`, or `"off"`.
* **`update_repo`**: Points the updater to a custom repository (e.g., `"owner/repo"`), useful for forks.
* **`update_channel`**: Set to `"release"` to track GitHub Releases instead of the default branch.

### CLI Commands & Overrides

| Command / Flag | Description |
| :--- | :--- |
| `harbor --check-update` | Display the current update status. |
| `harbor --update` (or `/update`) | Trigger an immediate update. |
| `--no-update-check` | Disable the update check for a single run. |
| `HARBOR_NO_UPDATE=1` | Disable update checks via environment variable. |


Project instructions go in a `HARBOR.md` file in your repository; they are read at the start of every session. More examples: `examples/` (custom agents, skills, tools, MCP config, hooks).

### Safety

Edits and commands ask first (a diff is shown); "don't ask again" is scoped to a command prefix and never covers chained commands. Destructive-looking commands (rm -rf /, git push --force, curl | sh, format C: ...) always ask, even with --yolo. Plan mode exposes read-only tools only. Project-level extensions (.mcp.json, .harbor/tools, agents, skills) load only after you trust the project once. API keys are read from environment variables and never printed. Read harbor/updater.py for exactly what the updater does.

### Project layout
text
main.py            entry point (python main.py)
harbor/            the application (agent loop, tools, UI, backends, MCP, updater)
tests/             offline test suite with mock servers and a terminal emulator
examples/          sample config, agents, skills, tools, MCP config
install.bat|.sh    one-step installers        harbor.bat   Windows launcher
CHANGELOG.md       release notes (shown in the update window)
Development
bash
python tests/test_harbor.py        # fully offline

See `CONTRIBUTING.md`. Releases are pushed to the default branch: bump VERSION in harbor/__init__.py, add a section to CHANGELOG.md, and users get it the next time they start Harbor CLI.

### Status and limitations
Tested on Linux with mock servers. The Windows-specific code (PowerShell runner, key handling, GitHub Desktop git lookup, installer window) is covered by simulated tests but has had less real-world use - please report problems.
Quality depends on the model: large hosted models do very well; small local models need more guidance.
Bing search works by reading Bing's result page and can break if Bing changes it.

License

See `LICENSE.md`.
