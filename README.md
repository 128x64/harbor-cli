# harbor-cli

[![License](https://shields.io)](LICENSE.md)
[![Python Version](https://shields.io)](main.py)

**harbor-cli** is the official command-line interface for managing and interacting with the Harbor Agent infrastructure. Built entirely in Python, it provides developers and operators with an efficient, scriptable toolset to deploy, control, and monitor AI agents seamlessly.

---

##  Features

- **Agent Management**: Complete control over your remote or local AI agent workflows.
- **Structured Project Setup**: Out-of-the-box support for module organization via `nimcode`.
- **Integrated Validation**: Pre-configured test suites to verify agent behavior prior to deployment.
- **Extensible Architecture**: Easy-to-add custom commands and interactive evaluation routines.

---

## Repository Structure

The project follows a clean, modular Python repository layout:

* **`main.py`**: The primary entry point for executing CLI commands.
* **`nimcode/`**: Contains the core business logic and main operational project files.
* **`examples/`**: Useful reference scripts and baseline implementations for target AI models.
* **`tests/`**: Unit and integration test suites (`test.py`) to keep the interface reliable.

---

## Installation

Ensure you have **Python 3.8+** installed on your system. 
Ensure you have explicit permission

1. **Clone the repository:**
   ```bash
   git clone https://github.com/128x64/harbor-cli.git
   cd harbor-cli
   
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To run the Harbor Agent CLI, execute the main entry file from your terminal:

```bash
python main.py 
```

By Placing a model.gguf file in the directory of the main.py the cli will automatically load that model instead of the Nvidia NIM Backend

### Quick Start Example
To initialize or explore a default template run:
```bash
python main.py 
```

---

## Tests

```bash
pytest tests/
```
*Note: If you do not have `pytest` installed, you can execute individual scripts natively via `python tests/test.py`.*


## License

Protected under the terms of the project's license. See [`LICENSE.md`](LICENSE.md) for more details.
