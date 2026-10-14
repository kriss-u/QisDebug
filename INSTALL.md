# Installing QisDebug

QisDebug is published on [PyPI](https://pypi.org/project/qisdebug/).

## Requirements

- Python >= 3.10
- [Qiskit](https://www.ibm.com/quantum/qiskit) >= 1.0
- `qiskit-aer` >= 0.14, if you want statevector capture during execution
- `matplotlib`, if you want histogram/Bloch-sphere image rendering
- `pywebview`, if you want the native GUI timeline viewer
- `mcp`, if you want the `qisdebug-mcp` server for AI-agent debugging

## From PyPI

A virtual environment is strongly recommended:

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install qisdebug
```

### Optional extras

The core install pulls in `qiskit`, `pydantic`, and the terminal UI
dependencies. Enable extras depending on what you need:

```bash
pip install "qisdebug[aer]"    # statevector capture on Aer
pip install "qisdebug[viz]"    # histogram/Bloch-sphere image rendering
pip install "qisdebug[gui]"    # native GUI timeline viewer
pip install "qisdebug[mcp]"    # MCP server (qisdebug-mcp) for AI-agent debugging
```

Extras can be combined, e.g. `pip install "qisdebug[aer,viz,gui]"`.

## From source

To work on QisDebug itself, or to try unreleased changes, clone the
repository and install it in editable mode:

```bash
git clone https://github.com/kriss-u/QisDebug.git
cd QisDebug
pip install -e ".[dev]"        # test, lint and release toolchain
pip install -e ".[docs]"       # Sphinx toolchain for building documentation
```

## Verifying the install

```bash
qisdebug --version
python -c "import qisdebug; d = qisdebug.QisDebug(); print(d.start().active); d.end()"
```

The second command should print `True` and exit cleanly.

## Running the test suite

```bash
pip install -e ".[dev]"
pytest
```

## Building the documentation locally

```bash
pip install -e ".[docs]"
cd docs
make html
```

Open `docs/_build/html/index.html` in a browser.

## Uninstalling

```bash
pip uninstall qisdebug
```
