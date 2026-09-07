# Installing QisDebug

QisDebug is not yet published to PyPI. For now, install from source.

## Requirements

- Python >= 3.10
- [Qiskit](https://www.ibm.com/quantum/qiskit) >= 1.0
- `qiskit-aer` >= 0.14, if you want statevector capture during execution
- `matplotlib`, if you want histogram/Bloch-sphere image rendering
- `pywebview`, if you want the native GUI timeline viewer
- `mcp`, if you want the `qisdebug-mcp` server for AI-agent debugging

## From source

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/nullvector/qisdebug.git
cd qisdebug
pip install -e .
```

A virtual environment is strongly recommended:

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -e .
```

### Optional extras

The core install only pulls in `qiskit` and `pydantic`. Enable extras
depending on what you need:

```bash
pip install -e ".[aer]"        # statevector capture on Aer
pip install -e ".[viz]"        # histogram/Bloch-sphere image rendering
pip install -e ".[gui]"        # native GUI timeline viewer
pip install -e ".[mcp]"        # MCP server (qisdebug-mcp) for AI-agent debugging
pip install -e ".[dev]"        # test toolchain (pytest, hypothesis, pylint)
pip install -e ".[docs]"       # Sphinx toolchain for building documentation
```

Extras can be combined, e.g. `pip install -e ".[aer,viz,dev]"`.

## Verifying the install

```bash
python -c "import qisdebug; qisdebug.activate(); print(qisdebug.is_active())"
```

This should print `True` and exit cleanly.

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
