# QDebug

Zero-touch instrumentation and timeline debugging for
[Qiskit](https://www.ibm.com/quantum/qiskit).

QDebug attaches full circuit and execution visibility to your existing Qiskit
code the way OpenTelemetry or Sentry attach to a web framework: one import, one
activation call, and no changes to how you build, transpile, or run circuits.
Everything that happens to a circuit — every gate appended, every transpiler
pass, every statevector along the way, and the final results — is recorded onto
a single per-circuit **timeline** you can inspect, export, or explore
interactively in a notebook.

```python
import qdebug

qdebug.activate()

# --- your existing Qiskit code, completely unchanged -----------------
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2, 2, name="bell")
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

backend = AerSimulator()
counts = backend.run(transpile(qc, backend), shots=1000).result().get_counts()
# ----------------------------------------------------------------------

print(qdebug.timeline(qc).summary())
```

```text
Timeline 'bell' (id=ae87255e3948, 2 qubits, 2 clbits, 42 events)
  gate_added             4
  transpile_started      1
  transpile_pass         31
  transpile_finished     1
  execution_submitted    1
  state_snapshot         3
  execution_finished     1
```

The three state snapshots are the actual simulator statevectors after each gate
— `|00⟩`, then `(|00⟩+|01⟩)/√2`, then the Bell state — captured during the run
you already made, and stripped from the results you see.

## Install

QDebug is currently installed from source; see [INSTALL.md](INSTALL.md) for
full instructions.

```bash
pip install -e .              # core (requires qiskit>=1.0)
pip install -e ".[aer]"       # + statevector capture on Aer
pip install -e ".[viz]"       # + histogram/Bloch-sphere image rendering
pip install -e ".[mcp]"       # + MCP server for AI-agent debugging
pip install -e ".[dev]"       # + test toolchain
pip install -e ".[docs]"      # + Sphinx documentation toolchain
```

## What gets recorded

| Phase         | Event types                                                   | How                                                                                                                                                                                |
| ------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Construction  | `gate_added`                                                  | Wraps `QuantumCircuit.append`/`_append`; records each instruction with qubits, clbits, and params after broadcasting resolves                                                      |
| Transpilation | `transpile_started`, `transpile_pass`, `transpile_finished`   | Wraps `PassManager.run` and injects a per-pass callback through Qiskit's own supported `callback` mechanism                                                                        |
| Execution     | `execution_submitted`, `state_snapshot`, `execution_finished` | Wraps `AerBackend.run` and the Sampler/Estimator primitives; on Aer, runs an instrumented copy with `save_statevector` after every gate and strips the snapshots from your results |
| Diagnostics   | `note`                                                        | QDebug explains itself, e.g. why snapshot capture was skipped for a circuit                                                                                                        |

All three phases land on **one timeline per circuit**: transpiled outputs
inherit the source circuit's hidden id, so the run of a transpiled circuit is
recorded against the circuit you actually wrote.

## Public API

The entire surface is intentionally small:

| Function                | Purpose                                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `qdebug.activate()`     | Attach all instrumentation layers. Idempotent. Returns an `ActivationReport` saying which layers attached |
| `qdebug.deactivate()`   | Restore every original Qiskit method                                                                      |
| `qdebug.is_active()`    | Whether patches are currently applied                                                                     |
| `qdebug.timeline(qc)`   | The `Timeline` for a circuit (or id string); `qdebug.timeline()` returns the most recent                  |
| `qdebug.timelines()`    | All tracked timelines, most recent first                                                                  |
| `qdebug.clear()`        | Drop all recorded telemetry                                                                               |
| `qdebug.configure(...)` | Adjust limits and switches (validated; see below)                                                         |

Importing `qdebug` has **no side effects** — nothing is patched until
`activate()` is called.

### Timeline

```python
tl = qdebug.timeline(qc)
tl.events                 # every event, in true order (monotonic seq numbers)
tl.gate_events            # construction events
tl.transpile_events       # transpilation events
tl.execution_events       # submission / snapshots / completion
tl.snapshots              # state snapshots, each with .amplitudes() / .probabilities()
tl.circuit_at(n)          # rebuild the circuit as it looked after its first n gates
tl.summary()              # compact text overview
tl.export()               # JSON-serializable dict (amplitudes as [re, im] pairs)
tl.show()                 # native GUI window, or a text CLI with no display
```

### Configuration

| Option                      | Default     | Meaning                                                                            |
| --------------------------- | ----------- | ---------------------------------------------------------------------------------- |
| `max_circuits`              | `256`       | Tracked-circuit cap; least recently touched timelines are evicted first            |
| `capture_statevectors`      | `True`      | Insert statevector snapshots on Aer runs                                           |
| `max_statevector_qubits`    | `12`        | Skip capture for wider circuits (snapshot memory grows as 2^n)                     |
| `max_snapshots_per_circuit` | `256`       | Stop inserting snapshots past this count                                           |
| `transpile_snapshots`       | `"changed"` | Keep full circuit snapshots for `"all"` passes, only `"changed"` ones, or `"none"` |

## The timeline viewer

```python
qdebug.timeline(qc).show()
```

Auto-detects a native GUI window (PySide6) when a display is available, or
falls back to a text-based interactive CLI otherwise — force either with
`show(frontend="gui"|"cli")` or the `QDEBUG_FRONTEND` environment variable.
Either way, a slider/stepper walks the full event sequence; each position
shows the circuit as it existed at that moment (drawn with Qiskit's own
drawer), the statevector with amplitudes and probabilities, or per-pass
transpiler statistics. Both frontends read only the public `Timeline`
fields, so the instrumentation and the visualizer can evolve independently.

## Debugging with an AI agent (MCP)

The same telemetry both viewers read is also exposed to coding agents
through a [Model Context Protocol](https://modelcontextprotocol.io) server,
so an agent can debug the *runtime behavior* of your circuits instead of
reasoning about the code blind:

```bash
pip install -e ".[mcp,aer]"
claude mcp add qdebug -- qdebug-mcp     # or the equivalent for your agent
```

The division of labor is deliberate. QDebug does not try to decide what a
quantum program *should* do — the agent does. QDebug extracts the ground
truth and does the arithmetic:

- **Run under instrumentation** — `run_script` executes a script with
  QDebug active and reports every circuit's id, registers, and an anomaly
  scan in one shot. To focus on a suspect region, the agent inserts
  `qdebug.activate()`/`qdebug.deactivate()` around it and passes
  `auto_activate=false`, scoping capture to exactly that window.
- **Extract mid-execution state** — `get_events` (filtered, paged),
  `get_state_snapshot` (raw amplitudes and probabilities at any captured
  step), and `get_circuit_text` (the circuit as it looked after any gate).
- **Watch what matters** — `add_watch` pins qubits, clbits, or registers;
  `read_watches` reports their gate history, marginal P(1), and Bloch
  vector as of any point in the sequence — the same watch feature the GUI
  and CLI expose interactively.
- **Check an expectation** — the agent supplies the state or distribution
  it believes the program should produce; `compare_statevector` and
  `compare_counts` report fidelity, total variation distance, and exactly
  which basis states or outcomes diverge. `find_anomalies` is
  expectation-free and catches only impossible states (non-normalized or
  non-finite statevectors, failed runs, transpile blow-ups).
- **Keep the evidence** — `save_timeline` writes the full JSON export to a
  file; `set_configuration` tunes capture depth before a run.

The server is a thin protocol wrapper: every tool delegates to the same
store, rendering, and comparison code the GUI and CLI use.

## Version compatibility

Every patch target is checked for existence before patching; a missing internal
downgrades that layer with a logged warning instead of an error, and
`activate()` reports exactly which layers attached.

| Python | Qiskit | qiskit-aer | Status                                      |
| ------ | ------ | ---------- | ------------------------------------------- |
| 3.14   | 2.4.2  | 0.17.2     | Tested (full suite)                         |
| ≥3.10  | ≥1.0   | ≥0.14      | Expected to work; layers degrade gracefully |

## Limitations

- **Statevector capture** requires Aer and is skipped (with a `note` event
  explaining why) for circuits containing reset, initialize, mid-circuit
  measurement, control flow, noise instructions, or `parameter_binds`, and for
  circuits wider than `max_statevector_qubits`. Counts and timing are still
  recorded.
- `QuantumCircuit.compose(..., inplace=True)` writes to circuit data without
  passing through any append method, so composed instructions are not recorded
  as gate events.
- Per-pass transpile telemetry applies to single-circuit runs; transpiling a
  list of circuits records start/finish boundaries only (multi-circuit runs may
  fan out to worker processes).
- Primitives (`StatevectorSampler`, Aer `SamplerV2`, …) record submission and
  completion timing; statevector capture happens when a primitive delegates to
  an instrumented Aer backend.
- Circuits built before `activate()` only record events from that point on.
- `qc.copy()` carries the hidden id with it, so a copy reports onto the same
  timeline as its source.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite covers each hook independently (construction, execution,
transpilation), the telemetry store, and the activation lifecycle, including
hypothesis property tests that any random gate sequence is recorded exactly and
the store never exceeds its cap.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
ground rules.

## Documentation

Full API documentation is built with Sphinx:

```bash
pip install -e ".[docs]"
cd docs
make html
```

Open `docs/_build/html/index.html` in a browser.

## Prior art

[Qiskit Trebugger](https://github.com/qiskit-community/qiskit-trebugger)
requires replacing `transpile` with a special debug call and covers only the
transpiler stage. CircInspect provides step-through debugging for PennyLane.
QDebug's distinction is passive, near-zero-setup attachment with full lifecycle
coverage — construction through transpilation to execution — on unmodified
Qiskit code.

## License

MIT
