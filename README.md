# QisDebug

Zero-touch instrumentation and timeline debugging for
[Qiskit](https://www.ibm.com/quantum/qiskit).

QisDebug attaches full circuit and execution visibility to your existing Qiskit
code the way OpenTelemetry or Sentry attach to a web framework: one import, one
activation call, and no changes to how you build, transpile, or run circuits.
Everything that happens to a circuit — every gate appended, every transpiler
pass, every statevector along the way, and the final results — is recorded onto
a single per-circuit **timeline** you can inspect, export, or explore
interactively in a notebook.

```python
import qisdebug

debug = qisdebug.QisDebug()
debug.start()

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

print(debug.get(qc).summary())
debug.end()
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

QisDebug is currently installed from source; see [INSTALL.md](INSTALL.md) for
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
| Transpilation | `transpile_started`, `transpile_pass`, `transpile_finished`   | Wraps `PassManager.run` and injects a per-pass callback through Qiskit's own `callback` mechanism; `transpile_pass` carries a `property_changes` diff too                          |
| Execution     | `execution_submitted`, `state_snapshot`, `execution_finished` | Wraps `AerBackend.run` and the Sampler/Estimator primitives; on Aer, runs an instrumented copy with `save_statevector` after every gate and strips the snapshots from your results |
| Diagnostics   | `note`                                                        | QisDebug explains itself, e.g. why snapshot capture was skipped for a circuit                                                                                                      |

All three phases land on **one timeline per circuit**: transpiled outputs
inherit the source circuit's hidden id, so the run of a transpiled circuit is
recorded against the circuit you actually wrote.

## Public API

`QisDebug` is a class, not a module-level singleton: each instance is an
independently scoped instrumentation session with its own telemetry and its own
configuration, so more than one can be started at once (e.g. one scoped to a
suspect region, another covering a whole script) without either seeing the
other's timelines. The surface on it is intentionally small:

| Method                  | Purpose                                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `debug.start()`         | Attach all instrumentation layers. Idempotent. Returns an `ActivationReport` saying which layers attached |
| `debug.end()`           | Restore every original Qiskit method this instance was using                                              |
| `debug.active`          | Whether this instance's patches are currently applied                                                     |
| `debug.get(qc)`         | The `Timeline` for a circuit, name, or `Timeline`; raises `TimelineNotFoundError` on a miss               |
| `debug.timelines()`     | Every timeline this instance has recorded, most recently touched first                                    |
| `debug.circuits()`      | The names of tracked circuits, most recently touched first                                                |
| `debug.clear()`         | Drop all recorded telemetry without touching active patches                                               |
| `debug.configure(...)`  | Adjust limits and switches (validated; see below)                                                         |
| `debug.diff(a, b, ...)` | Gate-alignment diff plus statevector/counts/depth deltas between two points in time; see below            |

Importing `qisdebug` has **no side effects** — nothing is patched until
`start()` is called, and `with QisDebug() as debug:` is an optional shorthand
for `start()`/`end()` around a scoped block.

### Timeline

```python
tl = debug.get(qc)
tl.events                 # every event, in true order (monotonic seq numbers)
tl.gate_events            # construction events
tl.transpile_events       # transpilation events
tl.execution_events       # submission / snapshots / completion
tl.snapshots              # state snapshots, each with .amplitudes() / .probabilities()
tl.circuit_at(n)          # rebuild the circuit as it looked after its first n gates
tl.summary()              # compact text overview
```

Exporting timelines to JSON-serializable dicts (amplitudes as `[re, im]` pairs)
goes through the `QisDebug` instance too, with the same argument shapes as
`show()`. It always returns a list, even for a single circuit, so callers never
need to branch on the shape of the result:

```python
debug.export()              # every timeline this instance has recorded, as a list of dicts
debug.export(qc)             # just this circuit's timeline, as a one-item list
debug.export([qc1, qc2])     # several circuits, as a list of dicts
```

### Diffing two points in time

```python
diff = debug.diff(qc_a, qc_b)          # or debug.diff(qc, qc, seq_a=10, seq_b=40)
diff.gate_diff                          # aligned insert/delete/replace over gate_added events
diff.statevector_diff, diff.counts_diff # fidelity / total variation distance, when both sides have one
diff.depth_a, diff.depth_b              # depth/size deltas from each side's last transpile event
```

`a`/`b` take the same circuit/name/`Timeline` shapes as `debug.get()`;
`seq_a`/`seq_b` bound either side to "as of that event's `seq`" instead of the
whole timeline, so the same circuit can be diffed against an earlier moment of
its own replay. The interactive frontends (below) expose the same comparison as
"Mark A"/"Mark B" over the live timeline view.

### Configuration

| Option                      | Default     | Meaning                                                                            |
| --------------------------- | ----------- | ---------------------------------------------------------------------------------- |
| `max_circuits`              | `256`       | Tracked-circuit cap; least recently touched timelines are evicted first            |
| `capture_statevectors`      | `True`      | Insert statevector snapshots on Aer runs                                           |
| `max_statevector_qubits`    | `12`        | Skip capture for wider circuits (snapshot memory grows as 2^n)                     |
| `max_snapshots_per_circuit` | `256`       | Stop inserting snapshots past this count                                           |
| `transpile_snapshots`       | `"changed"` | Keep full circuit snapshots for `"all"` passes, only `"changed"` ones, or `"none"` |
| `capture_property_set`      | `"changed"` | Diff/snapshot property_set for `"all"` passes, only `"changed"` ones, or `"none"`  |

## The timeline viewer

Everything is shown through the `QisDebug` instance itself:

```python
debug.show()               # every timeline this instance has recorded
debug.show(qc)              # just this circuit's timeline
debug.show([qc1, qc2])      # several circuits, merged into one view
```

Auto-detects a native GUI window (HTML/CSS rendered via pywebview) when a display is available, or falls
back to a text-based interactive CLI otherwise — force either with
`debug.show(frontend="gui"|"cli")` or the `QISDEBUG_FRONTEND` environment
variable. Either way, a slider/stepper walks the full event sequence; each
position shows the circuit as it existed at that moment (drawn with Qiskit's own
drawer), the statevector with amplitudes and probabilities, or per-pass
transpiler statistics. Both frontends read only the public `Timeline` fields, so
the instrumentation and the visualizer can evolve independently.

Pin the event currently on screen as one side of a diff — "Mark A"/"Mark B" in
the GUI, `mark a`/`mark b` in the CLI — and once both are pinned, the diff
between them (the same comparison `debug.diff()` returns) renders automatically:
a per-gate diff, a side-by-side circuit-drawing split view, a summary across
every element common to both marked points, and a diff of any currently active
watch.

## Debugging with an AI agent (MCP)

The same telemetry both viewers read is also exposed to coding agents through a
[Model Context Protocol](https://modelcontextprotocol.io) server, so an agent
can debug the _runtime behavior_ of your circuits instead of reasoning about the
code blind:

```bash
pip install -e ".[mcp,aer]"
claude mcp add qisdebug -- qisdebug-mcp     # or the equivalent for your agent
```

The division of labor is deliberate. QisDebug does not try to decide what a
quantum program _should_ do — the agent does. QisDebug extracts the ground truth
and does the arithmetic:

- **Run under instrumentation** — `run_script` executes a script with QisDebug
  active and reports every circuit's id, registers, and an anomaly scan in one
  shot. To focus on a suspect region, the agent has the script build its own
  scoped `qisdebug.QisDebug()` and call `start()`/`end()` around it, passing
  `auto_activate=false` so this server's own instance stays untouched.
- **Extract mid-execution state** — `get_events` (filtered, paged),
  `get_state_snapshot` (raw amplitudes and probabilities at any captured step),
  `get_circuit_text` (the circuit as it looked after any gate), and
  `get_property_set` (the full layout/routing property_set a specific transpiler
  pass wrote — `get_events` surfaces the added/removed/changed keys inline as
  `property_changes`).
- **Watch what matters** — `add_watch` pins qubits, clbits, or registers;
  `read_watches` reports their gate history, marginal P(1), and Bloch vector as
  of any point in the sequence — the same watch feature the GUI and CLI expose
  interactively.
- **Compare two points in time** —
  `diff_timelines(circuit_id_a, circuit_id_b,
  seq_a=..., seq_b=...)` reports
  the same gate-alignment plus statevector/counts/depth diff as `debug.diff()`,
  for two different circuits or two moments of the same circuit's replay.
- **Check an expectation** — the agent supplies the state or distribution it
  believes the program should produce; `compare_statevector` and
  `compare_counts` report fidelity, total variation distance, and exactly which
  basis states or outcomes diverge. `find_anomalies` is expectation-free and
  catches only impossible states (non-normalized or non-finite statevectors,
  failed runs, transpile blow-ups).
- **Keep the evidence** — `save_timeline` writes the full JSON export to a file;
  `set_configuration` tunes capture depth before a run.

The server is a thin protocol wrapper: every tool delegates to the same store,
rendering, and comparison code the GUI and CLI use.

## Conditional breakpoints

Everything above is passive: it tells you what already happened. Conditional
breakpoints pause a **live** run the instant a condition matches, so you can
inspect state mid-execution instead of only replaying a finished timeline.
See [`examples/breakpoints.py`](examples/breakpoints.py) for a runnable,
end-to-end example (`python examples/breakpoints.py`).

A condition is one of a small closed set of kinds — never arbitrary code, so
there's no `eval`-style injection surface:

| Kind      | Matches when...                                                           |
| --------- | ------------------------------------------------------------------------- |
| `gate`    | A named gate (optionally on specific qubits) is added to a circuit        |
| `seq`     | Any event reaches a given sequence number                                 |
| `watch`   | A watched qubit/clbit/register's value (P(1)) crosses a numeric threshold |
| `anomaly` | The next event `find_anomalies` would flag, at or above a chosen severity |

### Python API

```python
import qisdebug
from qisdebug import GateMatchCondition, WatchThresholdCondition

debug = qisdebug.QisDebug()
debug.start()

handle = debug.add_breakpoint(GateMatchCondition(gate="cx"))
# or: debug.add_breakpoint(WatchThresholdCondition(
#         circuit_id=circuit_id, watch_kind="qubit", ref="0",
#         comparator="gt", threshold=0.9))

# On the thread building/running the circuit, execution blocks the instant
# the breakpoint matches. From another thread (or the one driving the run):
hit = debug.wait_for_breakpoint(timeout=10)   # -> BreakpointHit | None
print(hit.condition, hit.event_summary)
debug.resume()                                 # release the paused thread

debug.remove_breakpoint(handle)
debug.list_breakpoints()
```

### MCP (agent-driven stepping)

Live pausing needs the script to run in the background, since a pause blocks the
thread — `run_script(..., background=True)` returns immediately with a `run_id`
instead of waiting for the script to finish:

```text
add_breakpoint({"kind": "gate", "gate": "cx"})
run_script("script.py", background=True)      # -> {"run_id": "..."}
poll_run(run_id)                                # -> running | paused | finished | error | stopped
# when paused, poll_run's response includes which condition hit and on what event
resume_run(run_id)
stop_run(run_id)                                # best-effort; guaranteed only while paused
```

### Interactive frontends

Breakpoints also work as a navigation aid over an **already-captured** timeline
— "jump to next hit" walks recorded events instead of blocking anything live:

- **CLI**: `break gate cx`, `break seq 40`, `break watch qubit 0 gt 0.9`,
  `break anomaly warning`; `breakpoints` lists them, `unbreak <handle>` removes
  one, and `continue` jumps to the next matching event.
- **GUI**: a condition-builder row (kind dropdown + argument field, same syntax
  as the CLI) next to the watch row, a list of active breakpoints, and a "Run to
  next breakpoint →" button.

## Roadmap

Planned next, on top of what's above — see `CLAUDE.md` for the full design:

- **Visual polish** — a proper stylesheet for the GUI (buttons, spacing, type
  scale) and consistent prompt/column formatting for the CLI; both are currently
  unstyled toolkit defaults.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and ground
rules.

## Documentation

Full API documentation is built with Sphinx:

```bash
pip install -e ".[docs]"
cd docs
make html
```

Open `docs/_build/html/index.html` in a browser.

## Prior art

[Qiskit Trebugger](https://github.com/TheGupta2012/qiskit-timeline-debugger)
requires replacing `transpile` with a special debug call and covers only the
transpiler stage. CircInspect provides step-through debugging for PennyLane.
QisDebug's distinction is passive, near-zero-setup attachment with full
lifecycle coverage — construction through transpilation to execution — on
unmodified Qiskit code.

## License

MIT
