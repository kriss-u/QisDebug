# Qiskit Auto Instrumentation Debugger

## Project Overview

This project is a debugging and observability tool for Qiskit. The goal is to
let a developer attach full circuit and execution visibility to their existing
Qiskit code with almost no changes to how they normally write it. The design is
inspired by telemetry libraries in classical software engineering, where a
single import or a small setup call is enough to start capturing detailed traces
without touching the rest of the codebase.

This is meant to be a fully featured tool, not a demo. It should be reliable
enough for other researchers and developers to actually install and use on real
projects, since long term usage is what drives adoption and citations.

The tool tracks:

- Every gate added to a circuit as it is being built
- Circuit structure changes during transpilation
- Quantum state snapshots during execution when supported by the backend
- Execution results and timing information

The end result is a timeline object that captures everything that happened to a
circuit from construction through execution, along with a visualizer to explore
that timeline.

## Goals

- Build a fully working, production quality library, not a one off prototype
- Keep the user facing API surface extremely small, one import and one
  activation call
- Make the tool usable on real Qiskit code without requiring developers to
  change their existing calls
- Support multiple Qiskit versions and fail gracefully when internals change
- Publish as an open source project with clear documentation, tests, and example
  notebooks
- Eventually write a tool paper describing the design, since testing and
  debugging tools tend to get cited by anyone comparing against them

## Core Mechanism

### Why this is possible

Qiskit is regular Python. When a user calls something like qc.h(0), that call
eventually reaches QuantumCircuit.append. Since Python classes are mutable at
runtime, this method can be replaced with a wrapped version that performs the
original behavior and also records what happened. This technique is called
monkey patching. It is the same technique used by telemetry libraries such as
OpenTelemetry or Sentry to instrument web frameworks without requiring changes
to application code.

### Activation

The user should only need:

    debug = qisdebug.QisDebug()
    debug.start()
    ...
    debug.end()

Patching should not happen automatically on import, or on construction of a
`QisDebug()` instance. A library that mutates global state on import is
considered bad practice and can cause issues in larger codebases. Activation
is explicit via `start()`, and `end()` restores every original method this
instance was using.

`QisDebug` is a class, not a module-level singleton: each instance is an
independently scoped instrumentation session with its own telemetry store and
its own `QisDebugConfig`, so more than one can be started at once (e.g. one
scoped to a suspect region, another covering a whole script) without either
seeing the other's timelines or configuration. Everything that happens to
*any* Qiskit circuit between `start()` and `end()` is recorded automatically
by every instance that is currently started -- there is no manual "track this
circuit" call. `start()`/`end()` are flat calls that don't require
re-indenting existing user code, which matters for a near-zero-change tool;
`with QisDebug() as debug:` also works (it just calls `start()`/`end()`) as
an optional convenience for callers who want a scoped block, but it is not
the primary documented pattern.

Although each instance's telemetry is independent, the underlying Qiskit
monkeypatches are shared process-wide and refcounted (`qisdebug/_dispatcher.py`):
a given Qiskit method is patched at most once no matter how many instances are
started, and only unpatched once the last instance using it calls `end()`.
This avoids two instances patching the same class and racing each other's
restore. Each patched wrapper looks up the instances that are currently
started and fans the event it just observed out to each one's own store.

If a `QisDebug` instance is garbage collected while still started (`end()`
was never called), it emits a `ResourceWarning` rather than raising or
silently leaking -- the boundary should be enforced, but a debugging tool
should never crash the program it's attached to.

Identity is Qiskit's own: circuits, registers, and their `.name` are used
directly wherever a human-facing key is needed (`debug.get("my_circuit")`,
`debug.circuits()`). There is no separate qisdebug-invented naming scheme for
these; the hidden per-circuit id used internally to correlate a circuit
across construction, transpilation and execution (see below) is
implementation plumbing only, never part of the public surface.

Retrieval never returns `None` for a miss: `debug.get(target)` raises
`qisdebug.TimelineNotFoundError` when `target` (a circuit object, a
`Timeline`, or a name) isn't tracked, so callers never need a manual
`assert timeline is not None` -- if something should have been tracked and
wasn't, that is a real failure and QisDebug raises it, rather than the
caller re-deriving the check. The same principle applies inside the library
itself: don't use bare `assert` for anything that isn't a pure,
never-externally-triggerable type-narrowing hint; raise a real exception
instead so failures survive `python -O` and are catchable.

### Layer one, circuit construction hook

Save a reference to the original QuantumCircuit.append method, then replace it
with a wrapper, installed at most once process-wide regardless of how many
`QisDebug` instances are started (see Activation above). The wrapper calls the
original method first so behavior does not change, then records the
instruction, the qubits involved, and a timestamp into every currently started
instance's own telemetry store. Each circuit instance gets a hidden unique id
attribute the first time it is touched, which becomes the key into each
store -- the same id string is reused across every instance's store for a
given circuit, so different instances' `Timeline`s for the same circuit stay
correlated without those stores knowing about each other directly.

### Layer two, execution hook

The run method lives on the backend object or, in newer Qiskit versions, on
Primitives classes such as Sampler and Estimator. Patch the run method there.
The wrapper intercepts the circuit before it is sent to the backend. If the
backend supports state snapshots, such as Aer, the wrapper creates a temporary
copy of the circuit with save_statevector instructions inserted between every
gate, runs that instrumented copy instead of the original, and strips the extra
instructions from the results the user actually sees. The user experience does
not change, but the full state evolution is captured behind the scenes.
Because the patch is shared, one physical backend call may need to serve
several independently configured instances at once; the instrumentation
decision (whether to capture at all, and the qubit/snapshot limits) uses the
widest requirement among instances that want capture, and each instance still
applies its own configured policy when deciding what to keep on its own
events (`qisdebug/_hooks/execution.py` docstring has the full rationale).

### Layer three, transpiler hook

Qiskit provides a supported extension mechanism for injecting custom passes
through the pass manager. Add a custom pass automatically during activation.
This pass runs after every stage of transpilation and records a snapshot of the
circuit at that point. Since this uses Qiskit's own extension system rather than
raw monkey patching, this layer is the most stable and least likely to break
across versions.

### Telemetry store

Each `QisDebug` instance owns one `TelemetryStore` -- there is no process-wide
singleton. All three layers write into that per-instance structure, per
circuit. Each entry is an event with a type such as gate added, transpile
pass completed, or execution finished, a timestamp, and the relevant data for
that event. This store is what `debug.timelines()`/`debug.get()` and the
visualizer read from.

### Memory management

Since real programs build circuits in loops, each store needs a way to avoid
growing forever. Cap the number of tracked circuits per instance
(`QisDebugConfig.max_circuits`), evict the oldest ones once the cap is
reached, and give the user a manual `debug.clear()`.

### Version compatibility

Qiskit changes internal APIs across releases. Every patch should check that the
target method or class exists before patching it, and log a clear warning
instead of crashing if something has changed. Maintain a compatibility table in
the documentation listing which Qiskit versions have been tested.

### Visualizer

There is no HTML/webserver/browser based viewer, and none should be
reintroduced. Everything is shown through the `QisDebug` instance itself, not a
separate visualizer object: `debug.show()` shows every timeline the instance
has recorded, `debug.show(qc)` shows just one circuit's timeline, and
`debug.show([qc1, qc2])` shows several merged into one view (each argument
form accepts a live circuit, a circuit name, or a `Timeline`, resolved the same
way as `debug.get()`). `QisDebug.show()` presents timelines through one of two
interchangeable frontends, chosen by a factory:

- A native GUI window (PySide6, whose wheels bundle the Qt binaries they
  need, so `pip install "qisdebug[gui]"` is enough on its own with no extra
  dependency), when a display is available, including inside a notebook
- A text based interactive CLI, the universal fallback for environments with
  no display, such as an SSH session or a notebook with no display

Detection happens automatically: display availability first, then CLI.
Either can be forced explicitly, either with `debug.show(frontend="gui"|"cli")`
or the `QISDEBUG_FRONTEND` environment variable, which is useful for scripts
that should always behave the same way regardless of where they run.
Forcing a frontend that is not actually available (for example `"gui"` with
no display) raises a `RuntimeError` describing why, rather than crashing
lower down or hanging; CLI never fails to be forced since it has no
environment requirements.

This is implemented as a factory/strategy pattern in `qisdebug/_frontends/`: an
abstract `TimelineFrontend` base class defines `is_available()`,
`unavailable_reason()`, and `show(timelines)`, and each of the two frontends
(`gui.py`, `cli.py`) overrides the same interface. Frontend-agnostic data
prep (merging timelines, drawing circuits as text, formatting statevectors,
rendering histograms/Bloch spheres as images or ASCII bar charts) lives in
`_frontends/_render.py` so each frontend only handles presentation, not data
wrangling. `_frontends/detect.py` holds the environment detection and the
`get_frontend()` factory function.

Any feature that touches how a timeline is presented (a new panel, a new
piece of interactive state such as watches, a new keyboard/command
shortcut) must be implemented for both frontends, not just the one being
tested against. Put the frontend-agnostic computation in
`_frontends/_render.py` once, then wire the presentation and any
interaction (buttons, dropdowns, `cmd` commands, ...) into `gui.py` and
`cli.py` so the two stay equivalent in capability, each in its own idiom.

### Agent interface (MCP)

Alongside the two human-facing frontends, an MCP server in `qisdebug/_mcp/`
exposes the same telemetry to coding agents, so an agent can debug the
runtime behavior of circuits instead of reasoning about code blind. One
`QisDebug` instance, held module-level in `_mcp/tools.py`, scopes the whole
server process -- matching the fact that MCP tools are plain functions, not
objects handed back to the caller. The division of labor is deliberate:
qisdebug does not try to decide what a quantum program should do. The agent
scopes instrumentation (`run_script` with the default `auto_activate=true`
brackets the whole script with the server's instance; `auto_activate=false`
leaves that instance untouched, for cases where the target script builds its
own scoped `QisDebug()` and reports its own findings instead), chooses what
to watch, supplies the expected state or distribution, and interprets
discrepancies; qisdebug extracts ground truth (events, mid-execution
statevectors, circuit drawings, watch values) and does the comparison
arithmetic (`qisdebug/_analysis.py`: fidelity, total variation distance, plus
an expectation-free anomaly scan for impossible states).

The layering rule matches the frontend rule: `_mcp/tools.py` holds plain,
protocol-free query functions that delegate to `_store`, `_events`,
`_frontends/_render.py`, and `_analysis.py`; `_mcp/server.py` only
registers those functions with the protocol layer and is the sole module
that imports the `mcp` package. New agent-facing capability goes into
`_analysis.py` or `_render.py` (computation) and `tools.py` (query
surface), never hand-rolled in `server.py`. Features shared with the
frontends (e.g. watches) must reuse the same `_render.py` logic, not
reimplement it. The server ships as the `qisdebug-mcp` console script behind
the `mcp` extra.

## Planned Features

The next round of work, on top of the three-layer instrumentation, watches,
and comparison/anomaly helpers already built. Each item lists its Python,
MCP, and frontend surface together, since the project's rule is that a new
capability isn't done until all three exist.

### Property-set diffing

Qiskit's pass manager callback already hands `_hooks/transpile.py::_record_pass`
a `property_set` alongside `pass_`, `dag`, and `time`; today only the latter
three are used. Snapshot `property_set` into a plain `{key: repr}` dict after
each pass (values like a `Layout` or `TranspileLayout` aren't JSON-native, so
this needs a small best-effort serializer that falls back to `repr()` and
never raises), diff it against the previous pass's snapshot the same way
`_fingerprint` already detects a changed circuit, and attach the added/
removed/changed keys to `TranspilePassEvent` as `property_changes`. Governed
by a new `QisDebugConfig.capture_property_set: Literal["all", "changed",
"none"]` (default `"changed"`), mirroring `transpile_snapshots` so the two
knobs read the same way.

- **Python**: no new method needed -- `property_changes` rides along on the
  existing `Timeline`/event export.
- **MCP**: a new `get_property_set(circuit_id, seq)` in `_mcp/tools.py` for
  the full snapshot at one pass; `get_events`/`get_timeline` pick up
  `property_changes` automatically.
- **Frontends**: append a "Property changes" section to the per-pass event
  detail text in `_frontends/_render.py::render_event`, the same place
  watches and extra visuals are already appended in `gui.py`/`cli.py`.

### Conditional breakpoints

Neither Trebugger nor CircInspect can pause a *live* run for inspection;
qisdebug's MCP layer is built for exactly that, so it's worth doing properly
rather than approximating it as "run then look." A breakpoint is a small
closed set of predicate kinds -- never arbitrary `eval`'d code, so MCP tool
arguments stay plain JSON and there is no code-injection surface: a gate
name/qubit match, a `seq` threshold, a watch value crossing a numeric
threshold (reusing `WatchTarget` from `_frontends/_render.py`), or "the next
event `find_anomalies` would flag." `QisDebug.add_breakpoint(condition)`/
`remove_breakpoint(handle)` register these on an instance; `TelemetryStore.
add_event` -- the one choke point every hook already funnels through --
checks live breakpoints against each event as it lands and blocks the
calling thread on a `threading.Event` when one matches, so a breakpoint
pauses user code exactly between the instructions that triggered it,
regardless of which of the three hooks recorded the event.

- **Python**: `debug.add_breakpoint(condition)` / `debug.remove_breakpoint(handle)`;
  `debug.wait_for_breakpoint(timeout=None)` blocks the caller's thread until
  the next hit (for a caller running the target script on its own thread);
  `debug.resume()` releases the block.
- **MCP**: `run_script` gains a `background=True` mode -- since it runs
  synchronously inside one tool call today, a script that can hit a
  breakpoint needs to run on a worker thread instead, returning a `run_id`
  immediately. `poll_run(run_id)` reports running/paused/finished plus the
  triggering event; `resume_run(run_id)` continues; `stop_run(run_id)`
  aborts. This is the same synchronous-vs-backgrounded split `run_script`'s
  `auto_activate` flag already draws, just one level deeper.
- **Frontends**: breakpoints double as a navigation aid over an already
  captured `Timeline` -- "jump to next hit" reuses the same
  condition-matching function (implemented once, likely
  `_frontends/_breakpoints.py` next to `_render.py`) against recorded events
  instead of blocking anything live. `cli.py` gets a `break` command; `gui.py`
  gets a condition-builder row next to the existing watch row. Live-pause and
  jump-to-event are different operations, but there is exactly one
  definition of a breakpoint condition shared between them.

### Timeline diff

`_analysis.py` gets a pure `diff_timelines(a: Timeline, b: Timeline) ->
TimelineDiff`, following the same "computation lives here, presentation
lives in the caller" split as `compare_statevector`/`compare_counts`. It
aligns the two timelines' `gate_added` events with a standard sequence diff
(insert/delete/replace, keyed on gate name + qubits) so a construction-code
refactor that reordered equivalent gates reads clearly instead of as a wall
of mismatches, and separately reuses `compare_statevector`/`compare_counts`
on each timeline's final snapshot/counts when both have one, plus a
depth/size delta pulled from each timeline's last transpile event.

- **Python**: `debug.diff(a, b)`, a thin wrapper with no logic of its own.
- **MCP**: `diff_timelines(circuit_id_a, circuit_id_b, atol=...)` in
  `_mcp/tools.py`, same thin-wrapper pattern every other MCP tool follows.
- **Frontends**: a diff is a variant of the existing merged multi-timeline
  view, not a separate window -- `debug.show([a, b], diff=True)` runs the two
  timelines through `_render.merge_timelines` exactly as
  `debug.show([qc1, qc2])` already does for side-by-side tracks, then uses
  `diff_timelines`'s gate alignment to highlight the differing lines with the
  same span-highlight mechanism `_TrackTextEdit`/CLI rendering already use
  for the active-event outline, plus a one-line fidelity/TVD/depth-delta
  summary header. One window, one merged view; diff mode is an extra
  highlight layer and header line on top of what already exists, not a
  second code path to maintain in parallel.

### Visual polish

Both the GUI's buttons/labels and the CLI's prompt/output are currently
unstyled defaults (`QPushButton`/`QLabel` with no stylesheet in `gui.py`; the
CLI leans on raw `print()`). Before shipping the three features above: a
small QSS stylesheet in `gui.py` (spacing, button hover/pressed states, a
consistent type scale between the header and the monospace timeline body)
that respects the light/dark palette rather than hardcoding colors, the way
`_active_track_border_color` already does correctly for the highlight
border; and for the CLI, consistent prompt styling and column alignment for
tabular output (event lists, watch reports) using `shutil.get_terminal_size()`
rather than fixed-width formatting. This is presentation-only -- no
`_render.py` changes -- so it can land independently of the three features
above.

## Build Order

1. Circuit construction hook and the telemetry store
2. Execution hook with statevector capture on Aer
3. Transpiler pass hook
4. Version compatibility checks and graceful degradation
5. Environment-detecting visualizer factory (native GUI / text CLI),
   forceable via argument or environment variable
6. Test suite covering all three hooks, plus each visualizer frontend,
   independently
7. Agent interface: comparison/anomaly helpers in `_analysis.py` and the
   MCP server exposing telemetry extraction, watches, and scoped runs
8. Documentation, example notebooks, and README
9. Property-set diffing, conditional breakpoints, and timeline diff (see
   Planned Features above), each landed across the Python API, MCP, and
   both frontends together, plus the GUI/CLI visual polish pass
10. Tool paper draft describing the design and comparing it against existing
    tools like Trebugger and CircInspect

## Engineering Conventions

- Python, targeting current and recent Qiskit versions
- Keep the public API to as few functions as possible
- Never require the user to change their existing circuit building or execution
  code
- Write tests for each hook independently, since these are the most likely parts
  to break across Qiskit updates
- Keep the visualizer decoupled from the instrumentation layer so either piece
  can be replaced independently
- Every patch must check that the thing it is patching still exists before
  patching it
- All code must pass pylint and pyright clean before being considered done.
  When either flags something, find and fix the actual defect the warning is
  pointing at; do not silence it with a `# pylint: disable`/`# type: ignore`,
  a broadened type, or a rename/rewrite that only exists to dodge the
  checker. A suppression is only acceptable when the warning is a genuine
  false positive, and even then it needs a comment explaining why.

## Non Goals

- Multi agent coordination features are out of scope for this project
- Hardware specific debugging beyond what Aer and standard simulators expose is
  out of scope for the first version
- Supporting SDKs other than Qiskit is out of scope for the first version

## Prior Art

Qiskit Trebugger requires replacing the normal transpile call with a special
debug call, and only covers the transpiler stage. CircInspect supports
breakpoints and step through debugging but is built for PennyLane, not Qiskit.
The OpenQASM VS Code extension supports manual breakpoints. None of these attach
automatically and passively the way this project is designed to. That
distinction, near zero setup and full lifecycle coverage from construction
through transpilation to execution, is the core contribution worth highlighting
in any future paper about this tool.
