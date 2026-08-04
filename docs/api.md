# API Reference

The entire public surface is re-exported from the top-level `qisdebug` package.

## Activation

```{eval-rst}
.. autofunction:: qisdebug.activate
.. autofunction:: qisdebug.deactivate
.. autofunction:: qisdebug.is_active
.. autoclass:: qisdebug.ActivationReport
   :members:
```

## Timelines

```{eval-rst}
.. autofunction:: qisdebug.timeline
.. autofunction:: qisdebug.timelines
.. autofunction:: qisdebug.clear
.. autoclass:: qisdebug.Timeline
   :members:
```

## Configuration

```{eval-rst}
.. autofunction:: qisdebug.configure
.. autofunction:: qisdebug.get_config
.. autoclass:: qisdebug.QisDebugConfig
   :members:
```

## Events

```{eval-rst}
.. autoclass:: qisdebug.EventType
   :members:
.. autoclass:: qisdebug.TelemetryEvent
   :members:
.. autoclass:: qisdebug.GateEvent
   :members:
.. autoclass:: qisdebug.TranspileStartedEvent
   :members:
.. autoclass:: qisdebug.TranspilePassEvent
   :members:
.. autoclass:: qisdebug.TranspileFinishedEvent
   :members:
.. autoclass:: qisdebug.ExecutionSubmittedEvent
   :members:
.. autoclass:: qisdebug.StateSnapshotEvent
   :members:
.. autoclass:: qisdebug.ExecutionFinishedEvent
   :members:
.. autoclass:: qisdebug.NoteEvent
   :members:
```

## Visualizer

```{eval-rst}
.. autofunction:: qisdebug.visualizer.show
```
