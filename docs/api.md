# API Reference

The entire public surface is re-exported from the top-level `qdebug` package.

## Activation

```{eval-rst}
.. autofunction:: qdebug.activate
.. autofunction:: qdebug.deactivate
.. autofunction:: qdebug.is_active
.. autoclass:: qdebug.ActivationReport
   :members:
```

## Timelines

```{eval-rst}
.. autofunction:: qdebug.timeline
.. autofunction:: qdebug.timelines
.. autofunction:: qdebug.clear
.. autoclass:: qdebug.Timeline
   :members:
```

## Configuration

```{eval-rst}
.. autofunction:: qdebug.configure
.. autofunction:: qdebug.get_config
.. autoclass:: qdebug.QDebugConfig
   :members:
```

## Events

```{eval-rst}
.. autoclass:: qdebug.EventType
   :members:
.. autoclass:: qdebug.TelemetryEvent
   :members:
.. autoclass:: qdebug.GateEvent
   :members:
.. autoclass:: qdebug.TranspileStartedEvent
   :members:
.. autoclass:: qdebug.TranspilePassEvent
   :members:
.. autoclass:: qdebug.TranspileFinishedEvent
   :members:
.. autoclass:: qdebug.ExecutionSubmittedEvent
   :members:
.. autoclass:: qdebug.StateSnapshotEvent
   :members:
.. autoclass:: qdebug.ExecutionFinishedEvent
   :members:
.. autoclass:: qdebug.NoteEvent
   :members:
```

## Visualizer

```{eval-rst}
.. autofunction:: qdebug.visualizer.show
.. autofunction:: qdebug.visualizer.show_browser
```
