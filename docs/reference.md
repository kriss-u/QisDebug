# Internals reference

Every module in the package, private ones included, documented from its
docstrings. These are implementation details: they can change in any
release without notice, so build against the {doc}`api` instead. They're
here for contributors, and for anyone tracing how a hook, the telemetry
store, a frontend, or the MCP server works. The pages are generated at
build time by walking the package, so a new module appears here on the next
build.

```{toctree}
:maxdepth: 1
:glob:

reference/*
```
