# Contributing to QisDebug

Thanks for considering a contribution. QisDebug is a debugging and observability
tool for Qiskit, and its value comes from being reliable across Qiskit
versions and unobtrusive in real projects — contributions that keep it that
way are especially welcome.

## Ground rules

- Every patch QisDebug installs onto Qiskit internals must check that the
  target still exists before patching it, and degrade with a logged warning
  instead of raising if it doesn't. This applies to any new hook.
- The public API surface should stay small. If you're adding a new
  user-facing function, ask whether it can instead be a method on `Timeline`
  or a `configure()` option.
- Don't require users to change their existing circuit-building or execution
  code. QisDebug attaches passively; it never changes call signatures or return
  values that user code depends on.
- Keep the visualizer decoupled from the instrumentation layer. It should
  only read the public `Timeline` fields.

See [CLAUDE.md](CLAUDE.md) for the full design rationale and build order.

## Getting set up

```bash
git clone https://github.com/kriss-u/QisDebug.git
cd QisDebug
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` sets up the commit-message check and runs pylint and
pyright on each commit, so problems surface locally before CI.

See [INSTALL.md](INSTALL.md) for details on optional extras.

## Development workflow

1. Create a branch off `main`.
2. Make your change.
3. Add or update tests. Each of the three hooks (construction, transpilation,
   execution) has its own test module under `tests/` — new hook behavior
   should be tested there in isolation, since these are the parts most likely
   to break across Qiskit releases.
4. Run the test suite and lints:

   ```bash
   pytest
   pylint src tests
   pyright src tests
   ```

5. If you touched the public API or a hook's behavior, update the relevant
   section of `README.md`.
6. Open a pull request describing what changed and why. Reference the Qiskit
   version(s) you tested against, especially if you touched a hook.

## Commit messages and releases

Commits and pull request titles follow
[Conventional Commits](https://www.conventionalcommits.org/). Releases are
fully automated: every push to `main` runs
[python-semantic-release](https://python-semantic-release.readthedocs.io/),
which reads the commits since the last tag, picks the next version, updates
`CHANGELOG.md`, tags the release, and publishes it to PyPI. Nobody edits the
version number by hand.

| Commit                                   | Effect on version         |
| ---------------------------------------- | ------------------------- |
| `fix: ...`, `perf: ...`                  | patch (`0.1.0` → `0.1.1`) |
| `feat: ...`                              | minor (`0.1.0` → `0.2.0`) |
| `feat!: ...` or a `BREAKING CHANGE:` footer | minor while on `0.x`, major from `1.0` on |
| `docs:`, `chore:`, `refactor:`, `test:`, `ci:`, `build:`, `style:` | no release |

Pull requests are squash-merged, so the PR title becomes the commit on
`main` and is what decides the release; CI rejects titles that don't follow
the convention. Pushing to the `beta` branch publishes prereleases such as
`0.2.0b1`.

## Testing against multiple Qiskit versions

If you're changing a hook, test against both the pinned dev version and the
oldest supported version (`qiskit>=1.0`) if you can, since internals shift
between releases. If a Qiskit internal you're patching against has changed
shape, update the compatibility check for that patch rather than assuming the
new shape is universal.

## Reporting bugs

Open a GitHub issue with:

- The Qiskit, qiskit-aer, and Python versions you're using
- A minimal circuit that reproduces the issue
- What you expected vs. what QisDebug recorded (or the traceback, if it
  crashed)

## Code style

- Follow the conventions already in `src/qisdebug/` (Pydantic models for
  events and config, docstrings on public functions/classes/modules).
- Comments are reserved for non-obvious edge cases or tricky logic; avoid
  comments that restate what a line does.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).
