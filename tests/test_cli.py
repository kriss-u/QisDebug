"""The ``qisdebug`` console script: Typer wiring and its underlying logic."""

from typer.testing import CliRunner

from qisdebug._cli import commands
from qisdebug._cli.app import app

runner = CliRunner()


def test_app_help_exits_zero():
    """--help lists both commands without needing a running instrumentation session."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "version" in result.stdout


def test_app_version_flag_prints_version_and_exits():
    """-V is eager: it short-circuits before any command body runs."""
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "qisdebug" in result.stdout


def test_app_version_command_reports_layers_and_extras():
    """The version command reports real activation results, not guesses."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "qiskit" in result.stdout
    assert "construction: yes" in result.stdout


def test_run_script_captures_timelines(tmp_path):
    """run_script brackets the target script and reports what it tracked."""
    script = tmp_path / "toy.py"
    script.write_text(
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(1, name='toy')\n"
        "qc.h(0)\n"
    )
    result = commands.run_script(str(script))
    assert result.error is None
    assert [t.name for t in result.timelines] == ["toy"]
    assert result.active_names == ["construction", "execution", "transpile"]


def test_run_script_reports_the_exception_without_raising(tmp_path):
    """A failing script's error is captured, not propagated out of run_script."""
    script = tmp_path / "bad.py"
    script.write_text("raise RuntimeError('boom')\n")
    result = commands.run_script(str(script))
    assert isinstance(result.error, RuntimeError)
    assert not result.timelines


def test_app_run_exits_nonzero_on_script_error(tmp_path):
    """The CLI surfaces a failing script as a nonzero exit, not silent success."""
    script = tmp_path / "bad.py"
    script.write_text("raise RuntimeError('boom')\n")
    result = runner.invoke(app, ["run", str(script)])
    assert result.exit_code == 1
    assert "boom" in result.output


def test_collect_version_info_reflects_a_real_activation_cycle():
    """VersionInfo is built from an actual start()/end() cycle, not importability guesses."""
    info = commands.collect_version_info()
    assert info.qisdebug_version
    assert info.activation.layers["construction"] is True
    assert set(info.extras) == {"aer", "gui", "mcp"}
