import json
import runpy
import shlex
import subprocess
import sys
import tomllib
from importlib.metadata import PathDistribution
from pathlib import Path
from types import SimpleNamespace

import pytest

from ren_cli import main


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def git_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata = tmp_path / "ren-0.1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: ren\nVersion: 0.1.0\nRequires-Python: >=3.14\n")
    (metadata / "direct_url.json").write_text(
        json.dumps({
            "url": "https://github.com/fieldextension/ren.git",
            "vcs_info": {"vcs": "git", "requested_revision": "main", "commit_id": "a" * 40},
        })
    )
    monkeypatch.setattr("ren_cli.distribution", lambda name: PathDistribution(metadata))


def test_init_does_not_import_runtime(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlib.metadata import distribution; "
            "entry, = distribution('ren').entry_points.select(group='console_scripts', name='ren'); "
            "entry.load()(['init', sys.argv[1]]); assert 'ren' not in sys.modules",
            str(tmp_path / "project"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "project/main.py").is_file()
    config = tomllib.loads((tmp_path / "project/pyproject.toml").read_text())
    assert config["project"]["dependencies"] == ["ren"]
    assert config["tool"]["uv"]["sources"]["ren"] == {"path": str(Path(__file__).resolve().parents[1]), "editable": True}


def test_generated_example_honors_device(tmp_path: Path, monkeypatch):
    from ren.context import Context
    from ren.device import Device
    from ren import ckks
    from tests.ckks.params import small_params

    project = tmp_path / "project"
    main(["init", str(project)])
    keygen, devices = ckks.keygen, []

    def selected(params, *, device=None):
        devices.append(Device.canonicalize(device))
        return keygen(small_params(), device="python")

    monkeypatch.setattr(ckks, "keygen", selected)
    with Context(DEV="cuda:3"):
        runpy.run_path(str(project / "main.py"), run_name="__main__")
    assert devices == ["cuda:3"]


def test_init_tracks_main(tmp_path: Path, git_install, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "init.defaultBranch")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "starter")
    project = tmp_path / "My_Project"
    with monkeypatch.context() as runtime:
        runtime.setattr(sys, "version_info", SimpleNamespace(major=3, minor=99))
        main(["init", str(project)])

    config = tomllib.loads((project / "pyproject.toml").read_text())
    assert config["project"]["name"] == "my-project"
    assert config["project"]["dependencies"] == ["ren"]
    assert config["tool"]["uv"]["sources"]["ren"] == {"git": "https://github.com/fieldextension/ren.git", "rev": "main"}
    assert (project / ".python-version").read_text() == (Path(__file__).resolve().parents[1] / ".python-version").read_text()
    assert config["dependency-groups"]["dev"] == ["pytest", "ruff", "ty"]
    ignored = [".venv/", "__pycache__/main.pyc", ".pytest_cache/", ".ruff_cache/"]
    assert git("check-ignore", *ignored, cwd=project).splitlines() == ignored
    assert subprocess.run(["git", "check-ignore", "uv.lock"], cwd=project, capture_output=True).returncode == 1
    assert git("symbolic-ref", "--short", "HEAD", cwd=project) == "starter"
    assert subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=project, capture_output=True).returncode != 0


@pytest.mark.parametrize("arguments", [[], ["."]])
def test_init_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arguments: list[str]):
    project = tmp_path / "My_Project"
    project.mkdir()
    monkeypatch.chdir(project)

    main(["init", *arguments])

    config = tomllib.loads((project / "pyproject.toml").read_text())
    assert config["project"]["name"] == "my-project"
    assert (project / ".git").is_dir()
    output = capsys.readouterr().out
    assert "uv run main.py" in output
    assert "uv run pytest" in output
    assert "cd " not in output


def test_init_prints_usable_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.chdir(tmp_path)
    main(["init", "missing/../parent with spaces/my-project"])

    command = next(line.strip() for line in capsys.readouterr().out.splitlines() if line.startswith("  cd "))
    executable, directory = shlex.split(command)
    assert executable == "cd"
    assert Path(directory).is_dir()
    assert Path(directory).resolve() == tmp_path / "parent with spaces/my-project"


@pytest.mark.parametrize("location", ["repository", "worktree", "subdirectory"])
def test_init_preserves_existing_repository(tmp_path: Path, location: str):
    repository = tmp_path / "repository"
    git("init", "--quiet", str(repository), cwd=tmp_path)
    git("config", "ren.test", "preserve-me", cwd=repository)
    project = repository
    if location == "worktree":
        git(
            "-c",
            "user.name=Ren test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "Empty",
            cwd=repository,
        )
        project = tmp_path / "worktree"
        git("worktree", "add", "--quiet", "--detach", str(project), cwd=repository)
    elif location == "subdirectory":
        project = repository / "examples" / "hello-ren"
    config_before = (repository / ".git/config").read_bytes()
    head_before = (repository / ".git/HEAD").read_bytes()
    git_file = (project / ".git").read_bytes() if location == "worktree" else None

    main(["init", str(project)])

    assert (project / "main.py").is_file()
    assert git("ls-files", cwd=project) == ""
    assert (repository / ".git/config").read_bytes() == config_before
    assert (repository / ".git/HEAD").read_bytes() == head_before
    if git_file is not None:
        assert (project / ".git").read_bytes() == git_file
    if location == "subdirectory":
        assert not (project / ".git").exists()
    top = git("rev-parse", "--show-toplevel", cwd=project)
    assert Path(top).resolve() == (repository if location == "subdirectory" else project).resolve()


@pytest.mark.parametrize("entry", ["main.py", ".env"])
def test_init_preserves_nonempty_directory(tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]):
    project = tmp_path / "project"
    project.mkdir()
    original = project / entry
    original.write_text("keep this")

    with pytest.raises(SystemExit) as error:
        main(["init", str(project)])

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "empty directory" in message
    assert entry in message
    assert list(project.iterdir()) == [original]
    assert original.read_text() == "keep this"


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("location", ["standalone", "repository"])
def test_init_rejects_invalid_git_metadata(tmp_path: Path, kind: str, location: str):
    if location == "repository":
        git("init", "--quiet", str(tmp_path), cwd=tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    metadata = project / ".git"
    if kind == "file":
        metadata.write_text("keep this")
    else:
        metadata.mkdir()

    with pytest.raises(SystemExit) as error:
        main(["init", str(project)])

    assert error.value.code == 2
    assert list(project.iterdir()) == [metadata]
    if kind == "file":
        assert metadata.read_text() == "keep this"
    else:
        assert list(metadata.iterdir()) == []


@pytest.mark.parametrize("name", ["ren", "Ren", "!!!"])
def test_init_rejects_invalid_project_name(tmp_path: Path, name: str, capsys: pytest.CaptureFixture[str]):
    project = tmp_path / name
    with pytest.raises(SystemExit) as error:
        main(["init", str(project)])
    assert error.value.code == 2
    assert "Choose a project name" in capsys.readouterr().err
    assert not project.exists()
