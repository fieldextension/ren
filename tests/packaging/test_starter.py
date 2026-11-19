import ast
import email
import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from ren.device import Device

pytestmark = pytest.mark.slow


def run(*args: str, cwd: Path) -> str:
    env = {key: value for key, value in os.environ.items() if key not in {"PYTEST_ADDOPTS", "PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}}
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = Path(__file__).resolve().parents[2]
    dist = tmp_path_factory.mktemp("dist")
    run("uv", "build", str(root), "--out-dir", str(dist), cwd=dist)
    return dist


def locked_ren(project: Path) -> dict[str, str]:
    lock = tomllib.loads((project / "uv.lock").read_text())
    return next(package for package in lock["package"] if package["name"] == "ren")["source"]


def commit(directory: Path, message: str) -> str:
    run("git", "add", ".", cwd=directory)
    run(
        "git",
        "-c",
        "user.name=Ren test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        message,
        cwd=directory,
    )
    return run("git", "rev-parse", "HEAD", cwd=directory).strip()


def assert_installed_revision(project: Path, revision: int):
    code = "import ren, ren_cli; print(ren.PACKAGE_REVISION, ren_cli.PACKAGE_REVISION)"
    assert run("uv", "run", "--locked", "python", "-c", code, cwd=project).strip() == f"{revision} {revision}"


def test_generated_project_runs_from_distribution(tmp_path: Path, distributions: Path):
    root = Path(__file__).resolve().parents[2]
    wheel = next(distributions.glob("ren-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata = email.message_from_bytes(archive.read(next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))))
        assert not any(requirement.startswith("ren-cli") for requirement in metadata.get_all("Requires-Dist", []))
        for source in (root / "ren/backends/cuda/kernels").iterdir():
            if source.suffix in {".cu", ".cuh"}:
                assert archive.read(source.relative_to(root).as_posix()) == source.read_bytes()

    project = tmp_path / "hello-ren"
    project.mkdir()
    cli_env = tmp_path / "cli-env"
    python = str(cli_env / "bin/python")
    run("uv", "venv", "--python", sys.executable, str(cli_env), cwd=tmp_path)
    run("uv", "pip", "install", "--python", python, str(wheel), cwd=tmp_path)
    run(str(cli_env / "bin/ren"), "init", cwd=project)
    assert sorted(item.name for item in project.iterdir()) == [
        ".git",
        ".gitignore",
        ".python-version",
        "README.md",
        "main.py",
        "pyproject.toml",
        "test_main.py",
    ]
    config = tomllib.loads((project / "pyproject.toml").read_text())
    assert config["project"]["dependencies"] == ["ren"]
    assert config["tool"]["uv"]["sources"]["ren"] == {"git": "https://github.com/fieldextension/ren.git", "branch": "main"}
    assert config["project"]["requires-python"] == tomllib.loads((root / "pyproject.toml").read_text())["project"]["requires-python"]
    run("uv", "add", str(wheel), cwd=project)
    assert (project / ".venv/bin/ren").is_file()
    assert "init" in run("uv", "run", "--locked", "ren", "--help", cwd=project)
    selected = run("uv", "run", "--locked", "python", "-c", "from ren.device import Device; print(Device.DEFAULT)", cwd=project)
    assert selected.strip() == Device.DEFAULT
    output = run("uv", "run", "--locked", "main.py", cwd=project)
    decrypted = next(line.split(":", 1)[1] for line in output.splitlines() if line.startswith("decrypted:"))
    assert ast.literal_eval(decrypted) == pytest.approx([3.0, 7.0, 13.0], rel=0, abs=1e-6)
    run("uv", "run", "--locked", "pytest", "-q", cwd=project)
    run("uv", "run", "--locked", "ruff", "check", cwd=project)
    run("uv", "run", "--locked", "ruff", "format", "--check", cwd=project)
    run("uv", "run", "--locked", "ty", "check", cwd=project)
    installed = run("uv", "run", "--locked", "python", "-c", "import ren; print(ren.__file__)", cwd=project).strip()
    assert Path(installed).is_relative_to(project / ".venv")
    assert locked_ren(project) == {"path": str(wheel)}


def test_local_checkout_edits_are_used(tmp_path: Path, distributions: Path):
    with tarfile.open(next(distributions.glob("ren-*.tar.gz"))) as archive:
        archive.extractall(tmp_path, filter="data")
        source = (tmp_path / archive.getnames()[0].split("/")[0]).rename(tmp_path / "local checkout 🐍")
    library = source / "ren/__init__.py"
    cli = source / "ren_cli/__init__.py"
    project = tmp_path / "project"

    for revision in range(2):
        library.write_text(f"{library.read_text()}\nPACKAGE_REVISION = {revision}\n")
        cli.write_text(f"{cli.read_text()}\nPACKAGE_REVISION = {revision}\n")
        if revision == 0:
            run("uv", "run", "--no-dev", "ren", "init", str(project), cwd=source)
            run("uv", "sync", cwd=project)
        code = "import ren, ren_cli; print(ren.PACKAGE_REVISION, ren_cli.PACKAGE_REVISION)"
        assert run("uv", "run", "--no-sync", "python", "-c", code, cwd=project).strip() == f"{revision} {revision}"


def test_git_command_follows_branch(tmp_path: Path, distributions: Path):
    with tarfile.open(next(distributions.glob("ren-*.tar.gz"))) as archive:
        archive.extractall(tmp_path, filter="data")
        source = tmp_path / archive.getnames()[0].split("/")[0]
    run("git", "init", "--quiet", "--initial-branch=main", cwd=source)
    commit(source, "Initial Ren")
    run("git", "switch", "-c", "feature/starter", cwd=source)
    url = f"git+{source.as_uri()}"
    commits = []
    library = source / "ren/__init__.py"
    cli = source / "ren_cli/__init__.py"
    library_code, cli_code = library.read_text(), cli.read_text()

    unrelated = tmp_path / "unrelated"
    (unrelated / "src/unrelated").mkdir(parents=True)
    (unrelated / "src/unrelated/__init__.py").write_text("REVISION = 0\n")
    (unrelated / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["uv_build>=0.13.0,<0.14"]\nbuild-backend = "uv_build"\n'
        '[project]\nname = "unrelated"\nversion = "0.1.0"\nrequires-python = ">=3.14"\n'
    )
    run("git", "init", "--quiet", "--initial-branch=main", cwd=unrelated)
    unrelated_commit = commit(unrelated, "Initial dependency")

    for revision in range(2):
        marker = f"Template revision {revision}\n"
        (source / "ren_cli/templates/starter/README.md").write_text(marker)
        library.write_text(f"{library_code}\nPACKAGE_REVISION = {revision}\n")
        cli.write_text(f"{cli_code}\nPACKAGE_REVISION = {revision}\n")
        commits.append(commit(source, marker.strip()))

        project = tmp_path / f"project-{revision}"
        run("uvx", "--from", f"{url}@feature/starter", "ren", "init", str(project), cwd=tmp_path)
        config = tomllib.loads((project / "pyproject.toml").read_text())
        assert config["project"]["dependencies"] == ["ren"]
        assert config["tool"]["uv"]["sources"]["ren"] == {"git": source.as_uri(), "rev": "feature/starter"}
        assert (project / "README.md").read_text() == marker

        if revision == 0:
            run("uv", "add", f"unrelated @ git+{unrelated.as_uri()}@main", cwd=project)
            assert locked_ren(project)["git"].endswith(f"#{commits[0]}")
            assert_installed_revision(project, 0)

    (unrelated / "src/unrelated/__init__.py").write_text("REVISION = 1\n")
    assert commit(unrelated, "New dependency revision") != unrelated_commit
    project = tmp_path / "project-0"
    before = tomllib.loads((project / "uv.lock").read_text())["package"]
    run("uv", "sync", cwd=project)
    assert locked_ren(project)["git"].endswith(f"#{commits[0]}")
    assert_installed_revision(project, 0)
    run("uv", "sync", "-P", "ren", cwd=project)
    assert locked_ren(project)["git"].endswith(f"#{commits[1]}")
    assert_installed_revision(project, 1)
    after = tomllib.loads((project / "uv.lock").read_text())["package"]
    assert [package for package in before if package["name"] != "ren"] == [package for package in after if package["name"] != "ren"]
    run("uv", "run", "--locked", "pytest", "-q", cwd=project)
    assert (project / "README.md").read_text() == "Template revision 0\n"
    next_project = tmp_path / "next-project"
    run("uv", "run", "--locked", "ren", "init", str(next_project), cwd=project)
    assert (next_project / "README.md").read_text() == "Template revision 1\n"
