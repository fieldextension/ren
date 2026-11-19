import argparse
import json
import os
import re
import shlex
import subprocess
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
from string import Template


def init_project(directory: Path) -> Path:
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("Choose a new or empty directory.")
    directory = directory.resolve()
    if directory.exists():
        entries = sorted(item.name for item in directory.iterdir() if item.name != ".git")
        if entries:
            raise ValueError(f"Choose a new or empty directory. {directory} contains: {', '.join(entries)}")
    name = re.sub(r"[-_.]+", "-", directory.name).lower()
    if name == "ren" or not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", name):
        raise ValueError("Choose a project name containing letters, numbers, hyphens, underscores, or dots, other than 'ren'.")

    metadata = json.loads(distribution("ren").read_text("direct_url.json") or "{}")
    source: dict[str, str | bool]
    if "vcs_info" in metadata:
        source = {"git": metadata["url"]}
        if revision := metadata["vcs_info"].get("requested_revision"):
            source["rev"] = revision
    elif "dir_info" in metadata:
        source = {"path": str(Path.from_uri(metadata["url"])), "editable": True}
    else:
        source = {"git": "https://github.com/fieldextension/ren.git", "branch": "main"}

    contents = {
        item.name.removesuffix(".in"): item.read_text(encoding="utf-8")
        for item in files("ren_cli").joinpath("templates", "starter").iterdir()
        if item.is_file()
    }
    contents["pyproject.toml"] = Template(contents["pyproject.toml"]).substitute(
        name=json.dumps(name),
        source=", ".join(f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in source.items()),
    )
    directory.mkdir(parents=True, exist_ok=True)
    git = ["git"]
    if (directory / ".git").exists(follow_symlinks=False):
        git.append("--git-dir=.git")
    repository = subprocess.run(
        [*git, "rev-parse", "--is-inside-work-tree"],
        cwd=directory,
        capture_output=True,
        text=True,
        env=os.environ | {"LC_ALL": "C"},
    )
    if repository.returncode:
        if (directory / ".git").exists(follow_symlinks=False) or not repository.stderr.startswith("fatal: not a git repository"):
            raise ValueError(repository.stderr.strip())
        subprocess.run(["git", "init", "--quiet"], cwd=directory, check=True)
    elif repository.stdout.strip() != "true":
        raise ValueError(f"{directory} is not a Git working tree.")
    for filename, content in contents.items():
        with (directory / filename).open("x", encoding="utf-8") as output:
            output.write(content)
    return directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ren")
    init = parser.add_subparsers(required=True).add_parser("init", help="Create a Ren project")
    init.add_argument("directory", type=Path, nargs="?", default=Path("."), help="Project directory (default: current directory)")
    args = parser.parse_args(argv)
    try:
        directory = init_project(args.directory)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Created {directory}\n")
    if directory != Path.cwd():
        print(f"  cd {shlex.quote(str(directory))}")
    print("  uv run main.py\n  uv run pytest")
