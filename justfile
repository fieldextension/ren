# List contributor commands
default:
    @just --list --list-submodules

# Run tests on the selected backend, excluding slow tests
[positional-arguments]
test *args:
    uv run --group test pytest -m "not slow" "$@"

# Run the full test suite on the selected backend
[positional-arguments]
test-all *args:
    uv run --group test pytest "$@"

# Check lint and formatting
lint:
    uv run --group lint ruff check .
    uv run --group lint ruff format --check .
    just _clang-format --dry-run --Werror

# Apply lint fixes and format Python and native sources
fix:
    uv run --group lint ruff check --fix .
    uv run --group lint ruff format .
    just _clang-format -i

[positional-arguments]
_clang-format +args:
    #!/usr/bin/env sh
    set -eu
    formatter="${CLANG_FORMAT:-$(command -v clang-format-23 || command -v clang-format)}" || {
        echo "clang-format 23 is required. Install it or set CLANG_FORMAT to a version 23 executable." >&2
        exit 1
    }
    version=$("$formatter" --version)
    case "$version" in
        *"clang-format version 23."*) ;;
        *) echo "clang-format 23 is required; found: $version. Set CLANG_FORMAT to a version 23 executable." >&2; exit 1 ;;
    esac
    find ren tests \( -name '*.c' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \) ! -name doctest.h -exec "$formatter" "$@" {} +

# Run the type checker
typecheck:
    uv run --group typecheck --group test ty check

# Check lint, formatting, and Python types
check: lint typecheck
