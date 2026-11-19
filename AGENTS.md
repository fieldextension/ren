# AGENTS.md

- Setup: `uv sync --locked`. Run Python through `uv run`. Commands are in the `justfile`; run `just` to list them.
- `DEV` overrides backend selection. When unset, Ren selects CUDA if a GPU is available, otherwise Python. Use `Device.DEFAULT` to read the selected device.
- Run `just test` for fast tests or `just test-all` to include slow tests. When CUDA is selected, `just test` includes the native CUDA tests and requires `nvcc` on PATH.
- Run the Python fast tests with `DEV=python just test -n auto --dist worksteal`.
- To run CUDA tests in parallel, use `DEV=cuda just test -n N --dist worksteal`. Workers share the selected GPU, so choose `N` based on available GPU memory.
- Before committing, run `just fix` and `just typecheck`. Use `just check` to check lint, formatting, and Python types.
