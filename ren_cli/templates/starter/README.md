# Ren project

This is a [uv](https://docs.astral.sh/uv/) project. Use uv to install its dependencies so it resolves Ren from the source in `pyproject.toml`.

CUDA execution requires Linux, an NVIDIA GPU, and CUDA Toolkit 12.9.2 or newer.

```bash
uv run main.py
```

Ren selects CUDA when available, otherwise the Python reference backend. To choose explicitly, set `DEV=python` or `DEV=cuda`; use `DEV=cuda:1` to select a specific GPU.

The example computes `x**2 + x + 1` on encrypted inputs. The decrypted result should be close to `[3.0, 7.0, 13.0]`. It uses the default CKKS parameters with bootstrapping disabled.

To change the computation, edit `circuit()` in `main.py` and update the expected result in `test_main.py`. The surrounding code generates keys, encrypts the inputs, and decrypts the result. The circuit runs with public keys and does not need the secret key. All steps run in one process.

The inputs use a public magnitude bound of `4`. Choose this bound independently of private input values.

Test, lint, format, and type-check:

```bash
uv run pytest
uv run ruff check
uv run ruff format
uv run ty check
```

Ren's `main` branch may introduce breaking API changes. If you installed Ren from Git, commit `uv.lock` to pin the revision. To update Ren:

```bash
uv sync -P ren
```

See the [Ren README](https://github.com/fieldextension/ren#readme) for an overview of the library.
