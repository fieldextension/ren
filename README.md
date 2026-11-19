# REN

Ren is an engine for computing on encrypted data without decrypting it.

We want you to write circuits the way you think about them, using familiar Python syntax to operate on ciphertexts. Ren implements [CKKS](https://eprint.iacr.org/2016/421), a fully homomorphic encryption (FHE) scheme for approximate arithmetic on encrypted numbers. Its compiler lowers the underlying polynomial ring operations to GPU kernels.

Ren's design is heavily based on [tinygrad](https://github.com/tinygrad/tinygrad), and we share its preference for a small, simple codebase with few dependencies. We want every layer to be easy to understand and change, from the CKKS implementation down to the GPU kernels.

CUDA is the first supported target, with a Python reference backend for testing and small experiments.

Ren is in active development and we haven't relied on it in a high-stakes production setting.

## What we're exploring

We're especially interested in computations that don't happen today because people can't or don't want to share their data, even with people they trust.

We build applications to explore what becomes possible when people can contribute to a shared computation without revealing their individual inputs, and what it takes to make those computations practical. We use what we learn to improve the math and software underneath them.

## Getting started

You'll need:

- Git and [uv](https://docs.astral.sh/uv/)
- For CUDA: Linux and CUDA Toolkit >=12.9.2

```bash
uvx --from 'git+https://github.com/fieldextension/ren.git@main' ren init my-project
cd my-project
uv run main.py
uv run pytest
```

The starter runs the [example below](#example) with default CKKS parameters and bootstrapping disabled. To try a different computation, edit `circuit()` in `main.py` and update the expected result in `test_main.py`.

Ren selects CUDA when available, otherwise the Python reference backend. To choose explicitly, set `DEV=python` or `DEV=cuda`; use `DEV=cuda:1` to select a specific GPU.

## Example

The client encrypts three numbers. The evaluator computes `x**2 + x + 1` with public keys, and the client decrypts the result. All steps run in one process.

```python
from ren.ckks import Ciphertext, CKKSParams, decode, decrypt, encode, encrypt, keygen


def circuit(x: Ciphertext) -> Ciphertext:
    return x**2 + x + 1


params = CKKSParams.default(boot_params=None)
keys = keygen(params)

inputs = [1.0, 2.0, 3.0]

# The client encrypts the inputs.
with keys.context():
    encrypted = encrypt(encode(inputs, msg_bound=4))

# The evaluator runs the circuit using only public keys.
with keys.public().context():
    result = circuit(encrypted)

# The client decrypts and decodes the result.
with keys.context():
    decoded = decode(decrypt(result), length=len(inputs))

print([round(x.real, 3) for x in decoded])
# [3.0, 7.0, 13.0]
```

Use `ren.jit` around the evaluator's computation when running it repeatedly. It captures the kernels so later calls can replay them without rebuilding the computation in Python.

Keep raw decrypted CKKS results private. Sharing them can expose information about the secret key. Check [OpenFHE's security notes](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/security.html#notes-specific-to-the-ckks-scheme) for more information.

## Installation

Add Ren to an existing uv project (Python 3.14+).

```bash
uv add 'git+https://github.com/fieldextension/ren.git@main'
```

Ren follows a rolling release on `main`. Breaking API changes can land at any commit. Commit your project's `uv.lock` to pin the Ren revision.

To update Ren and test your project:

```bash
uv sync -P ren
uv run pytest
```

## Working on Ren

[Fork the repository](https://github.com/fieldextension/ren/fork), then clone your fork and create a branch for your change:

```bash
git clone https://github.com/YOUR-USERNAME/ren.git
cd ren
git remote add upstream https://github.com/fieldextension/ren.git
git switch -c my-change
uv sync --locked
```

Your fork is `origin`. To fetch the latest changes from Ren, run `git fetch upstream`.

### Tests and checks

You'll also need:

- [just](https://github.com/casey/just#installation)
- [clang-format](https://clang.llvm.org/docs/ClangFormat.html) 23

Run these commands from your Ren checkout:

```bash
just test                          # Run fast tests
just test tests/primitives -k ntt   # Run selected tests
just check                         # Check lint, formatting, and Python types
just fix                           # Apply lint fixes and format code
```

Run `just` to list all commands. `just test-all` includes slow tests. When CUDA is selected, `just test` also runs the native CUDA tests and requires `nvcc` on PATH.

### Try local changes in a project

From your application's directory, use your local Ren checkout as an editable dependency:

```bash
uv add --editable /path/to/ren
```

Your project now uses your checkout, so edits to Ren's Python source take effect on the next run.

Before sharing your project, replace the local path with a Git dependency others can install. If your project needs unmerged changes, use the branch on your fork that contains them. Once those changes land in `fieldextension/ren`, you can [switch back to `main`](#installation).

### Propose a change

Commit your changes, push your branch to your fork, and open a pull request against `main` in `fieldextension/ren`. See [GitHub's guide](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) for the full workflow.

PR guidelines:

- Keep PRs focused and small enough to review.
- Briefly explain what changed and why Ren needs it. Link related issues.
- Include a regression test for bug fixes.
- If your PR is based on another contributor's implementation, credit them as a co-author even if you rewrote it. Before squash-merging, check that the final commit message includes a `Co-authored-by: Name <email>` line for each co-author, using an email linked to their GitHub account.
- If you include or adapt third-party code, preserve its notices and record its source and license in [THIRD_PARTY.md](THIRD_PARTY.md).
- If you use AI, make sure you can explain the changes you submit. Don't list AI tools as co-authors.

## Acknowledgments

Thanks to everyone who has helped build Ren through code, math, tools, testing, and applications that informed its development.

Much of Ren's code and design is based on [tinygrad](https://github.com/tinygrad/tinygrad), and we keep learning from the project as it develops. Thank you to the tinygrad team and contributors.

Ren's original code is licensed under the [MIT License](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream notices and license texts.
