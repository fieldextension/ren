import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.requires_cuda
def test_module_scope_jit_exits_cleanly_with_cuda_graphs():
    script = textwrap.dedent(
        """
        import faulthandler

        faulthandler.enable()
        faulthandler.dump_traceback_later(60, exit=True)

        from ren.backends.cuda.graph import CUDAQueueGraphRunner
        from ren.poly import Poly
        from tests.utils import mixed_prime_ring
        import ren


        @ren.jit
        def add_one(a: Poly) -> Poly:
            return a + 1


        def main() -> None:
            ring = mixed_prime_ring(1 << 16)
            a = Poly([0] * ring.n, ring)
            b = add_one(a)
            print("realized", type(b.realize()).__name__)
            runner = next(iter(add_one._plans.values())).runner
            assert isinstance(runner, CUDAQueueGraphRunner)
            assert runner._graph_exec is not None


        if __name__ == "__main__":
            main()
            print("after main")
        """
    )
    env = os.environ.copy()
    env.setdefault("DEV", "cuda")
    env["GRAPH"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=75,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "after main" in result.stdout
