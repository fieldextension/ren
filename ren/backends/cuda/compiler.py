import hashlib
from pathlib import Path

from ren.context import CCACHE, CACHE_DIR, DEBUG, LINEINFO
from ren.prof import trace

from . import api as cuda


def _get_cache_dir() -> Path:
    """Get the cache directory, creating it if needed."""
    cache_dir = Path(CACHE_DIR.value)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class CUDACompiler:
    """
    Simple NVRTC-based compiler for CUDA C, parameterized by architecture.

    Supports in-memory headers via the headers dict, enabling #include
    without filesystem access (useful for packaged resources).

    Compiled CUBIN is cached to disk by default. Disable with CCACHE=0.
    """

    opts: list[str]
    headers: dict[str, str]

    def __init__(self, arch: str, headers: dict[str, str] | None = None):
        cuda.nvrtc_version()
        self.arch = arch
        self.headers = headers or {}
        self.opts = [f"--gpu-architecture={arch}", "--minimal", "--device-int128", "--std=c++17", f"-DDEBUG={int(DEBUG.value)}"]
        if LINEINFO:
            self.opts.append("--generate-line-info")

    def _cache_key(self, src: str, name: str) -> str:
        """Compute cache key from source, lowered-name expression, headers, and compiler options."""
        h = hashlib.sha256()
        h.update(src.encode())
        h.update(name.encode())
        for k, v in sorted(self.headers.items()):
            h.update(k.encode())
            h.update(v.encode())
        for opt in self.opts:
            h.update(opt.encode())
        return h.hexdigest()[:16]

    def compile(self, src: str, name: str) -> tuple[bytes, bytes]:
        """Compile CUDA source to CUBIN and lowered symbol name, using disk cache if enabled."""
        if CCACHE:
            cache_key = self._cache_key(src, name)
            cubin_file = _get_cache_dir() / f"cubin_{cache_key}.cubin"
            name_file = _get_cache_dir() / f"cubin_{cache_key}.name"
            if cubin_file.exists() and name_file.exists():
                return cubin_file.read_bytes(), name_file.read_bytes()
            with trace("nvrtc.compile", device="cuda"):
                cubin, lowered_name = cuda.nvrtc_compile_to_cubin(src, self.opts, name=name, headers=self.headers)
            cubin_file.write_bytes(cubin)
            name_file.write_bytes(lowered_name)
            return cubin, lowered_name
        with trace("nvrtc.compile", device="cuda"):
            return cuda.nvrtc_compile_to_cubin(src, self.opts, name=name, headers=self.headers)


__all__ = ["CUDACompiler"]
