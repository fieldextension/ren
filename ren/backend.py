from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Protocol

from ren.allocator import Allocator
from ren.buffer import Buffer
from ren.ir.node import Node
from ren.random import ChaChaRng

if TYPE_CHECKING:
    from ren.engine.realize import ExecUnit


class GraphRunner(Protocol):
    """Protocol for backend-specific runners over execution DAGs."""

    def __init__(self, units: tuple[ExecUnit, ...], bindings: list[Buffer]): ...

    def __call__(self, bindings: list[Buffer]) -> None:
        """Execute the captured graph/queue schedule."""
        ...

    def close(self) -> None:
        """Release any backend resources held by this runner."""
        ...


class Backend(ABC):
    graph: type[GraphRunner] | None = None

    @property
    @abstractmethod
    def allocator(self) -> Allocator: ...

    @abstractmethod
    def synchronize(self) -> None: ...

    @abstractmethod
    def finalize(self) -> None: ...

    @abstractmethod
    def default_rng(self) -> ChaChaRng: ...

    @abstractmethod
    def set_default_rng(self, rng: ChaChaRng): ...

    @abstractmethod
    def make_kernel(self, ast: Node) -> Callable[[list[Buffer]], None]:
        """Create a callable that runs a compute SINK carrying KernelInfo on caller-provided buffers."""
        ...


__all__ = ["Backend", "GraphRunner"]
