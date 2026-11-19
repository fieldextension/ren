from dataclasses import dataclass
from functools import cached_property

from ren.ir import Node, Op
from ren.ring import RingSpec, Domain


@dataclass(frozen=True, slots=True)
class KernelInfo:
    """Kernel metadata attached to compute SINK roots."""

    device: str
    ring: RingSpec | None = None  # Ring for inputs
    domain: Domain | None = None  # Domain for CONST expansion (NTT=broadcast, COEFF=sparse)


@dataclass(frozen=True)
class KernelSpec:
    """Lowered compute body and its buffer parameters, ordered by CALL slot."""

    ast: Node

    @cached_property
    def program(self) -> tuple[Node, ...]:
        return tuple(self.ast.toposort())

    @cached_property
    def params(self) -> tuple[Node, ...]:
        return tuple(sorted((n for n in self.program if n.op is Op.PARAM), key=lambda n: n.arg.slot))

    @property
    def stores(self) -> tuple[Node, ...]:
        return self.ast.src

    @property
    def ring(self) -> RingSpec:
        return self.ast.arg.ring

    @property
    def domain(self) -> Domain | None:
        return self.ast.arg.domain

    @property
    def argument_slots(self) -> tuple[int, ...]:
        return tuple(n.arg.slot for n in self.params)

    @property
    def output_slots(self) -> tuple[int, ...]:
        return get_output_slots(self.ast)


def get_output_slots(body: Node) -> tuple[int, ...]:
    """Output buffer positions for a lowered kernel or copy."""
    if body.op is Op.COPY:
        return (0,)
    assert body.op is Op.SINK and body.src, "compute body must contain STOREs"
    targets = tuple(store.src[0].src[0] if store.src[0].op is Op.INDEX else store.src[0] for store in body.src)
    assert all(store.op is Op.STORE for store in body.src) and all(target.op is Op.PARAM for target in targets)
    return tuple(target.arg.slot for target in targets)


__all__ = [
    "KernelInfo",
    "KernelSpec",
    "get_output_slots",
]
