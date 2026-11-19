"""Polynomial operations backed by type-selected tree traversal."""

from collections.abc import Callable, Iterable
import copy
from typing import Any, Self, cast

from ren import tree
from ren.poly import Poly, realize

__all__ = ["PolyContainer"]
_END = object()


class PolyContainer:
    """Polynomial operations for records and classes with custom tree reduction.

    Traversal follows ren.tree's stored-state rules and __tree_reduce__ hooks, and also
    accepts ordinary records and collections. Shared containers remain shared; repeated
    Poly positions stay separate. Structural reconstruction bypasses constructors and
    copies stored state unless __tree_reduce__ rebuilds.
    """

    __slots__ = ()

    def polys(self: object = None, /, *others: object) -> list[Poly]:
        """Collect Poly positions in storage order, visiting shared containers once."""
        return tree.collect(self, *others, leaf=Poly)

    @staticmethod
    def flatten[Root](value: Root) -> tuple[list[Poly], tree.Spec[Root]]:
        """Extract Poly fields and reusable structure, preserving shared containers."""
        return tree.extract(value, leaf=Poly)

    def map_polys(self, fn: Callable[[Poly], Poly], /) -> Self:
        """Transform each Poly position. The callback must not mutate the input."""
        return tree.replace(self, fn, leaf=Poly)

    def to(self, device: str) -> Self:
        """Move every polynomial to one device, preserving other stored state."""
        return self.map_polys(lambda poly: poly.to(device))

    def replace(self, /, *, polys: Iterable[Poly] | None = None, **changes: object) -> Self:
        """Replace Poly positions and root fields with literal values.

        Polys follow polys() order with an exact count; parent state is snapshotted first.
        Named changes use copy.replace, overriding Poly replacements through __replace__.
        Dataclasses run their constructor. Poly-only replacement bypasses constructors
        unless __tree_reduce__ rebuilds.
        """
        result = self
        if polys is not None:
            replacements = iter(polys)
            try:
                result = self.map_polys(lambda _: next(replacements))
            except StopIteration:
                raise ValueError("not enough polynomials to replace container") from None
            if next(replacements, _END) is not _END:
                raise ValueError("too many polynomials to replace container")
        return copy.replace(cast(Any, result), **changes) if changes else result

    def realize(self) -> Self:
        """Realize all polynomials together."""
        realize(self)
        return self
