from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal, overload, TYPE_CHECKING

from ren.device import Device
from ren.engine.jit import jit
from ren.poly import realize

from .keys import CKKSKeySet
from .params import CKKSParams
from .encoding import FLOAT_PRECISION_BITS, _normalize_precision

if TYPE_CHECKING:
    from .multiparty import MultipartyParams
    from .matrix import LinearTransform, SparseMatrix


@dataclass(slots=True, frozen=True)
class CKKSContext:
    """Keys and runtime settings for CKKS operations.

    Use ``with keys.context(...):`` to select the keys and device for operations
    in the block. These settings are not serialized with the key set.

    ``default_level`` selects the level for encoding when no level is supplied.
    ``keys.context()`` defaults it to ``params.default_level``.

    ``encoding_precision_bits`` sets the precision for encoding and decoding:
    NumPy at 53 bits, FLINT at higher precision. Pass ``prec`` to either
    operation to override it for that call.

    ``track_plaintext_values`` keeps slot values for diagnostics. Arithmetic
    updates them only when tracking is enabled and every encoded operand has
    values. Encryption, decryption, and level conversion preserve existing values.

    ``strict_noise_bounds`` controls what happens when a finite ciphertext
    message bound exceeds q/2: raise if enabled, or mark the noise metadata
    unbounded if disabled. This check is independent of plaintext tracking.
    """

    keyset: CKKSKeySet
    _: KW_ONLY
    default_level: int
    track_plaintext_values: bool = False
    strict_noise_bounds: bool = False
    encoding_precision_bits: int = FLOAT_PRECISION_BITS
    _bootstrap_transforms: dict[tuple[SparseMatrix, int, float], LinearTransform] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.params._require_level(self.default_level, name="default_level")
        object.__setattr__(self, "encoding_precision_bits", _normalize_precision(self.encoding_precision_bits))

    @property
    def device(self) -> str:
        return self.keyset.device

    @property
    def params(self) -> CKKSParams:
        return self.keyset.params

    @property
    def _slot_prec(self) -> int:
        """Precision, including guard bits, for tracking and raw dot products."""
        return max(self.encoding_precision_bits, self.params.max_precision + 10, 100)

    @property
    def multiparty(self) -> MultipartyParams | None:
        return self.keyset.multiparty

    def __enter__(self) -> CKKSContext:
        if _active_context.get() is not None:
            raise RuntimeError("There can only be one CKKSContext active at a time")
        key = id(self.keyset), self.default_level, self.track_plaintext_values, self.strict_noise_bounds, self.encoding_precision_bits
        with ExitStack() as scope:
            scope.enter_context(Device.scope(self.device))
            scope.enter_context(jit.scope(key, owner=self.keyset))
            _active_context.set((self, scope.pop_all()))
        return self

    def __exit__(self, *exc):
        active = _active_context.get()
        if active is None or active[0] is not self:
            raise RuntimeError("CKKSContext is not active")
        _active_context.set(None)
        return active[1].__exit__(*exc)

    def __reduce_ex__(self, _protocol):
        raise TypeError("CKKSContext is runtime-only; serialize CKKSKeySet instead")

    def precompute_bootstrap(self, log_bound: int) -> CKKSContext:
        """Precompute and realize bootstrap transform plans for the given message bound."""
        from .bootstrap import _precompute_bootstrap_polys

        current = context(required=False)
        if current is self:
            polys = _precompute_bootstrap_polys(log_bound=log_bound)
        elif current is None:
            with self:
                polys = _precompute_bootstrap_polys(log_bound=log_bound)
        else:
            raise RuntimeError("cannot precompute bootstrap while a different CKKSContext is active")
        realize(*polys)
        return self


_active_context: ContextVar[tuple[CKKSContext, ExitStack] | None] = ContextVar("ren_ckks_context", default=None)


@overload
def context(*, required: Literal[True] = True) -> CKKSContext: ...


@overload
def context(*, required: Literal[False]) -> CKKSContext | None: ...


def context(*, required: bool = True) -> CKKSContext | None:
    active = _active_context.get()
    if active is None:
        if not required:
            return None
        raise RuntimeError("no active CKKSContext; use `with ctx:`")
    return active[0]
