from dataclasses import dataclass
from functools import cache

from ren.dtype import dtypes
from ren.ring import RingSpec


@dataclass(frozen=True, slots=True)
class NTTLaunchConfig:
    """One launch covering one or more radix-2 stages, independent of CUDA handles."""

    kernel_name: str
    grid_xy: tuple[int, int]
    block: tuple[int, int]
    shared_bytes: int
    logm: int
    outer_count: int


@cache
def _build_merge_ntt_schedule(ring: RingSpec, *, inverse: bool) -> tuple[NTTLaunchConfig, ...]:
    """Build the complete merge-kernel schedule in execution order."""
    n_power = ring.n.bit_length() - 1
    if 1 <= n_power <= 10:
        tile = 1 << (n_power - 1)
        geometry = [((1, 1), (1, tile), 0, n_power)]
    elif 11 <= n_power <= 17:
        if n_power <= 12:
            first_log_num_threads = 7
            second_log_num_threads = 7
        elif n_power <= 16:
            first_log_num_threads = 8
            second_log_num_threads = 8
        else:
            first_log_num_threads = 8
            second_log_num_threads = 9
        first_outer = n_power - (second_log_num_threads + 1)
        geometry = [
            (
                (1 << (n_power - (first_log_num_threads + 1)), 1),
                (1 << (first_log_num_threads - (first_outer - 1)), 1 << (first_outer - 1)),
                0,
                first_outer,
            ),
            ((1, 1 << first_outer), (1, 1 << second_log_num_threads), first_outer, second_log_num_threads + 1),
        ]
    else:
        raise NotImplementedError(f"merge NTT schedule not implemented for n_power={n_power}")

    direction = "inverse" if inverse else "forward"
    name = f"ren_ntt_{direction}_merge<{ring.dtype.name}>"
    phases = reversed(geometry) if inverse else geometry
    return tuple(
        NTTLaunchConfig(name, grid, block, 2 * block[0] * block[1] * ring.dtype.itemsize, logm, count) for grid, block, logm, count in phases
    )


@cache
def _build_radix8_ntt_schedule(ring: RingSpec, *, inverse: bool) -> tuple[NTTLaunchConfig, ...]:
    """Build launches for the radix-8 kernels' supported 128-by-512 layout."""
    if ring.n != 65536:
        raise NotImplementedError(f"radix-8 NTT schedule not implemented for n={ring.n}")
    # Both directions keep intermediates below 4q, which must fit in one word.
    lazy = all(q < (1 << (8 * ring.dtype.itemsize - 2)) for q in ring.moduli)
    rows, columns, values_per_thread = 128, 512, 8
    tile_columns = 64 // ring.dtype.itemsize  # Match the kernels' 64-byte row tiles.
    log_rows = rows.bit_length() - 1
    geometry = [
        ((columns // tile_columns, 1), (rows * tile_columns // values_per_thread, 1), 0, log_rows),
        ((1, rows), (columns // values_per_thread, 1), log_rows, columns.bit_length() - 1),
    ]
    direction = "inverse" if inverse else "forward"
    phases = reversed(geometry) if inverse else geometry
    return tuple(
        NTTLaunchConfig(
            f"ren_ntt_{direction}_radix8_phase{i}<{ring.dtype.name},{str(lazy).lower()}>",
            grid,
            block,
            values_per_thread * block[0] * block[1] * ring.dtype.itemsize,
            logm,
            count,
        )
        for i, (grid, block, logm, count) in enumerate(phases)
    )


@cache
def _select_ntt_schedule(arch: str | None, ring: RingSpec, *, inverse: bool) -> tuple[NTTLaunchConfig, ...]:
    """Select measured configurations; other layouts and small uint64 rings use merge."""
    if arch == "sm_120" and ring.n == 65536:
        # Measured on RTX PRO 6000 with 256-thread uint32 / 128-thread uint64 row tiles.
        num_primes = len(ring.moduli)
        if ring.dtype is dtypes.uint32 or (ring.dtype is dtypes.uint64 and num_primes >= 5):
            return _build_radix8_ntt_schedule(ring, inverse=inverse)
    return _build_merge_ntt_schedule(ring, inverse=inverse)
