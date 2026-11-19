from dataclasses import dataclass
import hashlib
from collections import Counter
from collections.abc import Sequence
from textwrap import dedent

from ren.dtype import DType, dtypes
from ren.engine.pattern import pat, rules
from ren.ir import Op
from ren.ir.node import Node, from_rns_limb_count
from ren.kernel import KernelSpec
from ren.polydist import DiscreteGaussian, SparseTernary, Ternary, UniformBounded, UniformModQ
from ren.ring import Residue, RingSpec
from ren.tables import DistTables

from .kernels import load_kernel
from .ntt import _select_ntt_schedule

_THREAD_IDX_1D = "blockIdx.x * blockDim.x + threadIdx.x"


@dataclass(frozen=True, slots=True)
class RenderedKernel:
    source: str
    name: str

    @property
    def cache_key(self) -> tuple[str, str]:
        source_hash = hashlib.sha256(self.source.encode()).hexdigest()[:16]
        return (self.name, source_hash)


def _c_type(dtype: DType) -> str:
    if dtype is dtypes.uint128:
        return "unsigned __int128"
    return "unsigned long long" if dtype is dtypes.index else dtype.name


def _literal(n: Node) -> str:
    value = n.const_val
    assert value is not None
    if n.dtype is dtypes.bool:
        return "true" if value else "false"
    if n.dtype is dtypes.uint128:
        return f"((((unsigned __int128){value >> 64}ULL) << 64) + {value & ((1 << 64) - 1)}ULL)"
    return f"{value}{'U' if n.dtype is dtypes.uint32 else 'ULL'}"


def _coeff_tag(dtype: DType) -> str:
    return f"u{dtype.itemsize * 8}"


def _render_expr(program: Sequence[Node], tables: dict[Node, str]) -> tuple[list[str], dict[Node, str]]:
    materialize = {s for n in program if n.op is Op.AFTER for s in n.src[1:]}
    expr_for: dict[Node, str] = {}
    lines: list[str] = []
    # Output STOREs still count as uses; the caller emits their writes.
    child_count = Counter(child for n in program for child in n.src if n.op is not Op.CAST or child.op is not Op.CONST)

    def render_load(idx: Node) -> str:
        buf, offset = idx.src
        source = tables[buf] if buf.op is Op.CONST_ARRAY else f"buf{buf.arg.slot}"
        return f"__ldg(&{source}[{expr_for[offset]}])"

    render_rules = rules([
        (pat(Op.GIDX), lambda: "idx"),
        (pat.n(Op.AFTER), lambda n: expr_for[n.src[0]]),
        (pat.idx(Op.INDEX).load(), render_load),
        (pat(Op.CONST).cast(name="n"), _literal),
        (pat.x // pat.y, lambda x, y: f"({expr_for[x]} / {expr_for[y]})"),
        (pat.x + pat.y, lambda x, y: f"({expr_for[x]} + {expr_for[y]})"),
        (pat.x - pat.y, lambda x, y: f"({expr_for[x]} - {expr_for[y]})"),
        (pat.x * pat.y, lambda x, y: f"({expr_for[x]} * {expr_for[y]})"),
        (pat.x < pat.y, lambda x, y: f"({expr_for[x]} < {expr_for[y]})"),
        (pat.x.eq(pat.y), lambda x, y: f"({expr_for[x]} == {expr_for[y]})"),
        (pat.x & pat.y, lambda x, y: f"({expr_for[x]} & {expr_for[y]})"),
        (pat.x % pat.y, lambda x, y: f"({expr_for[x]} % {expr_for[y]})"),
        (pat.n(Op.BITREVERSE, pat.x), lambda n, x: f"(__brevll({expr_for[x]}) >> (64 - {n.arg}))"),
        # Index arithmetic wraps as unsigned; widening preserves the signed index value.
        (pat.x(dtype=dtypes.index).cast(dtypes.uint128), lambda x: f"((unsigned __int128)((long long){expr_for[x]}))"),
        (pat.x.cast(name="n"), lambda n, x: f"(({_c_type(n.dtype)}){expr_for[x]})"),
        (pat.x << pat.y, lambda x, y: f"({expr_for[x]} << {expr_for[y]})"),
        (pat.x >> pat.y, lambda x, y: f"({expr_for[x]} >> {expr_for[y]})"),
        (pat.cond.where(pat.x, pat.y), lambda cond, x, y: f"({expr_for[cond]} ? {expr_for[x]} : {expr_for[y]})"),
        (-pat.x, lambda x: f"(-{expr_for[x]})"),
    ])

    for n in program:
        if n.op is Op.CONST and not child_count[n]:
            continue
        if n.op in {Op.PARAM, Op.CONST_ARRAY, Op.INDEX, Op.SINK, Op.DEVICE, Op.STORE}:
            continue
        if (expr := render_rules.match(n)) is None:
            raise NotImplementedError(f"Unsupported CUDA scalar operation: {n.op}")

        if n.op is Op.AFTER and n.src[0] in materialize:
            expr_for[n] = expr
        elif n in materialize or (n.op not in {Op.GIDX, Op.CONST} and child_count[n] > 1):
            var = f"t{len(lines)}"
            lines.append(f"{_c_type(n.dtype)} {var} = {expr};")
            expr_for[n] = var
        else:
            expr_for[n] = expr

    return lines, expr_for


def _global_kernel(name: str, params: str, body: str) -> str:
    return f"""\
    extern "C" __global__ void {name}({params}){{
        {body}
    }}"""


def _constexpr_array(name: str, values, dtype: str = "unsigned long long") -> str:
    suffix = {"unsigned int": "U", "unsigned long long": "ULL"}.get(dtype, "")
    arr = ", ".join(f"{v}{suffix}" for v in values)
    return f"__device__ constexpr {dtype} {name}[] = {{{arr}}};"


def _kernel_params(spec: KernelSpec) -> str:
    outputs = set(spec.output_slots)
    parts = []
    for param in spec.params:
        writable = param.arg.slot in outputs
        const = "" if writable else "const "
        restrict = "" if writable else "__restrict__ "
        parts.append(f"{const}{param.dtype.name}* {restrict}buf{param.arg.slot}")
    parts.append("unsigned long long size")
    return ", ".join(parts)


_RESIDUE_MODE = {
    Residue.NONNEGATIVE: "RenResidueMode::Nonnegative",
    Residue.BALANCED: "RenResidueMode::Balanced",
}


def _tpl(name: str, *args: str) -> str:
    return f"{name}<{', '.join(args)}>"


def _loaded_kernel(filename: str, name: str) -> RenderedKernel:
    return RenderedKernel(source=load_kernel(filename), name=f"&{name}")


def _generated_kernel(source: str, name: str) -> RenderedKernel:
    return RenderedKernel(
        source=dedent(f"""\
    #include "common.cuh"
    {source}
    """),
        name=f"&{name}",
    )


def _base_change_kernel_name(dtype: DType, num_src: int, num_dst: int, residue: Residue) -> str:
    return _tpl("ren_base_change_coeff", dtype.name, f"{num_src}ULL", f"{num_dst}ULL", _RESIDUE_MODE[residue])


def _rns_prefix_extend_kernel_name(dtype: DType, num_old: int, num_new: int, residue: Residue) -> str:
    return _tpl("ren_rns_prefix_extend", dtype.name, f"{num_old}ULL", f"{num_new}ULL", _RESIDUE_MODE[residue])


def _rescale_kernel_name(src_ring: RingSpec, dst_ring: RingSpec) -> str:
    dst_moduli = set(dst_ring.moduli)
    num_removed = sum(1 for modulus in src_ring.moduli if modulus not in dst_moduli)
    kernel_name = "ren_rescale_quotient" if dst_ring.is_quotient_of(src_ring) else "ren_rescale_mixed"
    return _tpl(kernel_name, src_ring.dtype.name, f"{num_removed}ULL", f"{len(dst_ring.moduli)}ULL")


def _to_rns_kernel_name(dtype: DType, limb_count: int) -> str:
    return _tpl("ren_to_rns_limbs", dtype.name, f"{limb_count}U")


def _from_rns_kernel_name(dtype: DType, num_primes: int, limb_count: int) -> str:
    return _tpl("ren_from_rns_limbs", dtype.name, f"{num_primes}U", f"{limb_count}U")


class CUDARenderer:
    def __init__(self, arch: str | None):
        self.arch = arch

    def render(self, spec: KernelSpec) -> tuple[RenderedKernel, ...]:
        """Return the kernels needed for this operation in execution order."""
        ring = spec.ring
        render_rules = rules([
            (pat().sink(pat().store(pat(Op.NTT))), lambda: self._render_ntt(ring, inverse=False)),
            (pat().sink(pat().store(pat(Op.INTT))), lambda: self._render_ntt(ring, inverse=True)),
            (pat().sink(pat().store(pat.node(Op.RESCALE))), lambda node: self._render_rescale(node, ring)),
            (pat().sink(pat().store(pat.node(Op.TO_RING))), lambda node: self._render_to_ring(node, ring)),
            (pat().sink(pat().store(pat.node(Op.RANDOM))), lambda node: self._render_random(node, ring)),
            (pat().sink(pat().store(pat.node(Op.TO_RNS))), lambda node: self._render_to_rns(node, ring)),
            (pat().sink(pat().store(pat(Op.FROM_RNS))), lambda: self._render_from_rns(ring)),
            (pat(Op.SINK), lambda: self._render_generated(spec)),
        ])
        result = render_rules.match(spec.ast)
        if result is None:
            raise NotImplementedError(f"CUDARenderer doesn't handle {spec.ast.op}")
        return result

    def _render_ntt(self, ring: RingSpec, *, inverse: bool) -> tuple[RenderedKernel, ...]:
        schedule = _select_ntt_schedule(self.arch, ring, inverse=inverse)
        return tuple(_loaded_kernel("ntt.cu", phase.kernel_name) for phase in schedule)

    def _render_to_ring(self, node: Node, src_ring: RingSpec) -> tuple[RenderedKernel, ...]:
        dst_ring = node.arg.ring
        num_src = len(src_ring.moduli)
        num_dst = len(dst_ring.moduli)

        if num_dst > num_src and dst_ring.moduli[:num_src] == src_ring.moduli:
            return (_loaded_kernel("rns_ops.cu", _rns_prefix_extend_kernel_name(src_ring.dtype, num_src, num_dst, node.arg.residue)),)

        limb_map = self._get_permutation_map(src_ring.moduli, dst_ring.moduli)
        if limb_map is not None:
            return self._render_permutation(src_ring, limb_map)

        return self._render_base_change(src_ring, num_src, num_dst, node.arg.residue)

    def _render_random(self, node: Node, ring: RingSpec) -> tuple[RenderedKernel, ...]:
        dist = node.arg.dist
        match dist:
            case DiscreteGaussian(stdev=stdev):
                dtype, multi_sample = DistTables._discrete_gaussian_plan_shape(stdev)
                kernel_name = "ren_random_large_discrete_gaussian" if multi_sample else "ren_random_sample"
                return (_loaded_kernel("rand.cu", _tpl(kernel_name, ring.dtype.name, dtype.name)),)
            case SparseTernary():
                kernel_name = "ren_random_sparse_ternary"
            case Ternary():
                kernel_name = "ren_random_ternary"
            case UniformBounded():
                kernel_name = "ren_random_uniform_bounded"
            case UniformModQ():
                kernel_name = "ren_random_uniform_mod_q"
            case _:
                raise NotImplementedError(f"Unsupported distribution: {dist}")
        return (_loaded_kernel("rand.cu", _tpl(kernel_name, ring.dtype.name)),)

    def _render_to_rns(self, node: Node, ring: RingSpec) -> tuple[RenderedKernel, ...]:
        return (_loaded_kernel("rns_ops.cu", _to_rns_kernel_name(ring.dtype, node.arg.limb_count)),)

    def _render_from_rns(self, ring: RingSpec) -> tuple[RenderedKernel, ...]:
        return (_loaded_kernel("rns_ops.cu", _from_rns_kernel_name(ring.dtype, len(ring.moduli), from_rns_limb_count(ring))),)

    def _get_permutation_map(self, src_moduli: tuple[int, ...], dst_moduli: tuple[int, ...]) -> tuple[int, ...] | None:
        src_to_idx = {p: i for i, p in enumerate(src_moduli)}
        limb_map = []
        for p in dst_moduli:
            if p not in src_to_idx:
                return None
            limb_map.append(src_to_idx[p])
        return tuple(limb_map)

    def _render_permutation(self, ring: RingSpec, limb_map: tuple[int, ...]) -> tuple[RenderedKernel, ...]:
        fn_name = f"ren_permute_limbs_{_coeff_tag(ring.dtype)}"
        n = ring.n
        num_dst = len(limb_map)
        n_log2 = (n - 1).bit_length()
        coeff_type = ring.dtype.name

        src = f"""\
          {_constexpr_array("d_limb_map", limb_map, "unsigned int")}

          extern "C" __global__ void {fn_name}(
              const {coeff_type}* __restrict__ src,
              {coeff_type}* dst,
              unsigned long long n
          ){{
              unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx >= n * {num_dst}ULL) return;
              unsigned long long dst_limb = idx >> {n_log2};
              unsigned long long coeff = idx & {n - 1}ULL;
              dst[idx] = __ldg(&src[d_limb_map[dst_limb] * n + coeff]);
          }}
        """
        return (_generated_kernel(src, fn_name),)

    def _render_base_change(self, ring: RingSpec, num_src: int, num_dst: int, residue: Residue) -> tuple[RenderedKernel, ...]:
        return (_loaded_kernel("base_change.cu", _base_change_kernel_name(ring.dtype, num_src, num_dst, residue)),)

    def _render_rescale(self, node: Node, ring: RingSpec) -> tuple[RenderedKernel, ...]:
        return (_loaded_kernel("rescale.cu", _rescale_kernel_name(ring, node.arg)),)

    def _render_generated(self, spec: KernelSpec) -> tuple[RenderedKernel, ...]:
        ring = spec.ring
        program = spec.program
        tables = {n: f"table{i}" for i, n in enumerate(n for n in program if n.op is Op.CONST_ARRAY)}
        lines, expr_for = _render_expr(program, tables)
        store_lines = []
        for store in spec.stores:
            target, value = store.src
            slot = (target.src[0] if target.op is Op.INDEX else target).arg.slot
            index = expr_for[target.src[1]] if target.op is Op.INDEX else "idx"
            store_lines.append(f"buf{slot}[{index}] = {expr_for[value]};")
        fn_name = f"ren_generated_{_coeff_tag(ring.dtype)}"
        expr_body = "\n".join(lines)
        store_body = "\n".join(store_lines)
        params = _kernel_params(spec)
        kernel_body = f"""\
          unsigned long long idx = {_THREAD_IDX_1D};
          if (idx >= size) return;
          {expr_body}
          {store_body}"""

        src = f"""\
          {"".join(_constexpr_array(name, node.arg, _c_type(node.dtype)) for node, name in tables.items())}
          {_global_kernel(fn_name, params, kernel_body)}
        """
        return (_generated_kernel(src, fn_name),)


__all__ = ["CUDARenderer", "RenderedKernel"]
