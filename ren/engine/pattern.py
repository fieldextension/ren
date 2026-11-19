import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields
from functools import lru_cache, partial
from typing import Any

from ren.context import REWRITE_STACK_LIMIT
from ren.dtype import DType, dtypes
from ren.ir import Op
from ren.ir.node import Node
from ren.ir.op import GroupOp
from ren.mixin import ScalarMixin


def _ops(ops: Op | Iterable[Op] | None) -> tuple[Op, ...] | None:
    """Normalize an op selector to a sorted tuple (None = unconstrained)."""
    if ops is None:
        return None
    normalized = (ops,) if isinstance(ops, Op) else tuple(sorted(set(ops)))
    if not normalized:
        raise ValueError("expected at least one Op")
    return normalized


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Pat(ScalarMixin):
    """Pattern for matching IR nodes.

    src types after construction:
    - None: wildcard, match any children
    - Pat: variadic, match any number of children with this pattern
    - tuple[tuple[Pat, ...], ...]: flattened orderings (each ordering is flat)

    unwrap names wrapper ops (AS_RING, AFTER, ...) to see through, however
    deeply stacked, before the structural checks. `name` binds the node the
    checks ran on; `outer` binds the node as encountered, which is the outermost
    wrapper when there was one. `.const()` captures use `outer`, so they keep
    their ring wrapper.
    """

    arg: Any = field(hash=False)
    dtype: tuple[DType, ...] | None
    name: str | None
    op: tuple[Op, ...] | None
    outer: str | None
    src: tuple[tuple[Pat, ...], ...] | Pat | None
    unwrap: tuple[Op, ...] | None

    def __init__(
        self,
        op: Op | Iterable[Op] | None = None,
        src: Sequence[Pat | int] | Pat | None = None,
        arg: Any = None,
        dtype: DType | tuple[DType, ...] | None = None,
        name: str | None = None,
        outer: str | None = None,
        unwrap: Op | Iterable[Op] | None = None,
    ):
        object.__setattr__(self, "op", _ops(op))
        object.__setattr__(self, "dtype", None if dtype is None else (dtype,) if isinstance(dtype, DType) else tuple(dtype))
        object.__setattr__(self, "arg", arg)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "outer", outer)
        object.__setattr__(self, "unwrap", _ops(unwrap))

        if src is None or isinstance(src, Pat):
            normalized: tuple[tuple[Pat, ...], ...] | Pat | None = src
        else:
            # Every ordering (all permutations if commutative), expanded over each child's own orderings.
            children = tuple(_to_pat(s) for s in src)
            orderings = itertools.permutations(children) if isinstance(src, list) else (children,)
            expanded = (combo for ordering in orderings for combo in itertools.product(*(c._orderings() for c in ordering)))
            normalized = tuple(dict.fromkeys(expanded))
        object.__setattr__(self, "src", normalized)

    def _replace(self, **changes: Any) -> Pat:
        """Copy with fields overridden; values bypass normalization, so pass them normalized."""
        new = object.__new__(Pat)
        for f in fields(Pat):
            object.__setattr__(new, f.name, changes.get(f.name, getattr(self, f.name)))
        return new

    def _orderings(self) -> tuple[Pat, ...]:
        """This pattern as one flat pattern per child ordering."""
        if not isinstance(self.src, tuple) or len(self.src) == 1:
            return (self,)
        return tuple(self._replace(src=(o,)) for o in self.src)

    def _match_ordering(self, node: Node, bindings: dict[str, Node]) -> bool:
        """Match against node; self must hold a single child ordering (see _orderings)."""
        outer = node
        if self.unwrap is not None:
            while node.op in self.unwrap and node.src:
                node = node.src[0]

        # Structural checks
        if self.op is not None and node.op not in self.op:
            return False
        if self.arg is not None and self.arg != (outer.const_val if node.op is Op.CONST else node.arg):
            return False
        if self.dtype is not None and node.dtype not in self.dtype:
            return False

        # Match children
        if isinstance(self.src, Pat):  # variadic: every child matches it
            for child in node.src:
                if not self.src._match_ordering(child, bindings):
                    return False
        elif self.src is not None:  # fixed: this flat ordering
            ordering = self.src[0]
            if len(ordering) != len(node.src):
                return False
            for p, c in zip(ordering, node.src):
                if not p._match_ordering(c, bindings):
                    return False

        if self.outer is not None and not _capture(self.outer, outer, bindings):
            return False
        return self.name is None or _capture(self.name, node, bindings)

    def __call__(
        self,
        op: Op | Iterable[Op] | None = None,
        src: Sequence[Pat | int] | Pat | None = None,
        *,
        arg: Any = None,
        dtype: DType | tuple[DType, ...] | None = None,
        outer: str | None = None,
        unwrap: Op | Iterable[Op] | None = None,
    ) -> Pat:
        return Pat(op=op, src=src, arg=arg, dtype=dtype, name=self.name, outer=outer, unwrap=unwrap)

    def _binop(self, op: Op, other: Pat | int, reverse: bool = False) -> Pat:
        other_pat = _to_pat(other)
        lhs, rhs = (other_pat, self) if reverse else (self, other_pat)
        return Pat(op=op, src=[lhs, rhs] if op.is_commutative() else (lhs, rhs), dtype=dtypes.bool if op in {Op.CMPEQ, Op.CMPLT} else None)

    def _unary(self, op: Op, *, arg=None) -> Pat:
        return Pat(op=op, src=(self,), arg=arg)

    def const(self, value: int | None = None, *, dtype: DType | tuple[DType, ...] | None = None) -> Pat:
        return Pat(op=Op.CONST, arg=value, dtype=dtype, outer=self.name, unwrap=(Op.AS_RING, Op.CAST))

    def cast(self, dtype: DType | tuple[DType, ...] | None = None, *, name: str | None = None) -> Pat:
        return Pat(Op.CAST, (self,), dtype=dtype, name=name)

    def where(self, yes: Pat | int, no: Pat | int) -> Pat:
        return Pat(Op.WHERE, (self, _to_pat(yes), _to_pat(no)))

    def sink(self, *src: Pat | int | None) -> Pat:
        return Pat(op=Op.SINK, src=tuple(_to_pat(s) for s in src if s is not None) or None, dtype=dtypes.void, name=self.name)

    def index(self, idx: Pat | int | None = None) -> Pat:
        return Pat(op=Op.INDEX, src=(self, _to_pat(idx) if idx is not None else Pat()))

    def load(self) -> Pat:
        return Pat(op=Op.LOAD, src=(self,))

    def store(self, value: Pat | int | None = None) -> Pat:
        return Pat(op=Op.STORE, src=(self, _to_pat(value) if value is not None else Pat()))

    def base(self) -> Pat:
        """Also match through any stack of metadata wrappers, like `Node.base`; `name` binds the node underneath."""
        return self._replace(unwrap=_ops((*(self.unwrap or ()), *GroupOp.Metadata)))


class PatNamespace:
    def __getattr__(self, name: str) -> Pat:
        if name.startswith("_"):
            raise AttributeError(name)
        return Pat(name=name)

    def __call__(
        self,
        op: Op | Iterable[Op] | None = None,
        src: Sequence[Pat | int] | Pat | None = None,
        *,
        arg: Any = None,
        dtype: DType | tuple[DType, ...] | None = None,
    ) -> Pat:
        return Pat(op=op, src=src, arg=arg, dtype=dtype)


pat = PatNamespace()


def _to_pat(x: Pat | int) -> Pat:
    return x if isinstance(x, Pat) else Pat().const(x)


def _capture(name: str, node: Node, bindings: dict[str, Node]) -> bool:
    """Bind name to node; a name bound twice must be the same node."""
    if name in bindings:
        return bindings[name] is node
    bindings[name] = node
    return True


RuleAction = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class Rule:
    """A pattern and its action."""

    pat: Pat
    action: RuleAction


Bindings = dict[str, Node]
Matcher = Callable[[Node], Sequence[Bindings]]
CompiledState = tuple[dict[Op | None, tuple[tuple[int, Matcher], ...]], tuple[RuleAction, ...]]


@dataclass(frozen=True, slots=True)
class Gate:
    """Return the replacement and stop traversal at this occurrence.

    Parent patterns can inspect its structure.
    """

    node: Node


def _compile_matcher(pattern: Pat) -> Matcher:
    """Create a matcher closure for a pattern."""
    if pattern.src is None and pattern.arg is None and pattern.dtype is None and pattern.outer is None and pattern.unwrap is None:
        # Nothing to check beyond the op, which the dispatch index already filtered on.
        if pattern.name is None:
            return lambda node: ({},)
        name = pattern.name
        return lambda node: ({name: node},)

    patterns = pattern._orderings()
    # Per op-constrained child, the ops (own or wrapper) some child of a matching node must carry.
    early_reject: tuple[frozenset[Op], ...] = ()
    if isinstance(pattern.src, tuple):
        early_reject = tuple(frozenset(c.op) | frozenset(c.unwrap or ()) for c in pattern.src[0] if c.op is not None)

    def matcher(node: Node) -> Sequence[Bindings]:
        for child_ops in early_reject:
            if not any(source.op in child_ops for source in node.src):
                return ()
        matches: list[Bindings] = []
        for pat in patterns:
            bindings: Bindings = {}
            if pat._match_ordering(node, bindings):
                matches.append(bindings)
        return matches

    return matcher


@lru_cache(maxsize=512)
def _compile_patterns(patterns: tuple[Pat, ...]) -> dict[Op | None, tuple[tuple[int, Matcher], ...]]:
    """Compile patterns into an op-indexed matcher table.

    Each op bucket holds that op's rules followed by the wildcard rules, in rule
    order, so a lookup on `node.op` (falling back to the `None` bucket) yields
    every candidate in one sequence.

    Cached state is pattern-only (no actions), preventing closure-retention leaks.
    """
    by_op: dict[Op, list[tuple[int, Matcher]]] = {}
    wildcard: list[tuple[int, Matcher]] = []
    for idx, pattern in enumerate(patterns):
        entry = (idx, _compile_matcher(pattern))
        if pattern.op is None:
            wildcard.append(entry)
        else:
            # A node reaches the rule by its own op, or by a wrapper op the pattern sees through.
            for op in {*pattern.op, *(pattern.unwrap or ())}:
                by_op.setdefault(op, []).append(entry)
    index: dict[Op | None, tuple[tuple[int, Matcher], ...]] = {op: tuple(entries + wildcard) for op, entries in by_op.items()}
    index[None] = tuple(wildcard)
    return index


def _call(action: RuleAction, bindings: Bindings) -> Any:
    return action(**bindings)


def _bind(action: RuleAction, bindings: Bindings) -> RuleAction:
    return partial(action, **bindings) if bindings else action


def _apply_rules(
    node: Node,
    index: dict[Op | None, tuple[tuple[int, Matcher], ...]],
    actions: tuple[RuleAction, ...],
    call: Callable[[RuleAction, Bindings], Any],
) -> Any | None:
    """Apply the first matching rule; an action returning None or the node itself declines the match."""
    for ridx, matcher in index.get(node.op) or index[None]:
        for bindings in matcher(node):
            if (out := call(actions[ridx], bindings)) is not None and out is not node:
                return out
    return None


class RuleSet(Sequence[Rule]):
    """A flat sequence of rules.

    Op-constrained rules are tried before wildcard rules, preserving rule order
    within each group. A matched pattern's bindings become the action's keyword
    arguments: `match` calls the action, `dispatch` returns it partially applied.
    """

    __slots__ = ("_rules", "_cache")

    _rules: tuple[Rule, ...]
    _cache: CompiledState | None

    def __init__(self, values: Iterable[Rule]):
        self._rules = tuple(values)
        if not self._rules:
            raise ValueError("RuleSet must contain at least one rule")
        self._cache = None

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self):
        return iter(self._rules)

    def __getitem__(self, index):
        return self._rules[index]

    def __add__(self, other: RuleSet) -> RuleSet:
        return RuleSet(self._rules + other._rules)

    def dispatch(self, node: Node) -> RuleAction | None:
        """The first matching rule's action, with the pattern's bindings applied."""
        return _apply_rules(node, *self._compiled(), _bind)

    def match(self, node: Node) -> Any | None:
        """The first non-None result of a matching rule's action."""
        return _apply_rules(node, *self._compiled(), _call)

    def _compiled(self) -> CompiledState:
        """Reuse the structural matcher table and this ruleset's actions."""
        if self._cache is None:
            self._cache = (_compile_patterns(tuple(r.pat for r in self._rules)), tuple(r.action for r in self._rules))
        return self._cache


def rules(values: Sequence[tuple[Pat, RuleAction]]) -> RuleSet:
    """Build a RuleSet from pattern/action pairs."""
    return RuleSet(Rule(pat, action) for pat, action in values)


def rewrite(
    root: Node,
    ruleset: RuleSet,
    *,
    bottom_up: bool = False,
    enter_calls: bool = False,
    trace: Callable[[str | None, Node, Node], None] | None = None,
    name: str | None = None,
) -> Node:
    index, actions = ruleset._compiled()
    results: dict[Node, Node] = {}
    bottom_up_cache: dict[Node, Node | Gate | None] = {}

    def apply_fixpoint(node: Node) -> Node | Gate:
        seen: set[Node] = set()
        while True:
            if node in seen:
                raise RuntimeError("infinite loop in fixed_point_rewrite")
            seen.add(node)
            try:
                out = bottom_up_cache[node]
            except KeyError:
                out = bottom_up_cache[node] = apply_once(node)
            if out is None:
                return node
            if isinstance(out, Gate):
                return out
            node = out

    def apply_once(node: Node) -> Node | Gate | None:
        out = _apply_rules(node, index, actions, _call)
        if out is None or isinstance(out, Gate):
            return out
        if isinstance(out, int):
            out = node.const_like(out)
        if out is node:
            return None
        if trace is not None and len(node.src) == len(out.src):
            # Report children the rule wrapped in place (the new child has the old one as a source)
            for cs, os in zip(node.src, out.src):
                if cs is not os and cs in os.src:
                    trace(name, cs, os)
        return out

    # Stack entries: (original node, stage, current node)
    # 0: descend: bottom_up fixpoint on the node, then push its children
    # 1: ascend: rebuild from the rewritten children; default mode applies rules once and re-enters the result
    # 2: finish the original node from its rewritten replacement
    stack: list[tuple[Node, int, Node]] = [(root, 0, root)]
    stack_limit = REWRITE_STACK_LIMIT.value

    def finish(orig: Node, out: Node) -> None:
        results[orig] = out
        if trace is not None and orig is not out:
            trace(name, orig, out)

    while stack:
        if len(stack) > stack_limit:
            raise RuntimeError(f"infinite loop in rewrite (stack too big, limit={stack_limit})")
        orig, stage, cur = stack.pop()
        if stage == 2:
            finish(orig, results[cur])
            continue
        if orig in results:
            continue

        if stage == 0:  # Descending: optionally transform, then push children
            if bottom_up:
                res = apply_fixpoint(cur)
                if isinstance(res, Gate):
                    finish(orig, res.node)
                    continue
                cur = res
            stack.append((orig, 1, cur))
            for s in reversed(cur.src if enter_calls else cur.inputs):
                stack.append((s, 0, s))
            continue

        # Ascending: rebuild from rewritten children, optionally transform.
        sources = cur.src if enter_calls else cur.inputs
        src = tuple(results[source] for source in sources)
        # Preserve the callable body while rebuilding its argument bindings.
        if not enter_calls and cur.op is Op.CALL:
            src = (cur.src[0], *src)
        if src != cur.src:
            cur = cur.replace(src=src)

        res = None if bottom_up else apply_once(cur)
        if res is None:
            finish(orig, cur)
        elif isinstance(res, Gate):
            finish(orig, res.node)
        else:
            # Re-enter rewritten node to allow multi-step rewrites.
            stack.append((orig, 2, res))
            stack.append((res, 0, res))

    return results[root]


__all__ = ["Gate", "Pat", "Rule", "RuleSet", "pat", "rewrite", "rules"]
