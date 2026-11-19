import functools
from collections.abc import Callable
from typing import Any

from ren.container import PolyContainer
from ren.device import Device
from ren.ir import Node, Op, ParamArg
from ren.poly import Poly, _function_tracing


class function[ReturnT]:
    """Trace a graph-building Python function into reusable callable IR.

    Python runs on every call with the actual arguments. Compatible traced
    bodies share a scheduled template, with fresh bindings for each invocation.
    Non-Poly results are rebuilt from that invocation's Python result.
    """

    __name__: str

    def __init__(self, fn: Callable[..., ReturnT]):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __get__(self, obj: Any, _objtype: type | None = None):
        return self if obj is None else functools.partial(self.__call__, obj)

    def __call__(self, *args: Any, **kwargs: Any) -> ReturnT:
        arg_polys = PolyContainer.polys(args, kwargs)
        if not arg_polys:
            raise RuntimeError("ren.function: function must receive at least one Poly argument")

        actuals: list[Node] = []
        params: dict[Node, Node] = {}
        for poly in arg_polys:
            source = poly.node
            if source._device is not None and source not in params:
                params[source] = Node.param(len(actuals), source).as_ring(poly.ring, poly.domain)
                actuals.append(source if source.has_buffer_identity else source.bufferize())

        _function_tracing.append(None)
        try:
            ret = self.fn(*args, **kwargs)
        finally:
            _function_tracing.pop()
        out_polys, output_spec = PolyContainer.flatten(ret)
        if not out_polys:
            raise TypeError("ren.function: outputs must contain at least one Poly")

        parametric_root = Node.sink(*(poly.node for poly in out_polys)).substitute(params)
        root_nodes = parametric_root.toposort()
        if any(node.op is Op.RANDOM for node in root_nodes):
            raise RuntimeError("ren.function: random values must be created outside the function")

        implicit = [node for node in root_nodes if node.op is Op.BUFFER]
        replacements = {node: Node.param(len(actuals) + slot, node) for slot, node in enumerate(implicit)}
        allocations = (node for node in root_nodes if node.op is Op.ALLOC)
        replacements.update({node: node.replace(arg=ParamArg(-slot - 1, node.arg.size, node.arg.device)) for slot, node in enumerate(allocations)})
        parametric_root = parametric_root.substitute(replacements)
        outputs = tuple(dict.fromkeys(parametric_root.src))
        buffers: list[Node] = []
        stores: list[Node] = []
        device = next((value.device for value in (*outputs, *actuals) if value._device is not None), None) or Device.DEFAULT
        for slot, value in enumerate(outputs, len(actuals) + len(implicit)):
            value = value.src[0] if value.op is Op.CONTIGUOUS else value
            ring = value.ring
            assert ring is not None, "function output missing ring metadata"
            output = Node.alloc(value._device or device, value._storage_size(), value.dtype).as_ring(ring, value.domain)
            target = Node.param(slot, output)
            buffers.append(output)
            stores.append(target.after(target.store(value)))
        call = Node.sink(*stores).call(*actuals, *implicit, *(output.base for output in buffers))
        results = {source: output.after(call) for source, output in zip(outputs, buffers, strict=True)}
        output_replacements: dict[int, Poly] = {}
        for poly, source in zip(out_polys, parametric_root.src, strict=True):
            if id(poly) not in output_replacements:
                output_replacements[id(poly)] = Poly(results[source])
        return output_spec.unflatten(output_replacements[id(poly)] for poly in out_polys)


__all__ = ["function"]
