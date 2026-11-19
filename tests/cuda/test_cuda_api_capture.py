from types import SimpleNamespace

import pytest

pytest.importorskip("cuda.bindings")

from ren.backends.cuda import api as cuda_api


def _patch_driver(monkeypatch: pytest.MonkeyPatch, fake_drv: object) -> None:
    monkeypatch.setattr(cuda_api, "_drv", fake_drv)
    monkeypatch.setattr(cuda_api, "_check", lambda x: x)


def test_graph_instantiate(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []

    class InstantiateParams:
        def __init__(self):
            self.flags = None
            self.hUploadStream = None

    def instantiate(graph, params):
        calls.append((graph, params.flags, params.hUploadStream))
        return "exec_handle"

    fake_drv = SimpleNamespace(
        cuGraphInstantiateWithParams=instantiate,
        CUDA_GRAPH_INSTANTIATE_PARAMS=InstantiateParams,
        CUgraphInstantiate_flags=SimpleNamespace(CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD="upload"),
    )
    _patch_driver(monkeypatch, fake_drv)

    exec_handle = cuda_api.graph_instantiate("graph", 7)
    assert exec_handle == "exec_handle"
    assert calls == [("graph", "upload", 7)]


def test_graph_exec_memcpy_node_set_params(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []

    def set_params(*args):
        calls.append(args)
        return None

    fake_drv = SimpleNamespace(cuGraphExecMemcpyNodeSetParams=set_params)
    _patch_driver(monkeypatch, fake_drv)

    cuda_api.graph_exec_memcpy_node_set_params("exec", "node", "params", ctx="ctx")

    assert calls == [("exec", "node", "params", "ctx")]


def test_graph_create_and_add_nodes(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, tuple] = {}

    def create(flags):
        calls["create"] = (flags,)
        return "graph_handle"

    def add_kernel(graph, deps, ndeps, params):
        calls["add_kernel"] = (graph, deps, ndeps, params)
        return "kernel_node"

    def add_memcpy(graph, deps, ndeps, params, ctx):
        calls["add_memcpy"] = (graph, deps, ndeps, params, ctx)
        return "copy_node"

    fake_drv = SimpleNamespace(
        cuGraphCreate=create,
        cuGraphAddKernelNode=add_kernel,
        cuGraphAddMemcpyNode=add_memcpy,
    )
    _patch_driver(monkeypatch, fake_drv)

    g = cuda_api.graph_create()
    k = cuda_api.graph_add_kernel_node(g, ["d0", "d1"], "kparams")
    m = cuda_api.graph_add_memcpy_node(g, ["d1"], "mparams")

    assert g == "graph_handle"
    assert k == "kernel_node"
    assert m == "copy_node"
    assert calls["create"] == (0,)
    assert calls["add_kernel"] == ("graph_handle", ("d0", "d1"), 2, "kparams")
    assert calls["add_memcpy"] == ("graph_handle", ("d1",), 1, "mparams", None)
