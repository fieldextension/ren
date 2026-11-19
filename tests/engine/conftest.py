import importlib
from collections import OrderedDict

import pytest

from ren.ir import Node


@pytest.fixture
def schedule_cache(monkeypatch: pytest.MonkeyPatch) -> OrderedDict[tuple[Node, int], tuple[Node, tuple[Node, ...]]]:
    scheduler = importlib.import_module("ren.engine.schedule")
    monkeypatch.setattr(scheduler, "_prepare_cache", OrderedDict())
    cache = OrderedDict()
    monkeypatch.setattr(scheduler, "_schedule_cache", cache)
    return cache


@pytest.fixture
def prepared_roots(monkeypatch: pytest.MonkeyPatch, schedule_cache) -> list[Node]:
    scheduler = importlib.import_module("ren.engine.schedule")
    roots: list[Node] = []
    prepare = scheduler._prepare_body

    def record(root: Node, trace):
        roots.append(root)
        return prepare(root, trace)

    monkeypatch.setattr(scheduler, "_prepare_body", record)
    return roots


@pytest.fixture
def scheduled_roots(monkeypatch: pytest.MonkeyPatch, schedule_cache) -> list[Node]:
    scheduler = importlib.import_module("ren.engine.schedule")
    roots: list[Node] = []
    bufferize_plan = scheduler.rules.bufferize_plan

    def record(root: Node):
        roots.append(root)
        return bufferize_plan(root)

    monkeypatch.setattr(scheduler.rules, "bufferize_plan", record)
    return roots
