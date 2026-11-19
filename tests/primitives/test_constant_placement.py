import pickle

import pytest

from ren import function, jit, realize
from ren.device import Device
from ren.poly import Poly
from ren.ring import RingSpec


@pytest.mark.parametrize("ntt", [False, True])
def test_constant_value_and_explicit_storage(ntt):
    ring = RingSpec(n=4, moduli=(17, 97))
    value = Poly.const(7, ring)
    if ntt:
        value = value.ntt()
    assert value.device is None
    assert value.to(Device.DEFAULT) is value
    realize(value)
    assert value.device is None
    restored = pickle.loads(pickle.dumps(value))
    assert restored.device is None
    assert list(restored) == [7, 0, 0, 0]
    copies = [value.clone(device=device) for device in ("python", Device.DEFAULT)]
    realize(*copies)
    for stored, device in zip(copies, ("python", Device.DEFAULT), strict=True):
        assert stored.device == device and stored.is_realized
        assert list(stored) == [7, 0, 0, 0]
    assert copies[0].node.base.buffer.base is not copies[1].node.base.buffer.base
    assert value.device is None


def test_device_free_expression_inherits_each_consumer():
    ring = RingSpec(n=4, moduli=(17, 97))
    value = (Poly.const(7, ring).rotate(1) + 2).intt()
    left = Poly([1, 2, 3, 4], ring, device="python")
    right = Poly([4, 3, 2, 1], ring, device=Device.DEFAULT)
    a, b = left + value, right + value
    realize(a, b)
    assert a.device == "python" and b.device == Device.DEFAULT
    assert list(a) == [3, 9, 3, 4]
    assert list(b) == [6, 10, 2, 1]
    assert value.device is None


@pytest.mark.parametrize("decorate", [lambda fn: fn, function, jit], ids=["eager", "function", "jit"])
def test_simplified_output_keeps_requested_device(decorate):
    ring = RingSpec(n=4, moduli=(17, 97))

    @decorate
    def rotate_constant(x):
        return (x * 0 + 7).rotate(1).intt()

    for device in dict.fromkeys(("python", Device.DEFAULT)):
        source = Poly([1, 2, 3, 4], ring, device=device).ntt().realize()
        result = rotate_constant(source).realize()
        assert result.device == device
        assert list(result) == [0, 7, 0, 0]


@pytest.mark.parametrize("power", [0, 1])
def test_functions_and_jit_accept_device_free_constants(power):
    ring = RingSpec(n=4, moduli=(17, 97))

    captures = 0

    @function
    def add(x, c):
        nonlocal captures
        captures += 1
        return x + c, Poly.const(3, ring)

    run = jit(add)
    x = Poly([1, 2, 3, 4], ring, device=Device.DEFAULT)
    for c in (7, 8, 7):
        y, constant = run(x, Poly.monomial(c, power, ring))
        expected = [1, 2, 3, 4]
        expected[power] += c
        assert list(y) == expected
        assert list(constant) == [3, 0, 0, 0]
        assert y.device == constant.device == Device.DEFAULT
    assert captures == 2


def test_constant_only_jit_uses_active_device():
    ring = RingSpec(n=4, moduli=(17, 97))
    run = jit(function(lambda x: x + 1))
    value = Poly.const(7, ring)
    for device in ("python", Device.DEFAULT, "python"):
        with Device.scope(device):
            result = run(value)
            assert result.device == device
            assert list(result) == [8, 0, 0, 0]


def test_pickle_device_free_expression():
    ring = RingSpec(n=4, moduli=(17, 97))
    for value in (Poly.const(7, ring).ntt() + 2, Poly.monomial(7, 1, ring) + 2):
        restored = pickle.loads(pickle.dumps(value))
        assert restored.tolist() == value.tolist()
        assert value.device is None
