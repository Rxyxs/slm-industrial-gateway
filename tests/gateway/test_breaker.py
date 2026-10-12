import pytest

from src.gateway.breaker import BreakerState, CircuitBreaker, CircuitOpenError


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


async def ok():
    return "ok"


async def boom():
    raise RuntimeError("down")


async def test_opens_after_threshold_and_rejects_without_calling():
    clock = Clock()
    br = CircuitBreaker("edge", failure_threshold=3, reset_timeout_s=10, clock=clock)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await br.call(boom)
    assert br.state == BreakerState.OPEN
    called = False

    async def spy():
        nonlocal called
        called = True

    with pytest.raises(CircuitOpenError):
        await br.call(spy)
    assert not called and br.total_rejections == 1


async def test_success_resets_consecutive_count():
    br = CircuitBreaker("edge", failure_threshold=2, clock=Clock())
    with pytest.raises(RuntimeError):
        await br.call(boom)
    await br.call(ok)
    with pytest.raises(RuntimeError):
        await br.call(boom)
    assert br.state == BreakerState.CLOSED


async def test_half_open_probe_closes_on_success():
    clock = Clock()
    br = CircuitBreaker("edge", failure_threshold=1, reset_timeout_s=5, clock=clock)
    with pytest.raises(RuntimeError):
        await br.call(boom)
    clock.t = 4.9
    assert br.state == BreakerState.OPEN
    clock.t = 5.0
    assert br.state == BreakerState.HALF_OPEN
    assert await br.call(ok) == "ok"
    assert br.state == BreakerState.CLOSED


async def test_half_open_probe_failure_reopens_and_restarts_timer():
    clock = Clock()
    br = CircuitBreaker("edge", failure_threshold=1, reset_timeout_s=5, clock=clock)
    with pytest.raises(RuntimeError):
        await br.call(boom)
    clock.t = 6.0
    with pytest.raises(RuntimeError):
        await br.call(boom)
    assert br.state == BreakerState.OPEN
    clock.t = 10.9
    assert br.state == BreakerState.OPEN
    clock.t = 11.0
    assert br.state == BreakerState.HALF_OPEN


def test_invalid_threshold():
    with pytest.raises(ValueError):
        CircuitBreaker("x", failure_threshold=0)
