from phosphor.core.events import EventBus, bus
from phosphor.observe import Metrics, Tracer


def test_metrics_counters_and_latency():
    m = Metrics()
    m.inc("q", 2)
    m.inc("q")
    with m.timer("lat"):
        pass
    snap = m.snapshot()
    assert snap["counters"]["q"] == 3
    assert "lat" in snap["latency"]


def test_percentile_of_single_value():
    from phosphor.core.mathx import percentile

    assert percentile([5.0], 0.5) == 5.0
    assert percentile([], 0.5) == 0.0


def test_bus_fans_out_to_subscribers():
    events = EventBus()
    seen = []
    events.subscribe("x", lambda e: seen.append(e.name))
    events.emit("x", k=1)
    assert seen == ["x"]


def test_tracer_collects_spans_by_trace():
    events = EventBus()
    tracer = Tracer()
    original = bus()
    try:
        # Re-point the shared tracer at a private bus for isolation.

        tracer._traces.clear()
        events.subscribe_all(tracer._on_event)
        events.emit("retrieval.search.start", trace_id="t1")
        events.emit("retrieval.search", trace_id="t1", elapsed_ms=3)
        spans = tracer.trace("t1")
        assert spans and spans[0]["name"] == "retrieval.search"
        assert spans[0]["duration_ms"] >= 0
    finally:
        del original


def test_metrics_reset():
    m = Metrics()
    m.inc("a")
    m.reset()
    assert m.snapshot()["counters"] == {}
