"""
Unit tests for the PruneQuest Unified Event System (UX Release B).
Tests OptimizationEvent schema, EventBus pub/sub, subscriber isolation, and EventLogger.
"""

import os
import json
import tempfile
from prunequest.events.models import OptimizationEvent, EventCategory, EventPhase
from prunequest.events.bus import EventBus, EventLogger


def test_event_schema_and_serialization():
    evt = OptimizationEvent.create(
        run_id="run_test_123",
        sequence=1,
        event_type=EventCategory.RUN_STARTED,
        phase=EventPhase.PREFLIGHT,
        payload={"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "benchmark": "gsm8k"},
    )
    assert evt.run_id == "run_test_123"
    assert evt.sequence == 1
    assert evt.event_type == "run_started"
    assert evt.phase == "preflight"
    assert evt.payload["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"

    # Test dictionary serialization
    d = evt.to_dict()
    assert d["sequence"] == 1
    assert d["event_type"] == "run_started"

    # Test JSON serialization and round-trip
    json_str = evt.to_json()
    reconstructed = OptimizationEvent.from_json(json_str)
    assert reconstructed.run_id == evt.run_id
    assert reconstructed.sequence == evt.sequence
    assert reconstructed.event_type == evt.event_type
    assert reconstructed.payload == evt.payload


def test_event_bus_pub_sub_and_ordering():
    bus = EventBus("run_seq_test")
    received_events = []

    def on_event(event: OptimizationEvent):
        received_events.append(event)

    bus.subscribe(on_event)

    # Emit series of events
    bus.emit(EventCategory.RUN_STARTED, EventPhase.PREFLIGHT, {"step": 1})
    bus.emit(EventCategory.MODEL_LOADING_STARTED, EventPhase.MODEL_LOADING, {"step": 2})
    bus.emit(EventCategory.BASELINE_COMPLETED, EventPhase.BASELINE, {"accuracy": 0.95})

    assert len(received_events) == 3
    assert received_events[0].sequence == 1
    assert received_events[1].sequence == 2
    assert received_events[2].sequence == 3
    assert received_events[0].event_type == EventCategory.RUN_STARTED
    assert received_events[2].payload["accuracy"] == 0.95

    # Test unsubscribe
    bus.unsubscribe(on_event)
    bus.emit(EventCategory.RUN_COMPLETED, EventPhase.COMPLETED)
    assert len(received_events) == 3  # Did not receive 4th event


def test_event_bus_listener_isolation():
    bus = EventBus("run_err_test")
    valid_events = []

    def bad_listener(event: OptimizationEvent):
        raise RuntimeError("Presentation error in listener")

    def good_listener(event: OptimizationEvent):
        valid_events.append(event)

    bus.subscribe(bad_listener)
    bus.subscribe(good_listener)

    # Emitting an event must not raise exception even if bad_listener fails
    bus.emit(EventCategory.RUN_STARTED, EventPhase.PREFLIGHT)
    assert len(valid_events) == 1
    assert valid_events[0].event_type == EventCategory.RUN_STARTED


def test_event_logger_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "events.jsonl")
        logger = EventLogger(log_path)

        evt1 = OptimizationEvent.create("run_log", 1, EventCategory.RUN_STARTED, EventPhase.PREFLIGHT, {"test": True})
        evt2 = OptimizationEvent.create("run_log", 2, EventCategory.CANDIDATE_ACCEPTED, EventPhase.VALIDATION, {"gain": 0.1})

        logger.handle_event(evt1)
        logger.handle_event(evt2)
        logger.close()

        assert os.path.exists(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["event_type"] == EventCategory.RUN_STARTED
        assert lines[1]["event_type"] == EventCategory.CANDIDATE_ACCEPTED
        assert lines[1]["payload"]["gain"] == 0.1
