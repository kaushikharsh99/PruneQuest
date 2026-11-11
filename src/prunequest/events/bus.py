"""
Event Bus and Event Logging for PruneQuest.
Thread-safe pub/sub decoupled from presentation, tracking, and engine logic.
"""

import os
import threading
from typing import Callable, List, Optional, Dict, Any
from prunequest.events.models import OptimizationEvent


class EventBus:
    """
    Central event dispatcher for a single optimization run.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._listeners: List[Callable[[OptimizationEvent], None]] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._interrupted = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted

    def request_interruption(self):
        with self._lock:
            self._interrupted = True

    def subscribe(self, listener: Callable[[OptimizationEvent], None]):
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[OptimizationEvent], None]):
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit(
        self,
        event_type: str,
        phase: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> OptimizationEvent:
        with self._lock:
            self._seq += 1
            seq = self._seq

        event = OptimizationEvent.create(
            run_id=self.run_id,
            sequence=seq,
            event_type=event_type,
            phase=phase,
            payload=payload,
        )

        # Notify all listeners safely
        with self._lock:
            current_listeners = list(self._listeners)

        for listener in current_listeners:
            try:
                listener(event)
            except Exception:
                # Presentation / callback errors must never disrupt optimization
                pass

        return event


class EventLogger:
    """
    Persists optimization events sequentially to an events.jsonl file.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def handle_event(self, event: OptimizationEvent):
        line = event.to_json() + "\n"
        with self._lock:
            if self._file and not self._file.closed:
                self._file.write(line)
                self._file.flush()

    def close(self):
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()

    def __del__(self):
        self.close()
