import uuid
import time
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger("agent_tracing")
logger.setLevel(logging.INFO)

class Trace:
    def __init__(self, trace_id: str, query: str):
        self.trace_id = trace_id
        self.query = query
        self.start_time = time.time()
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, component: str, action: str, data: Any = None):
        step = {
            "timestamp": time.time() - self.start_time,
            "component": component,
            "action": action,
            "data": data
        }
        self.steps.append(step)
        logger.info(f"[{self.trace_id}] {component} -> {action}: {json.dumps(data) if data else ''}")

    def finalize(self):
        duration = time.time() - self.start_time
        logger.info(f"[{self.trace_id}] TRACE COMPLETE. Duration: {duration:.2f}s. Total Steps: {len(self.steps)}")
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "duration": duration,
            "steps": self.steps
        }

class TracingService:
    def __init__(self):
        self._active_traces: Dict[str, Trace] = {}

    def start_trace(self, query: str) -> str:
        trace_id = str(uuid.uuid4())
        self._active_traces[trace_id] = Trace(trace_id, query)
        return trace_id

    def get_trace(self, trace_id: str) -> Trace:
        return self._active_traces.get(trace_id)

    def log_step(self, trace_id: str, component: str, action: str, data: Any = None):
        trace = self.get_trace(trace_id)
        if trace:
            trace.add_step(component, action, data)

    def end_trace(self, trace_id: str) -> Dict[str, Any]:
        trace = self._active_traces.pop(trace_id, None)
        if trace:
            return trace.finalize()
        return {}

tracing_service = TracingService()
