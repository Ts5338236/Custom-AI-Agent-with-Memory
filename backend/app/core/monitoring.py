from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import time

# --- METRICS DEFINITION ---
REQUEST_COUNT = Counter(
    "api_requests_total", "Total count of API requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Latency of API requests", ["endpoint"]
)
AGENT_EXECUTION_COUNT = Counter(
    "agent_executions_total", "Total count of agent runs", ["agent_type", "status"]
)
MEMORY_ENTRIES = Gauge(
    "vector_db_entries_total", "Total number of entries in the long-term memory"
)

class MonitoringService:
    @staticmethod
    def log_request(method: str, endpoint: str, status: int, duration: float):
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    @staticmethod
    def log_agent_run(agent_type: str, success: bool):
        status = "success" if success else "failure"
        AGENT_EXECUTION_COUNT.labels(agent_type=agent_type, status=status).inc()

    @staticmethod
    def update_memory_count(count: int):
        MEMORY_ENTRIES.set(count)

    @staticmethod
    def get_metrics_app():
        return make_asgi_app()

metrics_service = MonitoringService()
