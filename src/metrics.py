"""Prometheus instrumentation for the serving API.

Division of labour with the prediction log (prediction_log.py)
--------------------------------------------------------------
Both observe the same requests, and that is not redundancy — they answer
different questions on different timescales:

  Prometheus   -> "is the service healthy RIGHT NOW?" Pre-aggregated counters
                  and histograms, seconds of latency, cheap, retention in days.
                  Cannot answer "what did request #4127 look like?".
  Postgres log -> "what exactly did production see?" Row-level, joinable,
                  replayable, retention in months. This is what Evidently reads.

Trying to do drift detection from Prometheus would mean shipping per-image
feature values as labels — unbounded cardinality, the classic way to kill a
Prometheus server. Trying to do alerting from Postgres would mean scanning a
growing table every 15 s. Each tool gets the job its data model fits.

On histogram buckets
--------------------
`histogram_quantile` interpolates INSIDE a bucket, so a p95 is only as accurate
as the bucket edges around it. The latency buckets below are placed around the
measured CPU serving cost (a single 224x224 forward pass on this machine is tens
to a couple hundred ms) and around the 400 ms budget already pinned in
params.yaml `nonreg.max_latency_ms` — so the alerting threshold falls on a
bucket EDGE rather than in the middle of a wide bucket where the reported p95
would be an interpolation artefact.

Label cardinality is deliberately bounded: endpoint (2 values), predicted_class
(10), model_version (a handful over the project's life). No request id, no
source tag, no image dimensions — those live in Postgres where a high-cardinality
column is just a column.
"""

from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# A DEDICATED registry rather than the global default: the API's metrics are the
# only thing on /metrics, tests can build a clean instance, and no library that
# happens to register into the default registry can pollute the output.
REGISTRY = CollectorRegistry()


PREDICTIONS = Counter(
    "terraops_predictions_total",
    "Predictions served, by endpoint, predicted class and serving model version.",
    ["endpoint", "predicted_class", "model_version"],
    registry=REGISTRY,
)

# The class distribution over time is the cheapest PREDICTION-drift signal there
# is: it needs no labels and no reference dataset, and a serving model that
# suddenly stops emitting a class is visible in one graph. It is a proxy for
# drift, not proof of it — a genuinely different input mix moves it too.

REQUESTS = Counter(
    "terraops_requests_total",
    "HTTP requests to the prediction endpoints, by endpoint and outcome.",
    ["endpoint", "outcome"],          # outcome: success | client_error | unavailable
    registry=REGISTRY,
)

LATENCY = Histogram(
    "terraops_prediction_latency_seconds",
    "Per-image inference latency (preprocessing + forward pass).",
    ["endpoint"],
    # Edges chosen around the CPU cost and around the 0.4 s non-regression
    # budget, so p95 alerting lands on an edge instead of inside a wide bucket.
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, float("inf")),
    registry=REGISTRY,
)

CONFIDENCE = Histogram(
    "terraops_prediction_confidence",
    "Top-1 softmax probability of served predictions.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)

ENTROPY = Histogram(
    "terraops_prediction_entropy",
    "Normalized Shannon entropy of the softmax vector (0 = certain, 1 = uniform).",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

# Rising mean entropy with a stable input distribution is the signature worth
# watching: the inputs look familiar but the model has stopped being sure. It is
# the one live signal that can catch what input-statistics drift misses.

MODEL_INFO = Gauge(
    "terraops_model_loaded",
    "1 when a champion is loaded, labelled with the registry version being served.",
    ["model_version"],
    registry=REGISTRY,
)

LOG_ROWS = Gauge(
    "terraops_prediction_log_rows",
    "Prediction-log row accounting, by state (written | dropped | failed_flushes).",
    ["state"],
    registry=REGISTRY,
)

LOG_QUEUE = Gauge(
    "terraops_prediction_log_queue_depth",
    "Rows waiting in the prediction-log queue (rising = the DB is the bottleneck).",
    registry=REGISTRY,
)


def observe_prediction(*, endpoint: str, predicted_class: str,
                       model_version: str, confidence: float, entropy: float,
                       latency_seconds: float) -> None:
    """Record one served image. Keyword-only: these are all floats and strings
    that would transpose silently."""
    PREDICTIONS.labels(endpoint=endpoint, predicted_class=predicted_class,
                       model_version=model_version).inc()
    LATENCY.labels(endpoint=endpoint).observe(latency_seconds)
    CONFIDENCE.observe(confidence)
    ENTROPY.observe(entropy)


def observe_request(endpoint: str, outcome: str) -> None:
    REQUESTS.labels(endpoint=endpoint, outcome=outcome).inc()


def set_model_version(version: Optional[str]) -> None:
    """Reflect the CURRENTLY served version, clearing the previous label set.

    Without the clear, a hot-swap via /reload would leave the old version's
    series at 1 forever and every dashboard would show two live champions.
    """
    MODEL_INFO.clear()
    if version is not None:
        MODEL_INFO.labels(model_version=version).set(1)


def sync_log_gauges(logger) -> None:
    """Copy the prediction logger's counters into gauges, at scrape time.

    Pull-at-scrape rather than push-on-write: these are cheap attribute reads,
    and doing it in the request path would put monitoring bookkeeping inside the
    latency it is measuring.
    """
    LOG_ROWS.labels(state="written").set(logger.written)
    LOG_ROWS.labels(state="dropped").set(logger.dropped)
    LOG_ROWS.labels(state="failed_flushes").set(logger.failed)
    LOG_QUEUE.set(logger.queue_depth)


def render() -> tuple[bytes, str]:
    """The /metrics payload: (body, content-type) in the Prometheus text format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
