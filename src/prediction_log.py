"""Prediction log — every served prediction lands in Postgres.

This is the substrate the whole sprint stands on: drift detection, the accuracy
curve and the CT trigger all read this table. Without a record of what production
actually saw, "monitoring" is just a latency chart.

Three design decisions worth defending in a review
--------------------------------------------------
1. NON-BLOCKING BY CONSTRUCTION. Writes go to a bounded in-memory queue drained
   by a background thread; /predict never waits on the database and never fails
   because of it. If the queue saturates we DROP rows and count the drops. That
   is the correct trade for observability data: a monitoring pipeline must never
   be able to take down the service it observes. The drop counter is exposed as
   a metric, because silent loss would make the drift numbers quietly wrong.

2. SCHEMA `monitoring` INSIDE THE EXISTING MLflow DATABASE. Not a second
   Postgres, and not a new database: the pg volume is already initialized, so
   docker-entrypoint-initdb.d scripts would never run again, and creating one
   would mean `docker compose down -v` — which destroys the run history. A
   dedicated schema gives the separation (distinct namespace, own grants, own
   lifecycle) at zero cost to existing state.

3. THE TABLE IS GENERATED FROM image_features.FEATURE_NAMES. Columns, INSERT
   statement and value tuple all derive from that one list, so a new feature
   cannot end up in the log with the schema and the writer disagreeing about
   column order — the kind of bug that produces plausible, wrong drift.

Degraded mode mirrors the model-loading policy in api.py: if Postgres is
unreachable the API boots, serves, and retries in the background. Observability
is important; it is not more important than serving.
"""

import os
import queue
import threading
import time
from typing import Dict, List, Optional

from image_features import FEATURE_NAMES

# Host-side default. Port 55433 is the compose-published port, deliberately not
# 5432: a native PostgreSQL install commonly owns 5432, and connecting to the
# wrong database is a failure mode that reports itself very badly (see
# _describe_error). Inside the compose network the API overrides this with
# TERRAOPS_DB_URI=postgresql://...@postgres:5432/mlflow.
DEFAULT_DB_URI = "postgresql://mlflow:mlflow@localhost:55433/mlflow"
DB_URI = os.environ.get("TERRAOPS_DB_URI", DEFAULT_DB_URI)

SCHEMA = "monitoring"
TABLE = f"{SCHEMA}.predictions"

# Fixed columns, then the generated feature columns. Order matters: it is reused
# verbatim to build the INSERT and to pack each row's values.
_FIXED_COLUMNS = [
    "model_version",     # WHICH registry version answered — joins drift to a release
    "endpoint",          # /predict or /predict/batch
    "source",            # free-form traffic tag (ui, sim:cloud:0.4, ...) — lets the
                         # drift simulator's traffic be isolated from real uploads
    "predicted_class",
    "confidence",
    "entropy",           # normalized softmax entropy: the model-side drift signal
    "latency_ms",
    "batch_size",
    "n_bytes",
    "width",
    "height",
]
COLUMNS = _FIXED_COLUMNS + FEATURE_NAMES

_DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

CREATE TABLE IF NOT EXISTS {TABLE} (
    id             BIGSERIAL PRIMARY KEY,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version  TEXT NOT NULL,
    endpoint       TEXT NOT NULL,
    source         TEXT,
    predicted_class TEXT NOT NULL,
    confidence     REAL NOT NULL,
    entropy        REAL NOT NULL,
    latency_ms     REAL NOT NULL,
    batch_size     INTEGER NOT NULL,
    n_bytes        INTEGER,
    width          INTEGER,
    height         INTEGER,
    {", ".join(f"{name} REAL" for name in FEATURE_NAMES)}
);

-- Every monitoring query is "the last N minutes/rows", so ts DESC is the access
-- path that matters; source is in the index because drift reports filter on it.
CREATE INDEX IF NOT EXISTS predictions_ts_idx ON {TABLE} (ts DESC);
CREATE INDEX IF NOT EXISTS predictions_source_ts_idx ON {TABLE} (source, ts DESC);
"""

_INSERT = (
    f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(COLUMNS))})"
)


class PredictionLogger:
    """Bounded queue + single writer thread. Start once, call log() from anywhere."""

    def __init__(self, db_uri: str = DB_URI, max_queue: int = 10_000,
                 batch_size: int = 50, flush_interval: float = 2.0):
        self.db_uri = db_uri
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._conn = None

        # Counters exposed to /metrics — a monitoring pipeline that loses rows
        # without saying so corrupts every downstream conclusion.
        self.written = 0
        self.dropped = 0
        self.failed = 0
        self.ready = False
        self.last_error: Optional[str] = None

    @property
    def queue_depth(self) -> int:
        """Rows waiting to be written — a rising depth means the DB is the bottleneck."""
        return self._queue.qsize()

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Create the schema if possible, then start the writer thread.

        Never raises: a database that is down must not prevent the API from
        serving. The thread keeps retrying, so a Postgres that comes up late
        (compose start order) is picked up without a restart.
        """
        # Clearing the stop flag makes start/stop/start valid — the API's test
        # suite builds several TestClient contexts against this same module-level
        # logger, and a sticky flag would leave the second one with a thread that
        # exits immediately and a queue nobody drains.
        self._stop.clear()
        self._ensure_schema()
        self._thread = threading.Thread(target=self._run, name="prediction-log",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Flush what is queued, then stop — used on API shutdown and in tests."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._close()

    # --- connection ----------------------------------------------------------

    def _connect(self):
        import psycopg2
        return psycopg2.connect(self.db_uri, connect_timeout=3)

    @staticmethod
    def _describe_error(exc: Exception) -> str:
        """Turn a connection failure into something a human can act on.

        The motivating case, hit while wiring this up: another PostgreSQL was
        already listening on the published port, authentication failed against
        the wrong database, and libpq returned its error in the Windows ANSI
        codepage. psycopg2 then raised a UnicodeDecodeError — an exception that
        names an encoding problem and says nothing about the actual cause. That
        is an hour lost for anyone who has not seen it before, so the hint is
        attached here rather than left to be rediscovered.
        """
        if isinstance(exc, UnicodeDecodeError):
            return ("UnicodeDecodeError while reading the server's error "
                    "message — this almost always means the CONNECTION failed "
                    "and libpq reported it in a non-UTF-8 locale. Check that "
                    "nothing else owns the port (a native PostgreSQL on 5432 is "
                    "the usual culprit) and that TERRAOPS_DB_URI points at the "
                    "TerraOps database.")
        return f"{type(exc).__name__}: {exc}"

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _ensure_schema(self) -> bool:
        """Idempotent CREATE SCHEMA/TABLE/INDEX. Returns False if the DB is down.

        The API owns this DDL rather than a migration tool because the table is
        append-only and derived from FEATURE_NAMES; a schema that regenerates
        itself from the code cannot drift away from the code.
        """
        try:
            conn = self._connect()
            with conn, conn.cursor() as cur:
                cur.execute(_DDL)
            conn.close()
            self.ready = True
            self.last_error = None
            return True
        except Exception as exc:
            self.ready = False
            self.last_error = self._describe_error(exc)
            return False

    # --- producer side (called from request handlers) ------------------------

    def log(self, row: Dict[str, object]) -> None:
        """Enqueue one prediction record. Never blocks, never raises.

        put_nowait + count-the-drop is deliberate: blocking here would push
        database latency straight into the p95 of /predict, which is the metric
        the same monitoring stack is supposed to be measuring.
        """
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    # --- consumer side -------------------------------------------------------

    def _run(self) -> None:
        """Drain the queue in batches until stopped, reconnecting as needed."""
        pending: List[Dict[str, object]] = []
        last_flush = time.monotonic()

        while not self._stop.is_set() or not self._queue.empty() or pending:
            timeout = 0.5
            try:
                pending.append(self._queue.get(timeout=timeout))
            except queue.Empty:
                pass

            due = (len(pending) >= self.batch_size
                   or (pending and time.monotonic() - last_flush >= self.flush_interval)
                   or (pending and self._stop.is_set()))
            if due:
                if self._flush(pending):
                    pending = []
                    last_flush = time.monotonic()
                elif len(pending) > self.batch_size * 10:
                    # DB has been down long enough to threaten memory: shed the
                    # oldest records and account for them. Bounded loss beats an
                    # unbounded buffer taking the API down with it.
                    self.dropped += len(pending)
                    pending = []
                    last_flush = time.monotonic()
                else:
                    # Back off so a dead database is not hammered every 500 ms.
                    self._stop.wait(2.0)

    def _flush(self, rows: List[Dict[str, object]]) -> bool:
        """Write a batch. Returns False if it could not be written (rows kept)."""
        if not rows:
            return True
        try:
            if self._conn is None or self._conn.closed:
                self._conn = self._connect()
                # A late-starting Postgres may never have received the DDL.
                with self._conn, self._conn.cursor() as cur:
                    cur.execute(_DDL)
            values = [tuple(r.get(col) for col in COLUMNS) for r in rows]
            with self._conn, self._conn.cursor() as cur:
                cur.executemany(_INSERT, values)
            self.written += len(rows)
            self.ready = True
            self.last_error = None
            return True
        except Exception as exc:
            self.failed += 1
            self.ready = False
            self.last_error = self._describe_error(exc)
            self._close()
            return False


def build_row(*, model_version: str, endpoint: str, source: Optional[str],
              predicted_class: str, confidence: float, entropy: float,
              latency_ms: float, batch_size: int, n_bytes: Optional[int],
              width: Optional[int], height: Optional[int],
              features: Dict[str, float]) -> Dict[str, object]:
    """Assemble one log row, keyword-only so no caller can transpose two fields.

    Unknown feature keys are ignored and missing ones become NULL rather than
    raising: a feature-extraction change must degrade the log, never the API.
    """
    row: Dict[str, object] = {
        "model_version": model_version,
        "endpoint": endpoint,
        "source": source,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "entropy": float(entropy),
        "latency_ms": float(latency_ms),
        "batch_size": int(batch_size),
        "n_bytes": n_bytes,
        "width": width,
        "height": height,
    }
    for name in FEATURE_NAMES:
        value = features.get(name)
        row[name] = float(value) if value is not None else None
    return row
