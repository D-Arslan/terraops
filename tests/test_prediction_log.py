"""Unit tests for the prediction log — WITHOUT a database.

Everything here is about the two properties that must hold whether Postgres is
up or not:

  1. the generated schema, the INSERT and the row dict agree on columns
     (a mismatch would write features into the wrong columns — plausible
     numbers, wrong meaning, undetectable downstream);
  2. a dead database degrades into counted drops, never into a failed or slowed
     prediction.

The write path against a real Postgres is exercised by the container smoke test
in CI, not here: a unit test that needs a database is not a unit test.
"""

import pytest

from image_features import FEATURE_NAMES
from prediction_log import COLUMNS, PredictionLogger, build_row


def _row(**overrides):
    kwargs = dict(
        model_version="1",
        endpoint="/predict",
        source="test",
        predicted_class="Forest",
        confidence=0.97,
        entropy=0.12,
        latency_ms=42.0,
        batch_size=1,
        n_bytes=1234,
        width=64,
        height=64,
        features={name: 0.5 for name in FEATURE_NAMES},
    )
    kwargs.update(overrides)
    return build_row(**kwargs)


# --- schema / row agreement ----------------------------------------------------

def test_row_covers_exactly_the_insert_columns():
    """The writer packs values by iterating COLUMNS; any gap becomes a NULL or a
    shifted value. Pinning the equality here makes adding a feature safe."""
    assert set(_row()) == set(COLUMNS)


def test_every_declared_feature_is_a_column():
    assert set(FEATURE_NAMES).issubset(set(COLUMNS))


def test_missing_feature_becomes_null_instead_of_raising():
    """Feature extraction changing shape must degrade the log, never the API."""
    row = _row(features={"brightness": 0.4})
    assert row["brightness"] == 0.4
    assert row["sharpness"] is None


def test_unknown_feature_is_ignored():
    row = _row(features={**{n: 0.1 for n in FEATURE_NAMES}, "not_a_column": 9.9})
    assert "not_a_column" not in row


def test_build_row_is_keyword_only():
    """Positional args would let two same-typed fields be transposed silently."""
    with pytest.raises(TypeError):
        build_row("1", "/predict", "test")  # type: ignore[misc]


# --- degraded behavior ---------------------------------------------------------

def test_log_never_raises_when_the_database_is_unreachable():
    """The whole point: monitoring cannot take down the thing it monitors."""
    logger = PredictionLogger(db_uri="postgresql://nobody:nobody@127.0.0.1:1/none")
    logger.start()          # DDL fails, thread starts anyway
    try:
        for _ in range(10):
            logger.log(_row())
        assert logger.ready is False
        assert logger.last_error is not None
    finally:
        logger.stop(timeout=2.0)


def test_full_queue_drops_and_counts_instead_of_blocking():
    """Bounded queue + counted drops. An unbounded buffer would trade a lost row
    for an out-of-memory kill of the API — a much worse failure."""
    logger = PredictionLogger(db_uri="postgresql://nobody@127.0.0.1:1/none",
                              max_queue=5)
    for _ in range(20):     # no writer thread started: nothing drains
        logger.log(_row())
    assert logger.queue_depth == 5
    assert logger.dropped == 15
