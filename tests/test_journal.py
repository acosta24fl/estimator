"""The decision log, and the liveness it feeds the dashboard.

The log exists so that a question asked next month - "why did it skip that
move?" - has an answer. Two properties matter more than the rest and are tested
hardest here:

1. **Writing must never break the caller.** A trading loop that dies because a
   disk filled up has turned an observability feature into an outage.
2. **A truncated final line must cost one record, not the file.** Killing the
   process mid-write is the normal way this program ends.
"""

import json

import pytest

from mnq.journal import DecisionLog, NullLog


@pytest.fixture
def log(tmp_path):
    return DecisionLog(tmp_path / "decisions.jsonl")


class TestWriting:
    def test_records_land_as_one_json_object_per_line(self, log):
        log.record("poll", "Pulled 3 bars.", new_bars=3)
        log.record("score", "Scored at 21000.", close=21000.0)

        lines = log.path.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["kind"] == "poll"
        assert first["summary"] == "Pulled 3 bars."
        assert first["new_bars"] == 3
        assert first["t"].endswith("+00:00")

    def test_every_record_carries_a_plain_english_summary(self, log):
        """The dashboard prints ``summary`` verbatim; it cannot be optional."""
        entry = log.record("gate", "No setup: 51% is short of the 58% needed.")
        assert entry["summary"]
        assert "%" in entry["summary"]

    def test_nan_never_reaches_the_file(self, log):
        """A bare NaN parses in Python and fails in every browser."""
        log.record("score", "s", p_long=float("nan"), p_short=float("inf"))
        text = log.path.read_text()
        assert "NaN" not in text and "Infinity" not in text
        assert json.loads(text)["p_long"] is None

    def test_numpy_scalars_are_coerced(self, log):
        np = pytest.importorskip("numpy")
        log.record("score", "s", a=np.int64(4), b=np.float32(1.5), c=np.bool_(True))
        row = json.loads(log.path.read_text())
        assert row["a"] == 4 and row["b"] == pytest.approx(1.5) and row["c"] is True

    def test_a_write_failure_does_not_raise(self, tmp_path):
        """The loop must survive a full or read-only disk.

        A log that can take the trading loop down with it has turned an
        observability feature into an outage.
        """
        # Aim the log at a directory: opening it for append raises OSError.
        blocked = DecisionLog(tmp_path)
        assert blocked.record("poll", "should not raise")["kind"] == "poll"
        assert blocked.tail(5) == []


class TestReading:
    def test_tail_returns_newest_first(self, log):
        for i in range(5):
            log.record("poll", f"event {i}", i=i)
        got = log.tail(3)
        assert [e["i"] for e in got] == [4, 3, 2]

    def test_tail_can_filter_by_kind(self, log):
        log.record("poll", "a")
        log.record("signal", "b")
        log.record("poll", "c")
        assert [e["summary"] for e in log.tail(10, kinds=["signal"])] == ["b"]

    def test_a_truncated_final_line_costs_one_record(self, log):
        """The process is normally killed with Ctrl+C, sometimes mid-write."""
        log.record("poll", "good one")
        with log.path.open("a") as fh:
            fh.write('{"kind": "poll", "summ')      # killed here
        got = log.tail(10)
        assert len(got) == 1 and got[0]["summary"] == "good one"

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert DecisionLog(tmp_path / "absent.jsonl").tail(10) == []

    def test_counts_tally_by_kind(self, log):
        log.record("poll", "a"); log.record("poll", "b"); log.record("exit", "c")
        assert log.counts() == {"poll": 2, "exit": 1}

    def test_stats_reports_the_newest_event(self, log):
        log.record("poll", "older")
        log.record("signal", "newest")
        stats = log.stats()
        assert stats["latest"]["summary"] == "newest"
        assert stats["bytes"] > 0

    def test_load_returns_a_time_indexed_frame(self, log):
        log.record("poll", "a", price=1.0)
        log.record("poll", "b", price=2.0)
        frame = log.load()
        assert list(frame["price"]) == [1.0, 2.0]
        assert frame.index.is_monotonic_increasing


class TestRotation:
    def test_the_file_rotates_instead_of_growing_forever(self, tmp_path):
        small = DecisionLog(tmp_path / "d.jsonl", max_bytes=400)
        for i in range(40):
            small.record("poll", f"padding padding padding {i}")
        assert small.path.with_suffix(".jsonl.1").exists()
        assert small.path.stat().st_size <= 400 + 200

    def test_rotation_keeps_the_log_readable(self, tmp_path):
        small = DecisionLog(tmp_path / "d.jsonl", max_bytes=400)
        for i in range(40):
            small.record("poll", f"padding padding padding {i}")
        assert small.tail(1)[0]["summary"].endswith("39")


class TestNullLog:
    def test_it_writes_nothing_and_still_answers(self):
        """Call sites log unconditionally; the guard is what gets forgotten."""
        null = NullLog()
        assert null.record("poll", "ignored")["summary"] == "ignored"
        assert null.tail(10) == [] and null.counts() == {}
