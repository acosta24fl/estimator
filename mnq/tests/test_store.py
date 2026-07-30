import json

from app.core.store import BarStore
from app.models import Bar

MINUTE = 60
BASE = 1_700_000_000 // MINUTE * MINUTE


def bar(ts, close=100.0, complete=True):
    return Bar(
        ts=ts, open=close, high=close + 1, low=close - 1, close=close,
        volume=5.0, complete=complete,
    )


def store_at(tmp_path):
    store = BarStore(tmp_path / "bars")
    store.ensure_dirs()
    return store


class TestMinuteLogging:
    def test_complete_bars_are_written(self, tmp_path):
        store = store_at(tmp_path)
        written = store.upsert_minute([bar(BASE), bar(BASE + MINUTE)])
        assert len(written) == 2
        assert store.logged_count == 2

    def test_forming_bar_is_not_written(self, tmp_path):
        store = store_at(tmp_path)
        written = store.upsert_minute([bar(BASE, complete=False)])
        assert written == []
        assert store.logged_count == 0
        assert store.minute_count == 1  # still queryable in memory

    def test_unchanged_bar_is_not_rewritten(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_minute([bar(BASE)])
        again = store.upsert_minute([bar(BASE)])
        assert again == []
        assert store.logged_count == 1

    def test_revision_appends_rather_than_edits(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_minute([bar(BASE, close=100.0)])
        store.upsert_minute([bar(BASE, close=101.0)])

        files = list((tmp_path / "bars" / "1m").glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2  # both versions retained, nothing overwritten
        assert json.loads(lines[-1])["close"] == 101.0

    def test_reload_takes_the_last_record(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_minute([bar(BASE, close=100.0)])
        store.upsert_minute([bar(BASE, close=101.0)])

        reloaded = BarStore(tmp_path / "bars")
        reloaded.load()
        assert reloaded.minute_series()[0].close == 101.0

    def test_survives_a_corrupt_line(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_minute([bar(BASE)])
        path = next((tmp_path / "bars" / "1m").glob("*.jsonl"))
        with path.open("a") as fh:
            fh.write("{not json}\n")

        reloaded = BarStore(tmp_path / "bars")
        reloaded.load()
        assert reloaded.minute_count == 1

    def test_timestamps_are_floored_to_the_minute(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_minute([bar(BASE + 37)])
        assert store.minute_series()[0].ts == BASE

    def test_series_is_sorted(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_minute([bar(BASE + 2 * MINUTE), bar(BASE), bar(BASE + MINUTE)])
        assert [b.ts for b in store.minute_series()] == [
            BASE, BASE + MINUTE, BASE + 2 * MINUTE
        ]

    def test_trims_to_the_cap(self, tmp_path):
        store = BarStore(tmp_path / "bars", max_1m_bars=10)
        store.ensure_dirs()
        store.upsert_minute([bar(BASE + i * MINUTE) for i in range(25)])
        assert store.minute_count == 10
        # The newest bars are the ones kept.
        assert store.minute_series()[-1].ts == BASE + 24 * MINUTE

    def test_bars_split_across_daily_files(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_minute([bar(BASE), bar(BASE + 2 * 86400)])
        files = sorted((tmp_path / "bars" / "1m").glob("*.jsonl"))
        assert len(files) == 2


class TestDaily:
    def test_appends_only_changed_bars(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_daily([bar(BASE, close=100.0)])
        assert store.upsert_daily([bar(BASE, close=100.0)]) == []
        assert len(store.upsert_daily([bar(BASE, close=105.0)])) == 1

    def test_reloads_from_disk(self, tmp_path):
        store = store_at(tmp_path)
        store.upsert_daily([bar(BASE, close=100.0), bar(BASE + 86400, close=110.0)])

        reloaded = BarStore(tmp_path / "bars")
        reloaded.load()
        assert [b.close for b in reloaded.daily_series()] == [100.0, 110.0]
