from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.timeframes import EpochBucket, SessionBucket, get, ordered

NY = ZoneInfo("America/New_York")


def epoch(y, m, d, hh=0, mm=0, tz=NY):
    return int(datetime(y, m, d, hh, mm, tzinfo=tz).timestamp())


class TestEpochBucket:
    def test_floors_to_interval(self):
        bucket = EpochBucket(300)
        # 10:07:33 UTC -> 10:05:00
        ts = epoch(2026, 3, 10, 10, 7, tz=ZoneInfo("UTC")) + 33
        assert bucket.start(ts) == epoch(2026, 3, 10, 10, 5, tz=ZoneInfo("UTC"))

    def test_exact_boundary_stays_put(self):
        bucket = EpochBucket(900)
        ts = epoch(2026, 3, 10, 14, 30, tz=ZoneInfo("UTC"))
        assert bucket.start(ts) == ts
        assert bucket.end(ts) == ts + 900

    def test_hour_and_four_hour_align(self):
        assert EpochBucket(3600).start(1_700_000_123) % 3600 == 0
        assert EpochBucket(14400).start(1_700_000_123) % 14400 == 0


class TestSessionBucket:
    bucket = SessionBucket("America/New_York", 18)

    def test_after_open_belongs_to_new_session(self):
        # Tue 19:00 ET -> session that opened Tue 18:00 ET
        ts = epoch(2026, 3, 10, 19, 0)
        assert self.bucket.start(ts) == epoch(2026, 3, 10, 18, 0)

    def test_before_open_belongs_to_previous_session(self):
        # Wed 09:30 ET -> session that opened Tue 18:00 ET
        ts = epoch(2026, 3, 11, 9, 30)
        assert self.bucket.start(ts) == epoch(2026, 3, 10, 18, 0)

    def test_exactly_at_open_starts_new_session(self):
        ts = epoch(2026, 3, 10, 18, 0)
        assert self.bucket.start(ts) == ts

    def test_end_is_next_open(self):
        start = epoch(2026, 3, 10, 18, 0)
        assert self.bucket.end(start) == epoch(2026, 3, 11, 18, 0)

    def test_session_length_absorbs_dst_shift(self):
        # US DST begins Sunday 2026-03-08; the session spanning it is 23h.
        start = self.bucket.start(epoch(2026, 3, 7, 20, 0))
        assert self.bucket.end(start) - start == 23 * 3600

    def test_all_times_in_a_session_share_one_bucket(self):
        start = epoch(2026, 6, 1, 18, 0)
        for offset_hours in (0, 1, 6, 12, 20, 22):
            ts = start + offset_hours * 3600
            assert self.bucket.start(ts) == start


class TestRegistry:
    def test_expected_timeframes_present(self):
        keys = [tf.key for tf in ordered()]
        assert keys == ["1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"]

    def test_lookup(self):
        assert get("15m").nominal_seconds == 900
