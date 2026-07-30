from app.config import Settings
from app.core.engine import Engine
from app.core.store import BarStore
from app.feed.synthetic import SyntheticFeed


def engine_for(tmp_path, poll_seconds=10.0) -> Engine:
    settings = Settings(feed="synthetic", data_dir=tmp_path, poll_seconds=poll_seconds)
    return Engine(settings, SyntheticFeed(settings), BarStore(tmp_path / "bars"))


class TestPollBackoff:
    def test_healthy_feed_polls_at_the_configured_interval(self, tmp_path):
        engine = engine_for(tmp_path)
        assert engine.next_delay() == 10.0

    def test_each_consecutive_failure_doubles_the_wait(self, tmp_path):
        engine = engine_for(tmp_path)
        delays = []
        for _ in range(4):
            engine.consecutive_errors += 1
            delays.append(engine.next_delay())
        assert delays == [20.0, 40.0, 80.0, 160.0]

    def test_backoff_is_capped(self, tmp_path):
        engine = engine_for(tmp_path)
        engine.consecutive_errors = 50
        assert engine.next_delay() == engine.max_backoff_seconds

    def test_a_success_resets_the_backoff(self, tmp_path):
        engine = engine_for(tmp_path)
        engine.consecutive_errors = 5
        assert engine.next_delay() > 10.0
        engine.consecutive_errors = 0  # what _poll_once does on success
        assert engine.next_delay() == 10.0

    def test_status_exposes_the_backoff_state(self, tmp_path):
        engine = engine_for(tmp_path)
        engine.consecutive_errors = 2
        status = engine.status()
        assert status["consecutive_errors"] == 2
        assert status["next_poll_seconds"] == 40.0
