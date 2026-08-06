from app.config import load_settings
from app.core.outlook import Outlook
from app.core.paper import LONG, SHORT, PaperTrade, PaperTrader
from app.core.timeframes import SessionBucket, get
from app.indicators.base import IndicatorContext
from app.indicators.paper_trades import PaperTrades
from app.models import Bar

STEP = 600
BASE = 1_700_000_000 // STEP * STEP


def bar(ts, o, c, complete=True):
    return Bar(ts=ts, open=o, high=max(o, c) + 1, low=min(o, c) - 1,
               close=c, volume=10, complete=complete)


def call(direction="bullish", confidence="medium"):
    return Outlook(direction=direction, horizon_minutes=10, confidence=confidence,
                   reason="test", expected_move=5.0)


def trader(tmp_path, cost=0.75):
    return PaperTrader(tmp_path / "trades.jsonl", cost_points=cost)


class TestEntry:
    def test_a_bullish_call_opens_a_long(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 101.0, complete=False)], call("bullish"), BASE + 5)
        assert t.open_trade.direction == LONG
        assert t.open_trade.entry == 100.0  # the forming bar's open

    def test_a_bearish_call_opens_a_short(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 99.0, complete=False)], call("bearish"), BASE + 5)
        assert t.open_trade.direction == SHORT

    def test_no_call_opens_nothing(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 101.0, complete=False)], call("neutral"), BASE + 5)
        assert t.open_trade is None

    def test_only_one_position_at_a_time(self, tmp_path):
        t = trader(tmp_path)
        bars = [bar(BASE, 100.0, 101.0, complete=False)]
        t.update(bars, call(), BASE + 5)
        t.update(bars, call(), BASE + 10)
        assert len(t.trades()) == 1

    def test_a_bar_is_only_traded_once(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 105.0, complete=False)], call(), BASE + 5)
        # bar completes, position closes, but the same bar must not re-enter
        t.update([bar(BASE, 100.0, 105.0, complete=True)], call(), BASE + STEP)
        assert len(t.trades()) == 1

    def test_a_stale_bar_is_not_entered(self, tmp_path):
        """Starting mid-bar must not fill at an open that already passed."""
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 101.0, complete=False)], call(), BASE + 500)
        assert t.open_trade is None

    def test_a_completed_bar_is_not_entered(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 101.0, complete=True)], call(), BASE + 5)
        assert t.open_trade is None


class TestExit:
    def test_a_long_closes_at_the_bar_close(self, tmp_path):
        t = trader(tmp_path, cost=0.0)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        t.update([bar(BASE, 100.0, 108.0, complete=True)], call("neutral"), BASE + STEP)
        done = t.closed()[0]
        assert done.exit == 108.0
        assert done.gross_points == 8.0

    def test_a_short_profits_when_price_falls(self, tmp_path):
        t = trader(tmp_path, cost=0.0)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call("bearish"), BASE + 5)
        t.update([bar(BASE, 100.0, 94.0, complete=True)], call("neutral"), BASE + STEP)
        assert t.closed()[0].gross_points == 6.0

    def test_costs_are_deducted(self, tmp_path):
        t = trader(tmp_path, cost=0.75)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        t.update([bar(BASE, 100.0, 110.0, complete=True)], call("neutral"), BASE + STEP)
        assert t.closed()[0].net_points == 10.0 - 0.75

    def test_the_exit_timestamp_is_the_horizon_end(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        t.update([bar(BASE, 100.0, 101.0, complete=True)], call("neutral"), BASE + STEP)
        assert t.closed()[0].closed_ts == BASE + STEP

    def test_a_new_bar_can_be_traded_after_the_exit(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        bars = [bar(BASE, 100.0, 101.0, complete=True),
                bar(BASE + STEP, 101.0, 101.0, complete=False)]
        t.update(bars, call(), BASE + STEP + 5)
        assert len(t.trades()) == 2
        assert t.open_trade.opened_ts == BASE + STEP


class TestUnrealised:
    def test_a_long_gains_as_price_rises(self, tmp_path):
        t = trader(tmp_path, cost=0.0)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        assert t.open_trade.unrealised(107.0) == 7.0

    def test_a_short_gains_as_price_falls(self, tmp_path):
        t = trader(tmp_path, cost=0.0)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call("bearish"), BASE + 5)
        assert t.open_trade.unrealised(93.0) == 7.0

    def test_costs_are_included_while_open(self, tmp_path):
        t = trader(tmp_path, cost=0.75)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        assert t.open_trade.unrealised(100.0) == -0.75


class TestLog:
    def test_trades_survive_a_restart(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        t.update([bar(BASE, 100.0, 110.0, complete=True)], call("neutral"), BASE + STEP)

        reloaded = PaperTrader(tmp_path / "trades.jsonl")
        reloaded.load()
        assert len(reloaded.closed()) == 1
        assert reloaded.closed()[0].exit == 110.0

    def test_the_close_supersedes_the_open_record(self, tmp_path):
        path = tmp_path / "trades.jsonl"
        t = PaperTrader(path)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        t.update([bar(BASE, 100.0, 110.0, complete=True)], call("neutral"), BASE + STEP)
        assert len(path.read_text().strip().split("\n")) == 2  # append-only

        reloaded = PaperTrader(path)
        reloaded.load()
        assert reloaded.open_trade is None  # last record wins

    def test_a_corrupt_line_is_skipped(self, tmp_path):
        path = tmp_path / "trades.jsonl"
        t = PaperTrader(path)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        with path.open("a") as fh:
            fh.write("{broken\n")
        reloaded = PaperTrader(path)
        assert reloaded.load() == 1


class TestStats:
    @staticmethod
    def _run(tmp_path, outcomes, cost=0.0):
        t = trader(tmp_path, cost=cost)
        for i, close in enumerate(outcomes):
            ts = BASE + i * STEP
            t.update([bar(ts, 100.0, 100.0, complete=False)], call(), ts + 5)
            t.update([bar(ts, 100.0, close, complete=True)], call("neutral"), ts + STEP)
        return t

    def test_counts_wins_and_losses(self, tmp_path):
        stats = self._run(tmp_path, [110.0, 90.0, 105.0]).stats()
        assert (stats.trades, stats.wins, stats.losses) == (3, 2, 1)

    def test_profit_factor(self, tmp_path):
        stats = self._run(tmp_path, [110.0, 95.0]).stats()  # +10, -5
        assert stats.profit_factor == 2.0

    def test_net_points(self, tmp_path):
        assert self._run(tmp_path, [110.0, 95.0]).stats().net_points == 5.0

    def test_summary_includes_the_open_position(self, tmp_path):
        t = trader(tmp_path, cost=0.0)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call(), BASE + 5)
        summary = t.summary(104.0)
        assert summary["open_position"]["side"] == "long"
        assert summary["open_unrealised"] == 4.0

    def test_summary_when_flat(self, tmp_path):
        summary = trader(tmp_path).summary(100.0)
        assert summary["open_position"] is None
        assert summary["trades"] == 0


class TestChartRendering:
    """The indicator that draws simulated trades on the candles."""

    @staticmethod
    def _ctx(trader_obj, tf_key="10m", bars=()):
        return IndicatorContext(
            timeframe=get(tf_key),
            bars=list(bars),
            minute_bars=[],
            daily_bars=[],
            session=SessionBucket("America/New_York", 18),
            settings=load_settings(),
            now=BASE,
            paper=trader_obj,
        )

    def _one_closed_trade(self, tmp_path):
        t = trader(tmp_path, cost=0.0)
        t.update([bar(BASE, 100.0, 100.0, complete=False)], call("bullish"), BASE + 5)
        t.update([bar(BASE, 100.0, 110.0, complete=True)], call("neutral"), BASE + STEP)
        return t

    def test_an_entry_draws_a_direction_arrow(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 101.0, complete=False)], call("bullish"), BASE + 5)
        markers = PaperTrades().compute(self._ctx(t)).markers
        assert [(m.shape, m.position) for m in markers] == [("arrowUp", "belowBar")]
        assert "BUY" in markers[0].text

    def test_a_bearish_entry_points_down(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 99.0, complete=False)], call("bearish"), BASE + 5)
        marker = PaperTrades().compute(self._ctx(t)).markers[0]
        assert (marker.shape, marker.position) == ("arrowDown", "aboveBar")
        assert "SELL" in marker.text

    def test_a_closed_trade_also_draws_an_exit(self, tmp_path):
        markers = PaperTrades().compute(
            self._ctx(self._one_closed_trade(tmp_path))
        ).markers
        assert [m.shape for m in markers] == ["arrowUp", "square"]
        assert markers[1].text == "+10.00"  # net points, signed

    def test_markers_snap_to_the_displayed_timeframe(self, tmp_path):
        """A 10m trade has no 1h candle of its own; without snapping the chart
        library silently drops the marker."""
        t = self._one_closed_trade(tmp_path)
        hour = get("1h").bucketer
        for marker in PaperTrades().compute(self._ctx(t, "1h")).markers:
            assert marker.time == hour.start(marker.time)

    def test_markers_are_sorted_by_time(self, tmp_path):
        t = self._one_closed_trade(tmp_path)
        times = [m.time for m in PaperTrades().compute(self._ctx(t)).markers]
        assert times == sorted(times)

    def test_an_open_position_draws_its_entry_line(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 101.0, complete=False)], call("bullish"), BASE + 5)
        bars = [bar(BASE - STEP, 90.0, 91.0), bar(BASE, 100.0, 101.0)]
        series = PaperTrades().compute(self._ctx(t, bars=bars)).series["entry_price"]
        # Only from the entry bar onward - not back across earlier candles.
        assert series == [{"time": BASE, "value": 100.0}]

    def test_a_flat_book_draws_no_entry_line(self, tmp_path):
        result = PaperTrades().compute(self._ctx(self._one_closed_trade(tmp_path)))
        assert result.series["entry_price"] == []

    def test_counts_are_whole_numbers(self, tmp_path):
        """`4.00 closed trades` would read as a price, not a count."""
        stats = PaperTrades().compute(self._ctx(self._one_closed_trade(tmp_path))).stats
        assert {s.key: s.precision for s in stats}["paper_trades"] == 0

    def test_the_open_position_is_reported(self, tmp_path):
        t = trader(tmp_path, cost=0.0)
        t.update([bar(BASE, 100.0, 101.0, complete=False)], call("bullish"), BASE + 5)
        bars = [bar(BASE, 100.0, 104.0, complete=False)]
        stats = {s.key: s.value for s in PaperTrades().compute(self._ctx(t, bars=bars)).stats}
        assert stats["paper_position"] == "LONG @ 100.00"
        assert stats["paper_open_pnl"] == 4.0

    def test_a_flat_book_says_so(self, tmp_path):
        stats = {s.key: s.value for s in PaperTrades().compute(self._ctx(trader(tmp_path))).stats}
        assert stats["paper_position"] == "Flat"
        assert stats["paper_none"] == 0

    def test_disabled_paper_trading_renders_a_hint_not_a_crash(self):
        result = PaperTrades().compute(self._ctx(None))
        assert result.markers == []
        assert [s.value for s in result.stats] == ["Off"]


class TestOpenPositionMarker:
    """The broker-style vertical marker on the candle a position opened on."""

    @staticmethod
    def _ctx(trader_obj, tf_key="10m", bars=()):
        return IndicatorContext(
            timeframe=get(tf_key),
            bars=list(bars),
            minute_bars=[],
            daily_bars=[],
            session=SessionBucket("America/New_York", 18),
            settings=load_settings(),
            now=BASE,
            paper=trader_obj,
        )

    def _open_long(self, tmp_path, cost=0.75):
        t = trader(tmp_path, cost=cost)
        t.update([bar(BASE, 100.0, 101.0, complete=False)], call("bullish"), BASE + 5)
        return t

    def test_a_flat_book_draws_no_marker(self, tmp_path):
        assert PaperTrades().compute(self._ctx(trader(tmp_path))).lines == []

    def test_an_open_position_draws_one(self, tmp_path):
        t = self._open_long(tmp_path)
        lines = PaperTrades().compute(self._ctx(t, bars=[bar(BASE, 100.0, 104.0, False)])).lines
        assert len(lines) == 1
        assert lines[0].label == "LONG @ 100.00"

    def test_it_sits_on_the_entry_bar(self, tmp_path):
        t = self._open_long(tmp_path)
        line = PaperTrades().compute(self._ctx(t, bars=[bar(BASE, 100.0, 104.0, False)])).lines[0]
        assert line.time == BASE

    def test_it_snaps_to_the_displayed_timeframe(self, tmp_path):
        """A line between candles reads as a position opened on nothing."""
        t = self._open_long(tmp_path)
        ctx = self._ctx(t, "1h", bars=[bar(BASE, 100.0, 104.0, False)])
        line = PaperTrades().compute(ctx).lines[0]
        assert line.time == get("1h").bucketer.start(BASE)

    def test_it_reports_points_moved_and_net_separately(self, tmp_path):
        """"How far has it gone" and "what is it worth" are different numbers,
        and only the second one carries the cost."""
        t = self._open_long(tmp_path, cost=0.75)
        line = PaperTrades().compute(self._ctx(t, bars=[bar(BASE, 100.0, 104.0, False)])).lines[0]
        assert "+4.00 pts from entry" in line.detail
        assert "+3.25 net" in line.detail

    def test_the_currency_sign_precedes_the_symbol(self, tmp_path):
        """'$-16.50' is what a bare +,.2f produces, and it reads as a typo."""
        t = trader(tmp_path, cost=0.75)
        t.update([bar(BASE, 100.0, 99.0, complete=False)], call("bullish"), BASE + 5)
        line = PaperTrades().compute(self._ctx(t, bars=[bar(BASE, 100.0, 90.0, False)])).lines[0]
        assert "(-$" in line.detail and "$-" not in line.detail

    def test_a_losing_position_is_toned_down(self, tmp_path):
        t = self._open_long(tmp_path)
        line = PaperTrades().compute(self._ctx(t, bars=[bar(BASE, 100.0, 90.0, False)])).lines[0]
        assert line.tone == "down"

    def test_a_winning_position_is_toned_up(self, tmp_path):
        t = self._open_long(tmp_path)
        line = PaperTrades().compute(self._ctx(t, bars=[bar(BASE, 100.0, 110.0, False)])).lines[0]
        assert line.tone == "up"

    def test_a_short_is_labelled_short(self, tmp_path):
        t = trader(tmp_path)
        t.update([bar(BASE, 100.0, 99.0, complete=False)], call("bearish"), BASE + 5)
        line = PaperTrades().compute(self._ctx(t, bars=[bar(BASE, 100.0, 95.0, False)])).lines[0]
        assert line.label.startswith("SHORT")
        assert line.tone == "up"  # a short profits as price falls

    def test_the_marker_survives_serialisation(self, tmp_path):
        t = self._open_long(tmp_path)
        payload = PaperTrades().compute(
            self._ctx(t, bars=[bar(BASE, 100.0, 104.0, False)])
        ).as_dict()
        assert payload["lines"][0]["label"] == "LONG @ 100.00"
        assert payload["lines"][0]["tone"] == "up"

    def test_moved_and_unrealised_both_reach_the_panel(self, tmp_path):
        t = self._open_long(tmp_path)
        stats = {s.key: s.value for s in PaperTrades().compute(
            self._ctx(t, bars=[bar(BASE, 100.0, 104.0, False)])).stats}
        assert stats["paper_open_move"] == 4.0
        assert stats["paper_open_pnl"] == 3.25
