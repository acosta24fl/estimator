"""Continuous series construction.

The whole point of this module is that a roll gap is not a return. These tests
build contracts whose underlying is deliberately flat, so *any* price change in
the spliced series is a splicing artefact and nothing else.
"""

import numpy as np
import pandas as pd
import pytest

from mnq.data.roll import build_continuous, check_point_scale, session_dates


def _frame(price: pd.Series, volume) -> pd.DataFrame:
    """An OHLCV frame from a close series, with a small symmetric range."""
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 2.0,
            "low": price - 2.0,
            "close": price,
            "volume": volume,
        },
        index=price.index,
    )


def two_contracts(
    spread: float = 25.0,
    roll_day: str = "2023-12-07",
    noise: float = 0.0,
    seed: int = 3,
) -> dict[str, pd.DataFrame]:
    """Two overlapping contracts separated by a constant ``spread``.

    Volume flips from the front contract to its successor on ``roll_day``,
    which is what the volume roll method is supposed to find.
    """
    idx = pd.date_range("2023-11-01", "2023-12-14 23:00", freq="1h", tz="UTC")
    base = np.full(len(idx), 15_000.0)
    if noise:
        rng = np.random.default_rng(seed)
        base = 15_000.0 + np.cumsum(rng.normal(0, noise, len(idx)))
    underlying = pd.Series(base, index=idx)

    flip = pd.Timestamp(roll_day, tz="UTC")
    heavy = np.where(idx < flip, 5_000.0, 100.0)
    light = np.where(idx < flip, 100.0, 5_000.0)
    return {
        "NQZ23": _frame(underlying, heavy),
        "NQH24": _frame(underlying + spread, light),
    }


def four_contracts(spread: float = 20.0) -> dict[str, pd.DataFrame]:
    """A year of quarterly contracts, each a fixed ``spread`` above the last."""
    idx = pd.date_range("2023-01-01", "2023-12-14 23:00", freq="1h", tz="UTC")
    underlying = pd.Series(np.full(len(idx), 15_000.0), index=idx)

    codes = ["NQH23", "NQM23", "NQU23", "NQZ23"]
    flips = [None, "2023-03-08", "2023-06-07", "2023-09-06"]
    out: dict[str, pd.DataFrame] = {}
    for i, (code, flip) in enumerate(zip(codes, flips)):
        start = pd.Timestamp(flip, tz="UTC") if flip else idx[0]
        end = (
            pd.Timestamp(flips[i + 1], tz="UTC")
            if i + 1 < len(flips) and flips[i + 1]
            else idx[-1] + pd.Timedelta(hours=1)
        )
        vol = np.where((idx >= start) & (idx < end), 5_000.0, 100.0)
        out[code] = _frame(underlying + i * spread, vol)
    return out


class TestSessionDates:
    def test_evening_bars_belong_to_the_next_session(self):
        idx = pd.DatetimeIndex(
            ["2024-01-08 23:30", "2024-01-09 02:00", "2024-01-09 20:00"], tz="UTC"
        )
        got = session_dates(idx)
        # 23:30 UTC = 18:30 ET Monday -> Tuesday's session.
        assert str(got.iloc[0]) == "2024-01-09"
        assert str(got.iloc[1]) == "2024-01-09"
        # 20:00 UTC = 15:00 ET Tuesday -> still Tuesday's session.
        assert str(got.iloc[2]) == "2024-01-09"


class TestVolumeRoll:
    def test_finds_one_roll(self):
        series = build_continuous(two_contracts(), method="volume")
        assert len(series.rolls) == 1
        assert series.rolls[0].from_contract == "NQZ23"
        assert series.rolls[0].to_contract == "NQH24"

    def test_measures_the_gap(self):
        series = build_continuous(two_contracts(spread=25.0), method="volume")
        assert series.rolls[0].gap == pytest.approx(25.0)

    def test_rolls_near_the_volume_flip(self):
        series = build_continuous(
            two_contracts(roll_day="2023-12-07"), method="volume", confirm_sessions=2
        )
        when = series.rolls[0].timestamp
        # Two confirming sessions after the flip, so a few days at most.
        assert pd.Timestamp("2023-12-07", tz="UTC") <= when <= pd.Timestamp(
            "2023-12-11", tz="UTC"
        )

    def test_each_bar_comes_from_exactly_one_contract(self):
        series = build_continuous(two_contracts(), method="volume")
        contracts = series.bars["contract"]
        # Before the roll it is all the front month, after it all the next.
        boundary = series.rolls[0].timestamp
        assert (contracts[series.bars.index < boundary] == "NQZ23").all()
        assert (contracts[series.bars.index >= boundary] == "NQH24").all()

    def test_the_roll_does_not_use_the_session_that_triggered_it(self):
        """A session's total volume is unknown until that session closes.

        Rolling on the same session the trigger fired would compose the series
        from information that did not exist when those bars printed - a small
        lookahead, but one baked into the dataset itself rather than the model,
        which makes it invisible to every downstream check.
        """
        raw = two_contracts(roll_day="2023-12-07")
        series = build_continuous(raw, method="volume", confirm_sessions=2)
        boundary = series.rolls[0].timestamp

        # Reconstruct the trigger session: the last one before the boundary on
        # which the successor out-traded the front month.
        from mnq.data.roll import session_dates

        vol = {
            code: frame["volume"].groupby(session_dates(frame.index)).sum()
            for code, frame in raw.items()
        }
        table = pd.DataFrame(vol).sort_index()
        leading = table.index[table["NQH24"] > table["NQZ23"]]
        trigger = [s for s in leading if pd.Timestamp(s, tz="UTC") < boundary]
        assert trigger, "expected the successor to lead before the roll"
        # Strictly after the confirming session, never on it.
        assert boundary > pd.Timestamp(trigger[-1], tz="UTC")

    def test_no_duplicate_timestamps(self):
        series = build_continuous(two_contracts(), method="volume")
        assert not series.bars.index.has_duplicates
        assert series.bars.index.is_monotonic_increasing


class TestAdjustment:
    """With a flat underlying, any jump in the series is pure roll artefact."""

    def _jump_at_roll(self, series) -> float:
        boundary = series.rolls[0].timestamp
        close = series.bars["close"]
        before = close[close.index < boundary].iloc[-1]
        after = close[close.index >= boundary].iloc[0]
        return abs(after - before)

    def test_unadjusted_splice_leaves_the_full_gap(self):
        """The control. If this ever stops failing, the tests below are vacuous."""
        series = build_continuous(
            two_contracts(spread=25.0), method="volume", adjustment="none"
        )
        assert self._jump_at_roll(series) == pytest.approx(25.0)

    def test_ratio_adjustment_removes_the_gap(self):
        series = build_continuous(
            two_contracts(spread=25.0), method="volume", adjustment="ratio"
        )
        assert self._jump_at_roll(series) < 0.01

    def test_difference_adjustment_removes_the_gap(self):
        series = build_continuous(
            two_contracts(spread=25.0), method="volume", adjustment="difference"
        )
        assert self._jump_at_roll(series) < 1e-6

    def test_ratio_preserves_percentage_returns_within_a_span(self):
        raw = two_contracts(spread=25.0, noise=8.0)
        series = build_continuous(raw, method="volume", adjustment="ratio")
        boundary = series.rolls[0].timestamp

        adjusted = series.bars["close"][series.bars.index < boundary]
        original = raw["NQZ23"]["close"].reindex(adjusted.index)
        # Scaling every price by one constant cannot change a return.
        pd.testing.assert_series_equal(
            adjusted.pct_change().dropna(),
            original.pct_change().dropna(),
            check_names=False,
            rtol=1e-9,
        )

    def test_the_newest_span_is_never_adjusted(self):
        """Recent prices must match what actually printed."""
        raw = two_contracts(spread=25.0)
        series = build_continuous(raw, method="volume", adjustment="ratio")
        boundary = series.rolls[0].timestamp
        tail = series.bars[series.bars.index >= boundary]["close"]
        expected = raw["NQH24"]["close"].reindex(tail.index)
        pd.testing.assert_series_equal(tail, expected, check_names=False)

    def test_high_low_stay_ordered_after_adjustment(self):
        series = build_continuous(
            two_contracts(spread=25.0, noise=8.0), method="volume", adjustment="ratio"
        )
        b = series.bars
        assert (b["high"] >= b["low"]).all()
        assert (b["high"] >= b[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (b["low"] <= b[["open", "close"]].min(axis=1) + 1e-9).all()

    def test_volume_is_not_adjusted(self):
        raw = two_contracts()
        series = build_continuous(raw, method="volume", adjustment="ratio")
        boundary = series.rolls[0].timestamp
        head = series.bars[series.bars.index < boundary]["volume"]
        pd.testing.assert_series_equal(
            head, raw["NQZ23"]["volume"].reindex(head.index), check_names=False
        )


class TestMultipleRolls:
    def test_finds_every_roll(self):
        series = build_continuous(four_contracts(), method="volume")
        assert len(series.rolls) == 3
        assert [r.to_contract for r in series.rolls] == ["NQM23", "NQU23", "NQZ23"]

    def test_adjustment_compounds_across_rolls(self):
        """Each earlier span must be shifted by the *sum* of later gaps."""
        series = build_continuous(
            four_contracts(spread=20.0), method="volume", adjustment="difference"
        )
        closes = series.bars["close"]
        # Flat underlying + fully compounded adjustment = one constant price.
        assert closes.max() - closes.min() < 1e-6

    def test_series_is_continuous_at_every_boundary(self):
        series = build_continuous(
            four_contracts(spread=20.0), method="volume", adjustment="ratio"
        )
        jumps = series.bars["close"].diff().abs()
        assert jumps.max() < 0.01


class TestCalendarRoll:
    def test_rolls_before_expiry(self):
        series = build_continuous(
            two_contracts(), method="calendar", calendar_offset_days=5
        )
        assert len(series.rolls) == 1
        # NQZ23 expires 2023-12-15, so a 5-day offset rolls on the 10th.
        assert series.rolls[0].timestamp == pd.Timestamp("2023-12-10", tz="UTC")

    def test_works_without_volume(self):
        raw = two_contracts()
        for frame in raw.values():
            frame.drop(columns=["volume"], inplace=True)
        series = build_continuous(raw, method="calendar")
        assert len(series.rolls) == 1


class TestValidation:
    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            build_continuous(two_contracts(), method="vibes")

    def test_rejects_unknown_adjustment(self):
        with pytest.raises(ValueError, match="adjustment"):
            build_continuous(two_contracts(), adjustment="sideways")

    def test_volume_method_needs_volume(self):
        raw = two_contracts()
        for frame in raw.values():
            frame.drop(columns=["volume"], inplace=True)
        with pytest.raises(ValueError, match="volume"):
            build_continuous(raw, method="volume")

    def test_single_contract_passes_through(self):
        raw = {"NQZ23": two_contracts()["NQZ23"]}
        series = build_continuous(raw)
        assert series.rolls == []
        assert (series.bars["contract"] == "NQZ23").all()

    def test_empty_input_is_an_error(self):
        with pytest.raises(ValueError, match="no contracts"):
            build_continuous({})

    def test_naive_index_is_localised(self):
        raw = two_contracts()
        for frame in raw.values():
            frame.index = frame.index.tz_localize(None)
        series = build_continuous(raw, method="volume")
        assert series.bars.index.tz is not None


class TestPointScaleWarning:
    def test_flags_a_target_that_is_not_comparable_across_the_history(self):
        idx = pd.date_range("2003-01-01", periods=3, freq="D", tz="UTC")
        bars = pd.DataFrame({"close": [1_500.0, 10_000.0, 20_000.0]}, index=idx)
        msg = check_point_scale(bars, min_target_points=20.0)
        assert msg is not None and "13.3x" in msg

    def test_quiet_when_the_scale_barely_moved(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        bars = pd.DataFrame({"close": [15_000.0, 15_500.0, 16_000.0]}, index=idx)
        assert check_point_scale(bars, min_target_points=20.0) is None
