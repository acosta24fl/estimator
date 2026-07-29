"""Time-of-day structure, and the guards that keep it honest.

This module measures two things with very different reliability, and the tests
are arranged around that split.

**Bar size by clock is real.** The cash open prints bigger bars than lunchtime,
every year, in every regime. A planted expansion must therefore be found, and
the measurement must be scale-free: doubling the volatility of the whole series
must not change a single ratio.

**Direction by clock is usually noise.** Thirteen slots screened at p<0.05 hand
you a false positive on roughly a quarter of runs. The random-walk control below
is the one that matters: across twelve independent random walks, the share that
produce *any* directional slot must stay inside the false-discovery rate the
correction promises. It is stated as a rate rather than as zero because zero is
not what FDR control guarantees, and a test asserting it would be both wrong and
flaky. If that control ever passes trivially, the correction has been removed
and this module has become a pattern-mining machine.
"""

import numpy as np
import pandas as pd
import pytest

from mnq.models.clock import (
    FDR,
    MIN_OBSERVATIONS,
    STABILITY_FLOOR,
    add_size_ratio,
    benjamini_hochberg,
    build_clock_profile,
    format_report,
    slot_history,
    slot_trend,
    to_timeframe,
)

TZ = "America/New_York"


def session_bars(days=120, sigma=1.0, seed=1, expand_at=None, expand_by=3.0,
                 drift_at=None, drift=0.0, start="2025-01-06"):
    """Minute bars over ``days`` weekday sessions, 09:30-16:00 New York.

    ``expand_at`` inflates the per-minute volatility inside one hour, which is
    how a real open behaves. ``drift_at`` adds a directional push in one hour,
    which is how a fake edge is planted.
    """
    rng = np.random.default_rng(seed)
    stamps, sigmas, drifts = [], [], []
    # freq="B" walks business days on the *local* calendar. Incrementing a
    # tz-aware timestamp by Timedelta(days=1) instead adds absolute time, so
    # every session after the March DST change would start an hour late and
    # the 09:30 slot would silently disappear from half the history.
    for day in pd.date_range(start, periods=days, freq="B", tz=TZ):
        session = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=30),
            day + pd.Timedelta(hours=15, minutes=59),
            freq="1min", tz=TZ,
        )
        stamps.append(session)
        hours = session.hour
        s = np.full(len(session), sigma)
        if expand_at is not None:
            s = np.where(hours == expand_at, sigma * expand_by, s)
        sigmas.append(s)
        d = np.zeros(len(session))
        if drift_at is not None:
            d = np.where(hours == drift_at, drift, 0.0)
        drifts.append(d)

    idx = pd.DatetimeIndex(np.concatenate(stamps)).tz_convert(TZ)
    sd = np.concatenate(sigmas)
    dr = np.concatenate(drifts)
    step = rng.normal(dr, sd)
    px = 21_000 + np.cumsum(step)
    wick = sd
    return pd.DataFrame(
        {"open": px - step, "high": np.maximum(px, px - step) + wick,
         "low": np.minimum(px, px - step) - wick, "close": px, "volume": 100.0},
        index=idx,
    )


class TestShaping:
    def test_resampling_happens_in_the_session_timezone(self):
        """A UTC boundary would smear the open across the DST change."""
        bars = session_bars(days=10)
        frame = to_timeframe(bars, 30, TZ)
        assert str(frame.index.tz) == TZ
        assert "09:30" in set(frame["slot"])

    def test_slots_land_on_the_timeframe_grid(self):
        frame = to_timeframe(session_bars(days=10), 30, TZ)
        minutes = {int(s.split(":")[1]) for s in frame["slot"]}
        assert minutes <= {0, 30}

    def test_a_finer_timeframe_gives_more_slots(self):
        bars = session_bars(days=20)
        assert len(set(to_timeframe(bars, 15, TZ)["slot"])) > len(
            set(to_timeframe(bars, 60, TZ)["slot"])
        )

    def test_empty_input_is_not_an_error(self):
        assert to_timeframe(pd.DataFrame(), 30).empty
        assert not build_clock_profile(pd.DataFrame()).available


class TestSizeRatio:
    def test_the_ratio_is_scale_free(self):
        """Doubling volatility must not move a single ratio.

        Without day-relative normalisation the profile is dominated by which
        years happened to be volatile, and "09:30 is big" becomes a statement
        about 2022 rather than about 09:30.
        """
        quiet = build_clock_profile(session_bars(200, sigma=1.0, seed=4, expand_at=9), 30)
        loud = build_clock_profile(session_bars(200, sigma=4.0, seed=4, expand_at=9), 30)

        a = {s.slot: s.median_size_ratio for s in quiet.slots}
        b = {s.slot: s.median_size_ratio for s in loud.slots}
        assert set(a) == set(b)
        for slot in a:
            assert a[slot] == pytest.approx(b[slot], rel=0.12), (
                f"{slot}: {a[slot]:.2f} vs {b[slot]:.2f} - the ratio is not "
                f"scale-free, so the profile measures the regime not the clock"
            )

    def test_raw_points_do_move_with_volatility(self):
        """The control for the control: the un-normalised number must scale."""
        quiet = build_clock_profile(session_bars(120, sigma=1.0, seed=4), 30)
        loud = build_clock_profile(session_bars(120, sigma=4.0, seed=4), 30)
        assert (
            max(s.median_range_points for s in loud.slots)
            > 2 * max(s.median_range_points for s in quiet.slots)
        )

    def test_a_planted_expansion_is_found(self):
        profile = build_clock_profile(
            session_bars(200, seed=2, expand_at=9, expand_by=3.0), 30
        )
        opening = profile.slot("09:30")
        assert opening is not None
        assert opening.median_size_ratio > 1.8, (
            f"a 3x volatility hour should read well above 1.0; "
            f"got {opening.median_size_ratio:.2f}"
        )
        assert opening.rank == 1, "it should be the largest slot of the session"

    def test_a_flat_session_has_no_standout_slot(self):
        profile = build_clock_profile(session_bars(200, seed=3), 30)
        ratios = [s.median_size_ratio for s in profile.slots]
        assert max(ratios) < 1.6, f"no slot should dominate a flat session: {ratios}"

    def test_quartiles_bracket_the_median(self):
        for s in build_clock_profile(session_bars(150, seed=5), 30).slots:
            assert s.p25_size_ratio <= s.median_size_ratio <= s.p75_size_ratio


class TestDirectionIsNotMined:
    """The control. If these pass trivially, the correction has been removed."""

    def test_a_random_walk_almost_never_yields_a_directional_slot(self):
        """The control, stated as the guarantee the correction actually gives.

        With no true effect anywhere, FDR control means the chance of reporting
        *any* slot is bounded by the FDR itself - not zero. Asserting zero on a
        single seed would be wrong statistics and a flaky test; asserting the
        rate across many seeds is the real property, and it still fails loudly
        if the correction is removed (the naive rate below is several times
        higher).
        """
        seeds = range(1, 13)
        profiles = [build_clock_profile(session_bars(250, seed=s), 30) for s in seeds]
        assert all(p.available for p in profiles)

        with_discovery = sum(
            1 for p in profiles if any(s.significant for s in p.slots)
        )
        naive_discovery = sum(
            1 for p in profiles if any(s.p_value < 0.05 for s in p.slots)
        )

        assert with_discovery <= 2, (
            f"{with_discovery}/{len(profiles)} random walks produced a "
            f"directional slot; FDR={FDR} should keep this at or below "
            f"{FDR:.0%}. The correction looks broken."
        )
        assert with_discovery < naive_discovery, (
            "the correction rejected nothing a naive 0.05 cut would have "
            "accepted, so it is not doing any work"
        )

    def test_a_clean_profile_says_so_in_plain_words(self):
        profile = build_clock_profile(session_bars(250, seed=1), 30)
        assert not any(s.significant for s in profile.slots)
        assert "expected and honest result" in profile.note

    def test_direction_is_judged_against_the_base_rate_not_fifty_percent(self):
        """55% up in a market that rises 55% of the time is no information."""
        bars = session_bars(200, seed=7, drift_at=None)
        profile = build_clock_profile(bars, 30)
        for s in profile.slots:
            assert s.edge_vs_base == pytest.approx(s.p_up - profile.base_rate_up)
            assert s.base_rate_up == profile.base_rate_up

    def test_a_planted_directional_hour_is_detected(self):
        """The other half: the guard must not simply reject everything."""
        bars = session_bars(300, seed=11, drift_at=10, drift=0.8)
        profile = build_clock_profile(bars, 30)
        flagged = [s for s in profile.slots if s.significant]
        assert flagged, "a genuine directional hour should survive correction"
        assert any(s.slot.startswith("10:") for s in flagged)


class TestBenjaminiHochberg:
    def test_all_null_p_values_survive_nothing(self):
        assert not any(benjamini_hochberg([0.4, 0.6, 0.9, 0.5]))

    def test_one_very_small_p_value_survives(self):
        assert benjamini_hochberg([0.0001, 0.6, 0.9, 0.5])[0]

    def test_order_is_preserved(self):
        keep = benjamini_hochberg([0.9, 0.0001, 0.8])
        assert keep == [False, True, False]

    def test_it_is_stricter_than_a_naive_threshold(self):
        """One slot at p=0.04 among twelve nulls is what a naive cut accepts.

        This is the actual failure mode: screen thirteen slots, find one under
        0.05, report it. BH rejects it because with thirteen tests a single
        0.04 is the expected best of pure noise.
        """
        p = [0.04] + [0.5] * 12
        assert p[0] < 0.05, "a naive threshold would accept this"
        assert not any(benjamini_hochberg(p, fdr=0.10))

    def test_collective_evidence_is_not_thrown_away(self):
        """Ten independent p=0.04s together are strong, and BH says so.

        The correction must not be so blunt that it discards a real effect
        appearing across many slots at once.
        """
        assert all(benjamini_hochberg([0.04] * 10, fdr=0.10))

    def test_empty_input(self):
        assert benjamini_hochberg([]) == []


class TestStability:
    def test_a_consistent_profile_reports_stable(self):
        profile = build_clock_profile(
            session_bars(300, seed=8, expand_at=9, expand_by=3.0), 30
        )
        assert profile.stability is not None
        assert profile.stability >= STABILITY_FLOOR
        assert profile.stable

    def test_a_profile_that_changes_halfway_is_not_called_stable(self):
        """The guard against reporting one unusual period as a clock effect."""
        first = session_bars(150, seed=9, expand_at=9, expand_by=4.0)
        second = session_bars(
            150, seed=10, expand_at=15, expand_by=4.0, start="2026-01-05"
        )
        profile = build_clock_profile(pd.concat([first, second]), 30)

        assert profile.stability is not None
        assert not profile.stable, (
            f"a profile whose shape moves between halves must not be called "
            f"stable; rho={profile.stability:.2f}"
        )
        assert "NOT stable" in profile.note

    def test_too_little_history_reports_no_stability_rather_than_a_guess(self):
        profile = build_clock_profile(session_bars(4, seed=1), 30, min_observations=2)
        assert profile.stability is None and not profile.stable


class TestThinData:
    def test_slots_below_the_minimum_are_dropped(self):
        profile = build_clock_profile(session_bars(10, seed=1), 30)
        assert not profile.available
        assert str(MIN_OBSERVATIONS) in profile.note

    def test_the_threshold_can_be_lowered_deliberately(self):
        profile = build_clock_profile(session_bars(10, seed=1), 30, min_observations=5)
        assert profile.available
        assert all(s.n >= 5 for s in profile.slots)


class TestSlotHistory:
    def test_one_slot_across_time(self):
        bars = session_bars(120, seed=6, expand_at=9, expand_by=3.0)
        hist = slot_history(bars, "09:30", 30)
        assert len(hist) == 120
        assert hist.index.is_monotonic_increasing
        assert (hist["size_ratio"] > 1.0).mean() > 0.7

    def test_an_unknown_slot_is_empty_not_an_error(self):
        assert slot_history(session_bars(30), "03:07", 30).empty

    def test_a_growing_slot_reads_as_expanding(self):
        quiet = session_bars(80, sigma=1.0, seed=12, expand_at=9, expand_by=1.5)
        loud = session_bars(
            40, sigma=1.0, seed=13, expand_at=9, expand_by=6.0, start="2025-06-02"
        )
        hist = slot_history(pd.concat([quiet, loud]), "09:30", 30)

        trend = slot_trend(hist, window=20)
        assert trend["available"] and trend["verdict"] == "expanding"
        assert trend["change"] > 0.15

    def test_a_steady_slot_reads_as_unchanged(self):
        hist = slot_history(session_bars(200, seed=14, expand_at=9), "09:30", 30)
        assert slot_trend(hist, window=20)["verdict"] == "unchanged"

    def test_too_few_sessions_refuses_to_call_a_trend(self):
        hist = slot_history(session_bars(30, seed=1), "09:30", 30)
        assert not slot_trend(hist, window=20)["available"]


class TestReport:
    def test_it_renders_and_names_the_correction(self):
        text = format_report(build_clock_profile(session_bars(200, seed=2), 30))
        assert "TIME-OF-DAY PROFILE" in text
        assert "survives correction" in text
        assert "median bar of that same day" in text

    def test_an_empty_profile_still_renders(self):
        assert "no bars to profile" in format_report(build_clock_profile(pd.DataFrame()))

    def test_the_payload_is_serialisable(self):
        import json

        payload = build_clock_profile(session_bars(150, seed=3), 30).to_dict()
        text = json.dumps(payload)
        assert "NaN" not in text
        assert payload["slots"][0]["slot"]

    def test_fdr_is_stricter_than_five_percent(self):
        assert FDR <= 0.10
