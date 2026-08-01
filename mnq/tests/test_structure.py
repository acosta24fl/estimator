from app.core.structure import as_of_labels, bias, classify, find_pivots, label_swings
from app.models import Bar

DAY = 86400
BASE = 1_700_000_000


def zig(levels):
    return [
        Bar(ts=BASE + i * DAY, open=l, high=l + 1, low=l - 1, close=l, complete=True)
        for i, l in enumerate(levels)
    ]


UPTREND = [20, 15, 10, 15, 20, 25, 30, 25, 20, 25, 30, 35, 40, 35, 30]
DOWNTREND = [20, 25, 30, 25, 20, 15, 10, 15, 20, 15, 10, 5, 0, 5, 10]


class TestPivots:
    def test_finds_a_peak(self):
        highs, _ = find_pivots(zig([1, 2, 5, 2, 1]), 2)
        assert highs == [2]

    def test_finds_a_trough(self):
        _, lows = find_pivots(zig([5, 4, 1, 4, 5]), 2)
        assert lows == [2]

    def test_labels_an_uptrend(self):
        highs, lows = label_swings(zig(UPTREND), 2)
        assert [h[2] for h in highs] == ["H", "HH"]
        assert [l[2] for l in lows] == ["L", "HL"]

    def test_labels_a_downtrend(self):
        highs, lows = label_swings(zig(DOWNTREND), 2)
        assert [h[2] for h in highs] == ["H", "LH"]
        assert [l[2] for l in lows] == ["L", "LL"]


class TestClassify:
    def test_uptrend(self):
        assert classify("HH", "HL") == "Uptrend"

    def test_downtrend(self):
        assert classify("LH", "LL") == "Downtrend"

    def test_missing_labels_are_a_range(self):
        assert classify(None, None) == "Range"

    def test_bias_is_signed_and_bounded(self):
        assert bias("Uptrend") == 1.0
        assert bias("Downtrend") == -1.0
        assert bias("Range") == 0.0
        assert bias("nonsense") == 0.0


class TestAsOfLabels:
    """The forecast fits on these, so they must never use future information."""

    def test_one_label_per_bar(self):
        assert len(as_of_labels(zig(UPTREND), 2)) == len(UPTREND)

    def test_early_bars_have_no_structure_yet(self):
        # No pivot can be confirmed before lookback*2 bars have printed.
        assert as_of_labels(zig(UPTREND), 2)[0] == "Range"

    def test_a_verdict_appears_only_once_its_pivots_are_confirmed(self):
        labels = as_of_labels(zig(UPTREND), 2)
        # Swings: low L@2, high H@6, low HL@8, high HH@12 — each confirmed
        # `lookback` (2) bars later. A verdict needs a labelled high *and* low,
        # and the first of each ("H"/"L") carries no direction.
        assert labels[9] == "Range"  # HL@8 not yet confirmed
        assert labels[10] == "Range (bullish tilt)"  # HL confirmed here
        assert labels[13] == "Range (bullish tilt)"  # HH@12 not yet confirmed
        assert labels[14] == "Uptrend"  # HH confirmed here

    def test_ends_in_the_expected_regime(self):
        assert as_of_labels(zig(UPTREND), 2)[-1] == "Uptrend"
        assert as_of_labels(zig(DOWNTREND), 2)[-1] == "Downtrend"

    def test_truncating_history_does_not_change_earlier_labels(self):
        """The defining property of an as-of series."""
        full = as_of_labels(zig(UPTREND), 2)
        for cut in (8, 10, 12):
            partial = as_of_labels(zig(UPTREND[:cut]), 2)
            # Labels are stable except within the unconfirmed tail.
            assert partial[: cut - 2] == full[: cut - 2]

    def test_empty_input(self):
        assert as_of_labels([], 2) == []
