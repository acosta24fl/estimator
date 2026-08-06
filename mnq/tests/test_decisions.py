import json

import pytest

from app.core.decisions import (
    SCHEMA,
    Decision,
    DecisionLog,
    group_by,
    outcome_summary,
    value_at,
)
from app.core.paper import LONG, SHORT, PaperTrade

BASE = 1_700_000_000


def decision(trade_id="1", ts=BASE, direction=LONG, **kw) -> Decision:
    d = Decision(
        trade_id=trade_id,
        ts=ts,
        direction=direction,
        entry=100.0,
        symbol="MNQ=F",
        timeframe="10m",
        horizon_minutes=10,
        outlook={"direction": "bullish", "confidence": "low", "strength": 0.2},
        forecast={"expected_move": 3.5, "volatility": 20.0, "band_half": 12.0},
        factors=[
            {"key": "momentum", "label": "Momentum (MACD)", "score": 0.5,
             "coefficient": 2.0, "points": 1.0},
            {"key": "vwap", "label": "VWAP Distance", "score": -0.3,
             "coefficient": 1.0, "points": -0.3},
        ],
        fit={"samples": 500, "r2": 0.01},
        accuracy={"skill_score": -0.05, "direction_rate": 0.48},
        envelope={"sigma": 18.0, "range": 40.0},
        entry_study={"fill_model": "sustained", "offset": 0.0},
        market={"vwap_distance": -12.5},
        config={"cost_points": 0.75},
    )
    for key, value in kw.items():
        setattr(d, key, value)
    return d


def closed_trade(trade_id="1", net=5.0, ts=BASE):
    t = PaperTrade(trade_id=trade_id, direction=LONG, opened_ts=ts, entry=100.0,
                   cost_points=0.75, horizon_seconds=600)
    t.close(100.0 + net + 0.75, ts + 600)
    return t


class TestSnapshot:
    def test_it_records_the_fitted_coefficients(self):
        """The ridge refits every poll, so a trade taken on a momentum
        coefficient of +2.0 came from a different model than one at +0.2.
        Without this the decision is unreconstructable."""
        d = decision()
        assert d.factor("momentum")["coefficient"] == 2.0

    def test_an_unknown_factor_is_none_not_a_crash(self):
        assert decision().factor("nope") is None

    def test_the_side_is_derived_from_the_direction(self):
        assert decision(direction=SHORT).side == "short"
        assert decision(direction=LONG).side == "long"

    def test_it_round_trips_through_json(self):
        original = decision()
        restored = Decision.from_dict(json.loads(json.dumps(original.as_dict())))
        assert restored.as_dict() == original.as_dict()

    def test_the_schema_is_stamped(self):
        assert decision().as_dict()["schema"] == SCHEMA


class TestPersistence:
    def test_decisions_survive_a_restart(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(decision("7"))
        again = DecisionLog(tmp_path / "d.jsonl")
        assert again.load() == 1
        assert again.get("7").factor("momentum")["points"] == 1.0

    def test_a_decision_is_never_overwritten(self, tmp_path):
        """Beliefs are fixed at entry. A later poll rewriting the same id would
        record a decision that was never made."""
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(decision("7", entry=100.0))
        log.record(decision("7", entry=999.0))
        assert log.get("7").entry == 100.0
        assert len(log) == 1

    def test_a_corrupt_line_is_skipped(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps(decision("1").as_dict()) + "\n"
            "{not json}\n" + json.dumps(decision("2").as_dict()) + "\n",
            encoding="utf-8",
        )
        log = DecisionLog(path)
        assert log.load() == 2

    def test_nothing_is_written_when_persistence_is_off(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl", persist=False)
        log.record(decision("1"))
        assert not (tmp_path / "d.jsonl").exists()
        assert log.get("1") is not None

    def test_an_empty_log_loads_cleanly(self, tmp_path):
        assert DecisionLog(tmp_path / "missing.jsonl").load() == 0


class TestOutcomes:
    def test_closing_attaches_the_result(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(decision("1"))
        log.close(closed_trade("1", net=5.0))
        assert log.get("1").outcome["net_points"] == 5.0
        assert log.get("1").outcome["won"] is True

    def test_closing_does_not_disturb_the_beliefs(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(decision("1"))
        log.close(closed_trade("1", net=5.0))
        assert log.get("1").factor("momentum")["coefficient"] == 2.0
        assert log.get("1").fit["samples"] == 500

    def test_the_outcome_survives_a_restart(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(decision("1"))
        log.close(closed_trade("1", net=-3.0))
        again = DecisionLog(tmp_path / "d.jsonl")
        again.load()
        assert again.get("1").outcome["net_points"] == -3.0
        assert again.get("1").outcome["won"] is False

    def test_an_open_trade_is_not_closed(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(decision("1"))
        still_open = PaperTrade(trade_id="1", direction=LONG, opened_ts=BASE,
                                entry=100.0, cost_points=0.75, horizon_seconds=600)
        assert log.close(still_open) is None
        assert log.get("1").outcome is None

    def test_closing_an_unknown_trade_is_harmless(self, tmp_path):
        assert DecisionLog(tmp_path / "d.jsonl").close(closed_trade("nope")) is None

    def test_closed_filters_out_open_positions(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(decision("1", ts=BASE))
        log.record(decision("2", ts=BASE + 600))
        log.close(closed_trade("1"))
        assert [d.trade_id for d in log.closed()] == ["1"]


class TestFieldAccess:
    def test_a_dotted_path_reaches_nested_values(self):
        assert value_at(decision(), "fit.samples") == 500

    def test_factors_are_addressed_by_key(self):
        """Factors are stored as a list, so the natural dotted path has to be
        translated rather than indexed."""
        assert value_at(decision(), "factors.momentum.points") == 1.0

    def test_a_missing_path_is_none(self):
        assert value_at(decision(), "fit.nope.deeper") is None
        assert value_at(decision(), "nothing") is None


class TestAttribution:
    """The question the log exists to answer: does a parameter predict results?"""

    @staticmethod
    def _log(tmp_path, pairs):
        log = DecisionLog(tmp_path / "d.jsonl")
        for i, (samples, net) in enumerate(pairs):
            d = decision(str(i), ts=BASE + i * 600)
            d.fit = {"samples": samples, "r2": 0.01}
            log.record(d)
            log.close(closed_trade(str(i), net=net))
        return log

    def test_it_splits_into_equal_count_buckets(self, tmp_path):
        log = self._log(tmp_path, [(i * 10, 1.0) for i in range(30)])
        groups = group_by(log.closed(), "fit.samples", buckets=3)
        assert len(groups) == 3
        assert [len(g[1]) for g in groups] == [10, 10, 10]

    def test_buckets_are_ordered_by_the_parameter(self, tmp_path):
        log = self._log(tmp_path, [(i * 10, 1.0) for i in range(30)])
        groups = group_by(log.closed(), "fit.samples", buckets=3)
        firsts = [g[1][0].fit["samples"] for g in groups]
        assert firsts == sorted(firsts)

    def test_a_real_relationship_shows_up(self, tmp_path):
        """Low sample counts lose, high ones win — the summary must reflect it."""
        pairs = [(i, -5.0) for i in range(15)] + [(100 + i, 5.0) for i in range(15)]
        log = self._log(tmp_path, pairs)
        groups = group_by(log.closed(), "fit.samples", buckets=2)
        low = outcome_summary(groups[0][1])
        high = outcome_summary(groups[1][1])
        assert low["win_rate"] == 0.0 and high["win_rate"] == 1.0

    def test_too_few_trades_group_into_nothing(self, tmp_path):
        log = self._log(tmp_path, [(10, 1.0)])
        assert group_by(log.closed(), "fit.samples", buckets=3) == []

    def test_a_non_numeric_parameter_cannot_be_grouped(self, tmp_path):
        log = self._log(tmp_path, [(i, 1.0) for i in range(30)])
        assert group_by(log.closed(), "outlook.direction", buckets=3) == []

    def test_open_trades_are_excluded_from_attribution(self, tmp_path):
        log = self._log(tmp_path, [(i, 1.0) for i in range(30)])
        log.record(decision("open-one", ts=BASE + 99_999))
        groups = group_by(log.all(), "fit.samples", buckets=3)
        assert sum(len(g[1]) for g in groups) == 30


class TestSummary:
    def test_it_counts_wins_and_losses(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        for i, net in enumerate([5.0, -2.0, 3.0]):
            log.record(decision(str(i), ts=BASE + i * 600))
            log.close(closed_trade(str(i), net=net))
        s = outcome_summary(log.closed())
        assert (s["trades"], s["wins"]) == (3, 2)
        assert s["profit_factor"] == pytest.approx(4.0)
        assert s["net_points"] == pytest.approx(6.0)

    def test_an_empty_set_summarises_to_zero(self):
        assert outcome_summary([])["trades"] == 0

    def test_no_losses_leaves_the_profit_factor_undefined(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(decision("1"))
        log.close(closed_trade("1", net=5.0))
        assert outcome_summary(log.closed())["profit_factor"] is None
