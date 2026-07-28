"""Multi-horizon forecasting with per-horizon skill measurement.

Answers "where is price likely to be in 5 / 10 / 15 / 30 minutes, 1 / 4 / 24
hours" - and, for each of those, whether that answer is worth anything.

The second half is the point. Short-horizon index futures moves are close to a
random walk: five minutes ahead is *harder* to predict than an hour, not
easier, because there is less signal and the same noise. A panel that printed a
confident destination at every horizon would be fabricating six of them.

So every horizon carries a **skill score**: how much the conditional forecast
beats the naive random-walk forecast ("price stays here"), measured on history.

    skill = 1 - MSE(conditional) / MSE(no-change)

Zero means no better than assuming price does not move. Negative means worse.
Only positive skill justifies looking at the number, and the dashboard fades
out the horizons that do not have it.

**How the forecast is made.** Not from a regressor - those shrink to zero on
noisy targets. Price history is bucketed by *state* (trend direction and
strength, crossed with volatility regime), and each bucket's realised forward
moves become an empirical distribution. The forecast for a horizon is that
bucket's median, with its 25-75 and 10-90 bands. Everything is a quantile of
something that actually happened in a comparable state.

**Continuation vs reversal** falls out of the same table: given the current
trend state, the share of analogues that continued in the trend's direction,
compared against the unconditional base rate so a 55% "continuation" in a
market that rises 55% of the time is correctly reported as no information.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: The horizons the dashboard shows, in minutes.
HORIZONS: tuple[int, ...] = (5, 10, 15, 30, 60, 240, 1440)

HORIZON_LABELS = {
    5: "5m", 10: "10m", 15: "15m", 30: "30m",
    60: "1h", 240: "4h", 1440: "24h",
}

#: Bars used to measure the prevailing trend before classifying it.
TREND_LOOKBACK = 30

#: Below this many analogues a bucket is not reported.
MIN_ANALOGUES = 60

#: Skill at or below this is treated as "none" - the honest default. Not zero:
#: an out-of-sample skill estimate has its own noise, and a hair above zero is
#: indistinguishable from luck.
SKILL_FLOOR = 0.01

#: Overlapping forward windows inflate the apparent sample. At a 1440-minute
#: horizon, 18,000 test bars hold only ~12 *independent* 24-hour windows, and a
#: skill estimate from twelve observations swings between +0.14 and -0.32 on
#: pure noise. Effective n = test bars / horizon must clear this.
MIN_EFFECTIVE_WINDOWS = 30

TREND_STATES = ("strong_down", "down", "range", "up", "strong_up")
VOL_STATES = ("low", "normal", "high")


@dataclass(frozen=True)
class HorizonForecast:
    """Where price is likely to be at one horizon, and whether to believe it."""

    minutes: int
    label: str
    from_price: float
    median_price: float
    median_points: float
    p25_points: float
    p75_points: float
    p10_points: float
    p90_points: float
    p_up: float
    base_rate_up: float          # unconditional, for comparison
    edge_vs_base: float          # p_up minus base rate
    skill: float                 # vs the no-change forecast, out of sample
    has_skill: bool
    n_analogues: int
    effective_windows: int = 0   # independent forward windows behind the skill
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrendRead:
    """The current trend state and what usually follows it."""

    state: str = "range"
    strength: float = 0.0            # normalised slope, ATR units per bar
    vol_state: str = "normal"
    atr_percentile: float = 0.5
    continuation_prob: float | None = None
    continuation_edge: float | None = None   # vs the unconditional base rate
    n_analogues: int = 0
    verdict: str = "unknown"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HorizonModel:
    """Empirical forward-move distributions, bucketed by state."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    base_rates: dict[int, float] = field(default_factory=dict)
    built_from: int = 0
    built_at: pd.Timestamp | None = None

    @property
    def available(self) -> bool:
        return not self.table.empty


# --------------------------------------------------------------- state


def classify(minutes: pd.DataFrame, lookback: int = TREND_LOOKBACK) -> pd.DataFrame:
    """Label every bar with a trend state and a volatility regime.

    Both are computed from price alone, so the same labelling applies to
    history and to the live bar - which is what makes the historical
    distribution a fair analogue for now.
    """
    close = minutes["close"]
    high, low = minutes["high"], minutes["low"]

    # True range, smoothed. Used both as the volatility regime and as the unit
    # that makes slope comparable across regimes.
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    safe_atr = atr.replace(0.0, np.nan)

    # Slope per bar in ATR units: "half an ATR per bar" means the same thing in
    # a quiet market and a violent one.
    slope = (close - close.shift(lookback)) / lookback / safe_atr

    out = pd.DataFrame(index=minutes.index)
    out["slope"] = slope
    out["atr"] = atr
    out["atr_pct"] = atr.rolling(1440, min_periods=120).rank(pct=True)

    # Thresholds in ATR-per-bar units. 0.05 is a drift; 0.15 is a real trend.
    out["trend"] = pd.cut(
        slope,
        bins=[-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf],
        labels=list(TREND_STATES),
    ).astype(object)

    out["vol"] = pd.cut(
        out["atr_pct"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=list(VOL_STATES),
    ).astype(object)

    return out


# ---------------------------------------------------------------- build


def build_horizon_model(
    minutes: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    lookback: int = TREND_LOOKBACK,
    train_frac: float = 0.7,
) -> HorizonModel:
    """Measure what price did after each (trend, volatility) state.

    ``minutes`` is the 1-minute bar history. Working from 1m rather than the
    feature matrix is what lets a 5-minute horizon exist at all when the model
    itself runs on hourly bars.

    **Skill is measured out of sample and this is not optional.** Bucket
    medians are fitted on the first ``train_frac`` of history and scored on the
    rest. Measured in-sample, a pure random walk scores skill up to +0.12 and an
    alpha grade of A, because a conditional median always fits the data it was
    computed from. The split is the only thing standing between this panel and
    confident nonsense.
    """
    if minutes is None or len(minutes) < max(horizons) + lookback:
        return HorizonModel()

    state = classify(minutes, lookback)
    close = minutes["close"]
    split = int(len(minutes) * train_frac)
    if split < MIN_ANALOGUES * 2:
        return HorizonModel()

    rows: list[dict[str, Any]] = []
    base_rates: dict[int, float] = {}
    skills: dict[int, float] = {}
    effective: dict[int, int] = {}

    for h in horizons:
        move = close.shift(-h) - close
        valid = move.notna() & state["trend"].notna() & state["vol"].notna()
        if valid.sum() < MIN_ANALOGUES:
            continue

        base_rates[h] = float((move[valid] > 0).mean())

        frame = pd.DataFrame(
            {"trend": state["trend"], "vol": state["vol"], "move": move}
        )[valid]
        # The label at bar t peeks h bars ahead, so the training block must end
        # h bars before the test block begins or the two overlap.
        train = frame.iloc[: max(0, split - h)]
        test = frame.iloc[split:]
        if len(train) < MIN_ANALOGUES or len(test) < MIN_ANALOGUES:
            continue

        fitted = (
            train.groupby(["trend", "vol"], observed=True)["move"]
            .median()
            .to_dict()
        )

        # Score the fitted medians on data they never saw.
        keys = list(zip(test["trend"], test["vol"]))
        predicted = np.array([fitted.get(k, 0.0) for k in keys], dtype=float)
        actual = test["move"].to_numpy(dtype=float)
        naive_mse = float((actual ** 2).mean())
        cond_mse = float(((actual - predicted) ** 2).mean())
        skills[h] = 1.0 - cond_mse / naive_mse if naive_mse > 0 else 0.0
        effective[h] = len(test) // h

        # Quantiles come from the full history: more data makes a better
        # distribution, and unlike the skill score they are descriptive rather
        # than a claim about predictive power.
        for (trend, vol), grp in frame.groupby(["trend", "vol"], observed=True):
            if len(grp) < MIN_ANALOGUES:
                continue
            m = grp["move"]
            rows.append(
                {
                    "horizon": h, "trend": trend, "vol": vol,
                    "n": int(len(m)),
                    "median": float(m.median()),
                    "p25": float(m.quantile(0.25)), "p75": float(m.quantile(0.75)),
                    "p10": float(m.quantile(0.10)), "p90": float(m.quantile(0.90)),
                    "p_up": float((m > 0).mean()),
                    "skill": skills[h],
                    "effective_n": effective[h],
                }
            )

    if not rows:
        return HorizonModel()

    table = pd.DataFrame(rows)
    table["skill"] = table.get("skill", 0.0)
    return HorizonModel(
        table=table.set_index(["horizon", "trend", "vol"]).sort_index(),
        base_rates=base_rates,
        built_from=int(len(minutes)),
        built_at=pd.Timestamp.now("UTC"),
    )


# -------------------------------------------------------------- forecast


def forecast(
    minutes: pd.DataFrame,
    model: HorizonModel | None,
    horizons: tuple[int, ...] = HORIZONS,
    lookback: int = TREND_LOOKBACK,
) -> tuple[list[HorizonForecast], TrendRead]:
    """Forecast every horizon from the current state, plus the trend read."""
    read = TrendRead()
    if minutes is None or minutes.empty or model is None or not model.available:
        read.note = "not enough history yet to measure what follows this state"
        return [], read

    state = classify(minutes, lookback)
    latest = state.iloc[-1]
    trend, vol = latest.get("trend"), latest.get("vol")
    price = float(minutes["close"].iloc[-1])

    read.state = str(trend) if isinstance(trend, str) else "range"
    read.vol_state = str(vol) if isinstance(vol, str) else "normal"
    read.strength = float(latest["slope"]) if pd.notna(latest["slope"]) else 0.0
    read.atr_percentile = (
        float(latest["atr_pct"]) if pd.notna(latest["atr_pct"]) else 0.5
    )

    out: list[HorizonForecast] = []
    for h in horizons:
        try:
            row = model.table.loc[(h, read.state, read.vol_state)]
        except KeyError:
            continue
        base = model.base_rates.get(h, 0.5)
        skill = float(row.get("skill", 0.0))
        eff = int(row.get("effective_n", 0))
        has_skill = skill > SKILL_FLOOR and eff >= MIN_EFFECTIVE_WINDOWS
        median = float(row["median"])

        out.append(
            HorizonForecast(
                minutes=h,
                label=HORIZON_LABELS.get(h, f"{h}m"),
                from_price=price,
                median_price=price + median,
                median_points=median,
                p25_points=float(row["p25"]), p75_points=float(row["p75"]),
                p10_points=float(row["p10"]), p90_points=float(row["p90"]),
                p_up=float(row["p_up"]),
                base_rate_up=base,
                edge_vs_base=float(row["p_up"]) - base,
                skill=skill,
                has_skill=has_skill,
                n_analogues=int(row["n"]),
                effective_windows=eff,
                note=_horizon_note(row, skill, has_skill, base, eff),
            )
        )

    _read_trend(read, out, model)
    return out, read


def _horizon_note(row, skill: float, has_skill: bool, base: float, eff: int) -> str:
    if eff < MIN_EFFECTIVE_WINDOWS:
        return (
            f"only {eff} independent {int(row.name[0]) if hasattr(row, 'name') else 0}-minute "
            f"windows in the test period - too few to measure skill at all. "
            f"Collect more history."
        ).replace("0-minute ", "")
    if not has_skill:
        return (
            f"no measurable skill here (skill {skill:+.3f}); price at this "
            f"horizon is not distinguishable from a random walk"
        )
    edge = float(row["p_up"]) - base
    return (
        f"{int(row['n'])} analogues, {float(row['p_up']):.0%} closed higher "
        f"vs {base:.0%} unconditionally ({edge:+.1%})"
    )


def _read_trend(read: TrendRead, forecasts: list[HorizonForecast], model) -> None:
    """Continuation or reversal, judged against the unconditional base rate."""
    if not forecasts:
        read.verdict = "unknown"
        read.note = "no horizon had enough comparable history"
        return

    if read.state == "range":
        read.verdict = "no trend"
        read.note = (
            f"slope {read.strength:+.3f} ATR/bar is inside the range band; "
            "continuation and reversal are not meaningful here"
        )
        return

    # Judge continuation on the shortest horizon that actually has skill;
    # falling back to the shortest available if none does.
    skilled = [f for f in forecasts if f.has_skill]
    anchor = (skilled or forecasts)[0]
    up_trend = read.state in ("up", "strong_up")

    cont = anchor.p_up if up_trend else 1.0 - anchor.p_up
    base = anchor.base_rate_up if up_trend else 1.0 - anchor.base_rate_up
    read.continuation_prob = float(cont)
    read.continuation_edge = float(cont - base)
    read.n_analogues = anchor.n_analogues

    if not skilled:
        read.verdict = "no signal"
        read.note = (
            f"trend is {read.state.replace('_', ' ')}, but no horizon shows "
            f"skill above a random walk - the direction is descriptive only"
        )
    elif read.continuation_edge > 0.03:
        read.verdict = "continuation"
        read.note = (
            f"after a {read.state.replace('_', ' ')} state in {read.vol_state} "
            f"volatility, price continued {cont:.0%} of the time over "
            f"{anchor.label} vs {base:.0%} unconditionally"
        )
    elif read.continuation_edge < -0.03:
        read.verdict = "reversal"
        read.note = (
            f"after a {read.state.replace('_', ' ')} state in {read.vol_state} "
            f"volatility, price reversed more often than usual - continuation "
            f"only {cont:.0%} vs {base:.0%} unconditionally"
        )
    else:
        read.verdict = "neutral"
        read.note = (
            f"continuation {cont:.0%} vs {base:.0%} unconditionally - a "
            f"{read.continuation_edge:+.1%} edge, which is noise"
        )


# ------------------------------------------------------------ alpha score


def alpha_score(
    forecasts: list[HorizonForecast],
    read: TrendRead,
    cost_points: float,
) -> dict[str, Any]:
    """A 0-100 conviction score, with every component shown.

    Deliberately anchored so it cannot flatter: with no measurable skill at any
    horizon the score is zero, however strong the trend looks. A composite that
    could read 60 on pure noise would be worse than no composite at all.
    """
    components: dict[str, float] = {}

    skilled = [f for f in forecasts if f.has_skill]
    if not skilled:
        return {
            "score": 0.0,
            "grade": "F",
            "components": {"skill": 0.0, "conviction": 0.0, "payoff": 0.0,
                           "agreement": 0.0},
            "tradeable": False,
            "note": (
                "no horizon beats a random walk right now, so there is nothing "
                "to score. This is the normal state most of the time."
            ),
        }

    best = max(skilled, key=lambda f: f.skill)

    # 1. How much better than a random walk, at the best horizon.
    components["skill"] = float(min(1.0, max(0.0, best.skill / 0.05)))

    # 2. How far the directional call is from a coin flip.
    components["conviction"] = float(min(1.0, abs(best.edge_vs_base) / 0.10))

    # 3. Is the expected move worth the round-turn cost?
    payoff = abs(best.median_points) / max(cost_points, 1e-9)
    components["payoff"] = float(min(1.0, payoff / 10.0))

    # 4. Do the skilled horizons point the same way? Disagreement is a warning.
    ups = sum(1 for f in skilled if f.median_points > 0)
    agreement = max(ups, len(skilled) - ups) / len(skilled)
    components["agreement"] = float((agreement - 0.5) * 2.0)

    weights = {"skill": 0.35, "conviction": 0.30, "payoff": 0.20, "agreement": 0.15}
    score = 100.0 * sum(components[k] * w for k, w in weights.items())

    grade = (
        "A" if score >= 75 else "B" if score >= 60 else
        "C" if score >= 45 else "D" if score >= 25 else "F"
    )
    return {
        "score": round(float(score), 1),
        "grade": grade,
        "components": {k: round(v, 3) for k, v in components.items()},
        "best_horizon": best.label,
        "tradeable": score >= 60,
        "note": (
            f"best horizon {best.label}: skill {best.skill:+.3f}, "
            f"{best.p_up:.0%} up vs {best.base_rate_up:.0%} base, "
            f"median {best.median_points:+.0f} pts against {cost_points:.1f} pts of cost"
        ),
    }


def path_payload(forecasts: list[HorizonForecast]) -> list[dict[str, Any]]:
    """The projected path, for the chart's forward cone."""
    return [
        {
            "minutes": f.minutes, "label": f.label,
            "median": f.median_price,
            "p25": f.from_price + f.p25_points, "p75": f.from_price + f.p75_points,
            "p10": f.from_price + f.p10_points, "p90": f.from_price + f.p90_points,
            "has_skill": f.has_skill, "skill": f.skill,
        }
        for f in forecasts
    ]
