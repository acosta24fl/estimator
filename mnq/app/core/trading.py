"""Turn scored projections into trades and measure them.

The dashboard forecasts; it does not trade. To talk about profit factor at all
you have to commit to a rule, so this module states the simplest one the
projection implies and measures it honestly:

* when the projected move is at least ``threshold`` points, take a position in
  that direction at the start of the 5-minute bar,
* close it at that bar's close — a fixed 5-minute hold, no stop, no target,
* charge costs on every trade.

Profit factor is ``gross profit / gross loss``. Two things about it are worth
knowing before reading any number this produces:

* It is undefined with no losing trades and unstable on small samples.
* For a symmetric bet — average win about equal to average loss, which a fixed
  5-minute hold with no stop or target essentially is — profit factor
  ``PF`` requires a win rate of ``PF / (1 + PF)``. **PF 2.0 therefore needs
  roughly a 66.7% win rate before costs.** Costs raise it further. That is the
  arithmetic any 5-minute strategy has to beat, and it is a very high bar.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

#: MNQ: $2 per index point (tick 0.25 pt = $0.50).
DOLLARS_PER_POINT = 2.0
#: Round-turn commission plus one tick of spread/slippage, in points.
#: ~$1.00 commission (0.5 pt) + 1 tick (0.25 pt).
DEFAULT_COST_POINTS = 0.75


@dataclass
class Trade:
    ts: int
    direction: int  # +1 long, -1 short
    entry: float
    exit: float
    gross: float  # points before costs
    net: float  # points after costs


@dataclass
class TradeStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_points: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    cost_points: float = 0.0

    @property
    def win_rate(self) -> float | None:
        decided = self.wins + self.losses
        return self.wins / decided if decided else None

    @property
    def profit_factor(self) -> float | None:
        """Gross profit / gross loss. None when there are no losses to divide by."""
        if self.gross_loss <= 0:
            return None
        return self.gross_profit / self.gross_loss

    @property
    def expectancy(self) -> float | None:
        return self.net_points / self.trades if self.trades else None

    @property
    def net_dollars(self) -> float:
        return self.net_points * DOLLARS_PER_POINT

    @property
    def required_win_rate_for(self) -> Any:
        """Win rate a symmetric bet needs to reach a given profit factor."""

        def _needed(pf: float) -> float:
            ratio = (self.avg_win / self.avg_loss) if self.avg_loss > 0 else 1.0
            # PF = (w * avg_win) / ((1 - w) * avg_loss)  =>  w = PF / (PF + ratio)
            return pf / (pf + ratio)

        return _needed

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "net_points": round(self.net_points, 2),
            "net_dollars": round(self.net_dollars, 2),
            "expectancy": self.expectancy,
            "max_drawdown": round(self.max_drawdown, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
        }


def simulate(
    scored: Sequence,
    cost_points: float = DEFAULT_COST_POINTS,
    threshold: float = 0.0,
) -> tuple[TradeStats, list[Trade]]:
    """Trade every projection whose size clears ``threshold``."""
    trades: list[Trade] = []
    for item in scored:
        prediction = item.prediction
        projected = prediction.target - prediction.anchor_price
        if abs(projected) < threshold or projected == 0:
            continue
        direction = 1 if projected > 0 else -1
        gross = direction * (item.actual - prediction.anchor_price)
        trades.append(
            Trade(
                ts=prediction.anchor_ts,
                direction=direction,
                entry=prediction.anchor_price,
                exit=item.actual,
                gross=gross,
                net=gross - cost_points,
            )
        )

    return summarise([t.net for t in trades], cost_points), trades


def summarise(nets: Sequence[float], cost_points: float = 0.0) -> TradeStats:
    """Roll a sequence of net results into the standard statistics.

    Shared by the replay backtest and the live paper trader so the two can
    never report the same trades differently.
    """
    stats = TradeStats(cost_points=cost_points, trades=len(nets))
    equity = 0.0
    peak = 0.0
    wins: list[float] = []
    losses: list[float] = []
    for net in nets:
        stats.net_points += net
        if net > 0:
            stats.wins += 1
            stats.gross_profit += net
            wins.append(net)
        elif net < 0:
            stats.losses += 1
            stats.gross_loss += -net
            losses.append(-net)
        else:
            stats.scratches += 1
        equity += net
        peak = max(peak, equity)
        stats.max_drawdown = max(stats.max_drawdown, peak - equity)

    stats.avg_win = (sum(wins) / len(wins)) if wins else 0.0
    stats.avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    return stats


def sharpe(trades: Sequence[Trade]) -> float | None:
    """Per-trade Sharpe ratio (mean / standard deviation of net points)."""
    if len(trades) < 2:
        return None
    nets = [t.net for t in trades]
    mean = sum(nets) / len(nets)
    var = sum((n - mean) ** 2 for n in nets) / (len(nets) - 1)
    sd = math.sqrt(var)
    return mean / sd if sd > 0 else None


def bootstrap_profit_factor(
    trades: Sequence[Trade], samples: int = 2000, seed: int = 12345
) -> tuple[float, float] | None:
    """95% bootstrap confidence interval for profit factor.

    Profit factor is a ratio of sums, so it is far noisier than it looks: on a
    few dozen trades a couple of large winners can move it by a whole point.
    Resampling the trades with replacement shows how wide that uncertainty
    really is, which is the difference between "PF 2.7" and "PF 2.7, and the
    data cannot rule out 0.9".
    """
    if len(trades) < 10:
        return None
    rng = random.Random(seed)
    nets = [t.net for t in trades]
    n = len(nets)
    factors: list[float] = []
    for _ in range(samples):
        profit = loss = 0.0
        for _ in range(n):
            value = nets[rng.randrange(n)]
            if value > 0:
                profit += value
            elif value < 0:
                loss -= value
        if loss > 0:
            factors.append(profit / loss)
    if len(factors) < samples // 2:
        return None
    factors.sort()
    return (
        factors[int(0.025 * len(factors))],
        factors[min(int(0.975 * len(factors)), len(factors) - 1)],
    )
