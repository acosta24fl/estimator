"""Does the result survive scrutiny, or is it selection noise?

A sweep will always return a best row. This module exists to attack that row
rather than celebrate it. Four independent checks, each able to sink a result
the others miss:

1. **Permutation test** - the decisive one. Keep the trade management, the
   costs, the market, and the number of trades; destroy only the alignment
   between model confidence and outcome. If shuffled predictions make similar
   money, the profit came from the stop/target geometry, not the model.

2. **Threshold sensitivity** - a real edge strengthens gradually as the
   confidence bar rises. Profit at exactly one threshold, with losses either
   side, is a spike in noise.

3. **Time stability** - quarter by quarter, not two halves. A strategy that
   made everything in one quarter has not been demonstrated.

4. **Multiple-comparison accounting** - how many configurations were scored,
   and how many would pass the both-halves filter by chance alone. This is the
   context that makes "both halves positive" either meaningful or worthless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from .engine import run_backtest

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    threshold_curve: pd.DataFrame
    quarterly: pd.DataFrame
    permutation: dict[str, Any]
    multiple_comparisons: dict[str, Any]
    actual: dict[str, Any]

    def verdict(self) -> tuple[str, list[str]]:
        """Overall call plus the reasons behind it.

        Deliberately conservative: a result has to pass on several independent
        axes, because any single one can be cleared by luck.
        """
        concerns: list[str] = []
        passes = 0

        p = self.permutation.get("p_value")
        if p is not None:
            if p <= 0.05:
                passes += 1
            else:
                concerns.append(
                    f"permutation p={p:.3f}: shuffled predictions match this result "
                    f"{p:.0%} of the time, so the model may add nothing over the "
                    "trade management rules"
                )

        mono = self.threshold_curve
        profitable = mono[mono["net_usd"] > 0]
        if len(profitable) >= 2:
            passes += 1
        elif len(profitable) == 1:
            concerns.append(
                "only one threshold is profitable - a real edge usually improves "
                "gradually as the confidence bar rises, rather than working at a "
                "single value"
            )

        q = self.quarterly
        if len(q) >= 3:
            pos = (q["net_usd"] > 0).mean()
            if pos >= 0.6:
                passes += 1
            else:
                concerns.append(
                    f"only {pos:.0%} of quarters were profitable; the edge is not "
                    "stable across time"
                )
            top = q["net_usd"].max()
            total = q["net_usd"].sum()
            if total > 0 and top / total > 0.8:
                concerns.append(
                    f"one quarter produced {top/total:.0%} of all profit - remove it "
                    "and the strategy is flat"
                )

        n_trades = self.actual.get("n_trades", 0)
        if n_trades < 100:
            concerns.append(
                f"only {n_trades} trades; below ~100 the statistics are weak "
                "regardless of what they say"
            )

        mc = self.multiple_comparisons
        # Only meaningful when a sweep was actually supplied.
        if mc.get("n_configs", 0) > 0 and mc.get("expected_by_chance", 0) >= max(
            1, mc.get("n_passing", 0)
        ) * 0.5:
            concerns.append(
                f"{mc['n_configs']} configurations were scored and about "
                f"{mc['expected_by_chance']:.0f} would pass the both-halves filter "
                "by chance, so passing it is weak evidence on its own"
            )

        if passes >= 3 and not any("permutation" in c for c in concerns):
            return "PROMISING", concerns
        if passes >= 2:
            return "INCONCLUSIVE", concerns
        return "LIKELY NOISE", concerns


def threshold_sensitivity(
    matrix: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: Config,
    thresholds: tuple[float, ...] = (0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.70),
) -> pd.DataFrame:
    """Profit as a function of the entry threshold.

    The shape matters more than any single point. Rising profitability with
    rising confidence is what a genuine edge looks like.
    """
    rows = []
    for t in thresholds:
        trial = replace(cfg, trade=replace(cfg.trade, min_probability=t))
        try:
            res = run_backtest(matrix, predictions, trial)
        except Exception as exc:  # noqa: BLE001
            log.debug("threshold %.2f failed: %s", t, exc)
            continue
        m = res.metrics
        rows.append(
            {
                "min_probability": t,
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "net_usd": m["net_usd"],
                "profit_factor": m["profit_factor"] if m["n_trades"] else 0.0,
                "expectancy_r": m["expectancy_r"] if m["n_trades"] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def quarterly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Net P&L per calendar quarter.

    Two halves can hide a single dominant month; quarters make that visible.
    """
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["quarter"] = pd.PeriodIndex(pd.to_datetime(df["entry_time"], utc=True), freq="Q")
    out = df.groupby("quarter").agg(
        n_trades=("net_usd", "size"),
        net_usd=("net_usd", "sum"),
        win_rate=("net_usd", lambda s: float((s > 0).mean())),
        points=("points", "sum"),
    )
    return out.reset_index().astype({"quarter": str})


def permutation_test(
    matrix: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: Config,
    n_permutations: int = 200,
    seed: int = 17,
    method: str = "shift",
) -> dict[str, Any]:
    """Compare the real result against a null where predictions carry no signal.

    ``method="shift"`` rotates the prediction series circularly against the
    price series. This is the more conservative null: it preserves the
    predictions' own autocorrelation and their tendency to cluster in volatile
    periods, breaking only their alignment with what actually happened next.
    ``method="shuffle"`` permutes freely, which destroys that structure too and
    therefore tends to produce an optimistically small p-value.

    The p-value is the fraction of null runs matching or beating the real net
    P&L. A high p-value means the profit came from the trade management, the
    market's drift, or luck - not from the model.
    """
    actual = run_backtest(matrix, predictions, cfg)
    actual_net = actual.metrics["net_usd"]

    cols = [c for c in ("long_p_meta", "short_p_meta") if c in predictions]
    if not cols:
        raise ValueError("predictions must contain long_p_meta / short_p_meta")

    rng = np.random.default_rng(seed)
    n = len(predictions)
    null_nets: list[float] = []

    for i in range(n_permutations):
        shuffled = predictions.copy()
        if method == "shift":
            # A single shared offset keeps long and short consistent with each
            # other; shifting them independently would create signal pairings
            # the model never produces.
            offset = int(rng.integers(n // 20, n - n // 20))
            for c in cols:
                shuffled[c] = np.roll(predictions[c].to_numpy(), offset)
        else:
            perm = rng.permutation(n)
            for c in cols:
                shuffled[c] = predictions[c].to_numpy()[perm]

        try:
            res = run_backtest(matrix, shuffled, cfg)
            null_nets.append(res.metrics["net_usd"])
        except Exception:  # noqa: BLE001
            continue

        if (i + 1) % 50 == 0:
            log.info("  permutation %d/%d", i + 1, n_permutations)

    if not null_nets:
        return {"error": "all permutations failed"}

    null = np.array(null_nets)
    # One-sided: how often does noise do at least as well as the real thing?
    p_value = float((null >= actual_net).mean())

    return {
        "method": method,
        "n_permutations": len(null),
        "actual_net_usd": float(actual_net),
        "actual_n_trades": int(actual.metrics["n_trades"]),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "null_median": float(np.median(null)),
        "null_p95": float(np.percentile(null, 95)),
        "null_max": float(null.max()),
        "null_pct_profitable": float((null > 0).mean()),
        "p_value": p_value,
        "z_score": float(
            (actual_net - null.mean()) / null.std(ddof=1)
        ) if len(null) > 1 and null.std(ddof=1) > 0 else float("nan"),
    }


def multiple_comparison_context(sweep: pd.DataFrame) -> dict[str, Any]:
    """How much credit does passing the both-halves filter actually deserve?

    Under the null, each half is positive about half the time, so roughly a
    quarter of configurations pass by chance. Duplicated rows (parameters that
    made no difference) are collapsed first, since they are not independent
    tests and counting them would overstate the problem.
    """
    if sweep.empty:
        return {"n_configs": 0, "n_passing": 0, "expected_by_chance": 0.0}

    n_configs = len(sweep)
    effective = n_configs
    if {"n_trades", "net_usd"}.issubset(sweep.columns):
        effective = int(sweep[["n_trades", "net_usd"]].round(6).drop_duplicates().shape[0])

    passing = 0
    if {"is_net_usd", "oos_net_usd"}.issubset(sweep.columns):
        passing = int(((sweep["is_net_usd"] > 0) & (sweep["oos_net_usd"] > 0)).sum())

    return {
        "n_configs": n_configs,
        "effective_configs": effective,
        "n_passing": passing,
        # ~25% of independent configurations clear both halves under the null.
        "expected_by_chance": 0.25 * effective,
    }


def validate(
    matrix: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: Config,
    sweep: pd.DataFrame | None = None,
    n_permutations: int = 200,
) -> ValidationReport:
    """Run every check against the configuration in ``cfg``."""
    log.info("threshold sensitivity...")
    curve = threshold_sensitivity(matrix, predictions, cfg)

    log.info("backtesting the chosen configuration...")
    result = run_backtest(matrix, predictions, cfg)
    quarterly = quarterly_breakdown(result.frame)

    log.info("permutation test (%d runs)...", n_permutations)
    perm = permutation_test(matrix, predictions, cfg, n_permutations)

    mc = multiple_comparison_context(sweep if sweep is not None else pd.DataFrame())

    return ValidationReport(
        threshold_curve=curve,
        quarterly=quarterly,
        permutation=perm,
        multiple_comparisons=mc,
        actual=result.metrics,
    )


def format_report(report: ValidationReport, cfg: Config) -> str:
    """Render the report for the console."""
    lines: list[str] = []
    w = 78
    lines.append("=" * w)
    lines.append(f"  VALIDATION  (min_probability = {cfg.trade.min_probability:.2f})")
    lines.append("=" * w)

    a = report.actual
    lines.append(
        f"\nChosen configuration: {a.get('n_trades', 0)} trades, "
        f"net ${a.get('net_usd', 0):,.0f}, PF {a.get('profit_factor', 0):.2f}, "
        f"win rate {a.get('win_rate', 0):.1%}"
    )

    lines.append("\n" + "-" * w)
    lines.append("1. THRESHOLD SENSITIVITY  (a real edge strengthens gradually)")
    lines.append("-" * w)
    if report.threshold_curve.empty:
        lines.append("  no results")
    else:
        lines.append(report.threshold_curve.to_string(index=False))

    lines.append("\n" + "-" * w)
    lines.append("2. STABILITY BY QUARTER  (was it one lucky period?)")
    lines.append("-" * w)
    if report.quarterly.empty:
        lines.append("  no trades")
    else:
        lines.append(report.quarterly.to_string(index=False))

    lines.append("\n" + "-" * w)
    lines.append("3. PERMUTATION TEST  (does the model beat shuffled predictions?)")
    lines.append("-" * w)
    p = report.permutation
    if "error" in p:
        lines.append(f"  {p['error']}")
    else:
        lines.append(
            f"  Real net P&L       : ${p['actual_net_usd']:>10,.0f}  "
            f"({p['actual_n_trades']} trades)"
        )
        lines.append(
            f"  Shuffled average   : ${p['null_mean']:>10,.0f}  "
            f"(std ${p['null_std']:,.0f}, {p['n_permutations']} runs, "
            f"method={p['method']})"
        )
        lines.append(f"  Shuffled 95th pct  : ${p['null_p95']:>10,.0f}")
        lines.append(f"  Shuffled best      : ${p['null_max']:>10,.0f}")
        lines.append(
            f"  Shuffled profitable: {p['null_pct_profitable']:>10.0%}  "
            "<- how often random timing still makes money"
        )
        lines.append(f"\n  p-value            : {p['p_value']:.4f}")
        if p["p_value"] <= 0.01:
            lines.append("  -> Strong: noise almost never does this well.")
        elif p["p_value"] <= 0.05:
            lines.append("  -> Reasonable: unlikely, though not conclusive at this sample size.")
        elif p["p_value"] <= 0.20:
            lines.append("  -> Weak: noise reproduces this result fairly often.")
        else:
            lines.append(
                "  -> Fails. Shuffled predictions match this result routinely, so "
                "the profit is coming from the trade management and the market, "
                "not from the model."
            )

    lines.append("\n" + "-" * w)
    lines.append("4. MULTIPLE-COMPARISON CONTEXT")
    lines.append("-" * w)
    mc = report.multiple_comparisons
    if mc["n_configs"]:
        lines.append(f"  Configurations scored     : {mc['n_configs']}")
        lines.append(f"  Distinct results          : {mc['effective_configs']}")
        lines.append(f"  Passed both-halves filter : {mc['n_passing']}")
        lines.append(f"  Expected to pass by chance: {mc['expected_by_chance']:.0f}")
        if mc["n_passing"] <= mc["expected_by_chance"]:
            lines.append(
                "  -> No more configurations passed than luck predicts. Passing that "
                "filter means little here."
            )
    else:
        lines.append("  (no sweep results supplied)")

    verdict, concerns = report.verdict()
    lines.append("\n" + "=" * w)
    lines.append(f"  VERDICT: {verdict}")
    lines.append("=" * w)
    if concerns:
        lines.append("\n  Concerns:")
        for c in concerns:
            lines.append(f"   - {c}")
    if verdict == "PROMISING":
        lines.append(
            "\n  Survived the checks. That justifies forward paper-trading, not\n"
            "  capital. Run it live without money for several weeks and confirm\n"
            "  the live win rate matches the backtest before risking anything."
        )
    elif verdict == "INCONCLUSIVE":
        lines.append(
            "\n  Not established either way. More history is the only honest fix -\n"
            "  keep running `fetch` weekly and re-check. Do not trade it yet."
        )
    else:
        lines.append(
            "\n  The result does not hold up. Do not trade it, and do not keep\n"
            "  sweeping for a configuration that passes - that search is what\n"
            "  produced this result in the first place."
        )
    return "\n".join(lines)
