# Extending the dashboard

Everything here is **add a file**. No zip, no replacing an existing document,
no edit to the chart code. Drop a file in, restart, and it is live.

- [Add an indicator](#add-an-indicator) ← the common one
- [Add a timeframe](#add-a-timeframe)
- [Add a data source](#add-a-data-source)
- [Add a setting](#add-a-setting)
- [Checklist](#checklist)

---

## Add an indicator

Create one file in `app/indicators/`. It is discovered automatically, appears
in `/api/config`, gets a chart pane and its stats show up in the metrics panel.

### Example: RSI in its own pane

`app/indicators/rsi.py`

```python
"""Relative Strength Index for the selected timeframe."""

from . import register
from .base import (
    PANE_OWN, Indicator, IndicatorContext, IndicatorResult,
    RenderSpec, SeriesSpec, Stat,
)


@register
class Rsi(Indicator):
    key = "rsi"                     # API key; must be unique
    name = "RSI"                    # heading in the metrics panel
    order = 50                      # display order (lower = higher up)
    description = "Relative strength index."
    params = {"period": 14}

    render = RenderSpec(
        pane=PANE_OWN,              # PANE_PRICE to overlay on the candles
        height=120,
        series=(SeriesSpec("rsi", "RSI", type="line", color="#4c9aff"),),
    )

    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        period = int(self.params["period"])
        result = IndicatorResult(series={"rsi": []})

        closes = [b.close for b in ctx.bars]
        if len(closes) <= period:
            result.stats = [Stat("rsi_state", "RSI", "Warming up")]
            return result

        gains = losses = 0.0
        for i in range(1, period + 1):
            change = closes[i] - closes[i - 1]
            gains += max(change, 0.0)
            losses += max(-change, 0.0)
        avg_gain, avg_loss = gains / period, losses / period

        latest = 50.0
        for i in range(period + 1, len(closes)):
            change = closes[i] - closes[i - 1]
            avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
            latest = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
            result.series["rsi"].append({"time": ctx.bars[i].ts, "value": round(latest, 2)})

        tone = "up" if latest >= 70 else "down" if latest <= 30 else "neutral"
        result.stats = [Stat("rsi", f"RSI ({period})", round(latest, 2), tone=tone)]
        return result
```

Restart. RSI now has its own pane and a metrics card. Nothing else changed.

### What you get in `ctx`

| Field | What it is |
| --- | --- |
| `ctx.bars` | bars of the **selected** timeframe — usually what you want |
| `ctx.timeframe` | the `Timeframe` (`.key`, `.label`, `.nominal_seconds`) |
| `ctx.minute_bars` | the raw 1-minute series |
| `ctx.daily_bars` | daily series, for higher-timeframe context |
| `ctx.session` | session bucketer — `ctx.session.start(ts)` groups by trade date |
| `ctx.bars_5m` | the 5-minute series, whatever timeframe is displayed |
| `ctx.predictions` | projection log — `.scored(bars_5m)`, `.accuracy(bars_5m)` |
| `ctx.settings` | the app config |
| `ctx.now` | wall-clock epoch seconds |

`ctx.bars` is the **full** history so warm-up periods work. The engine trims
your output to the visible window afterwards — don't trim it yourself.

### What you return

```python
IndicatorResult(
    series={"name": [{"time": epoch_seconds, "value": 1.23}, ...]},
    stats=[Stat("key", "Label", value, unit="pts", tone="up")],
    markers=[Marker(time=ts, position="aboveBar", shape="arrowDown", text="HH")],
)
```

- **series keys must match `SeriesSpec.key`** — that is the wiring.
- `Stat.tone` is `neutral` / `up` / `down` / `warn` and colours the number.
- `Stat.signed=True` prefixes a `+` on positive values. Use it for quantities
  that can go negative (a distance), not for ones that cannot (a range).
- `Stat.hint` becomes the tooltip.
- `Marker.time` must equal a bar timestamp on the displayed timeframe.

### Drawing options

```python
# overlay a line on the candles
RenderSpec(pane=PANE_PRICE, series=(SeriesSpec("ma", "MA 50", color="#f2b53c"),))

# a histogram that colours by sign, in its own pane
RenderSpec(pane=PANE_OWN, zero_line=True, series=(
    SeriesSpec("delta", "Delta", type="histogram",
               up_color="#2e9e6b", down_color="#d1495b"),
))

# dashed reference level that must not stretch the price scale
SeriesSpec("level", "Prior Day High", line_style=2, autoscale=False)

# draw a labelled horizontal line at the series' last value
SeriesSpec("target", "Target", price_line=True)
```

`line_style`: 0 solid, 1 dotted, 2 dashed.

### Stats only, no chart

Give it an empty `series` tuple — the card still renders:

```python
render = RenderSpec(pane=PANE_PRICE, series=())
```

### Test it

Add `tests/test_rsi.py`; reuse the helpers in `tests/test_indicators.py`:

```python
from app.indicators.rsi import Rsi
from tests.test_indicators import ctx_for, make_bars

def test_rsi_is_high_after_a_rally():
    bars = make_bars([float(100 + i) for i in range(60)])
    stat = Rsi().compute(ctx_for(bars)).stats[0]
    assert stat.value > 70
```

There is already a registry test that computes every indicator against an
empty series, so a new one is checked for that automatically.

---

## Add a timeframe

Append one line at the bottom of `app/core/timeframes.py`:

```python
register(Timeframe("2h", "2 Hours", EpochBucket(2 * HOUR), 2 * HOUR, 300, 65))
#                   key    label     bucketing              seconds  limit order
```

The button, the aggregation and every indicator pick it up. `order` controls
where the button sits.

For an interval that is not fixed-width (a cash-session-only bar, say), write a
`Bucketer` — a class with `start(ts)` and `end(start)` — and pass that instead.

---

## Add a data source

One file in `app/feed/`, selected with `MNQ_FEED=<key>`:

```python
# app/feed/mybroker.py
from ..models import Bar, FeedSnapshot, Quote
from . import register
from .base import PriceFeed


@register
class MyBrokerFeed(PriceFeed):
    key = "mybroker"
    name = "My Broker"

    async def fetch_intraday(self) -> FeedSnapshot:
        bars = [Bar(ts=..., open=..., high=..., low=..., close=...,
                    volume=..., complete=...)]
        return FeedSnapshot(bars=bars, quote=Quote(price=..., ts=...))

    async def fetch_daily(self) -> FeedSnapshot:
        ...

    async def aclose(self) -> None:
        ...
```

Contract: bars ascending, `ts` = the bar's **start** in UTC epoch seconds,
1-minute spacing for `fetch_intraday`, and `complete=False` on the bar still
forming. Get that right and storage, aggregation and every indicator work
unchanged.

---

## Add a setting

Two lines in `app/config.py` — a field on `Settings` and a line in
`load_settings()`:

```python
alert_threshold: float = 25.0
# ...
alert_threshold=float(_env("ALERT_THRESHOLD", "25")),
```

It is now settable as `MNQ_ALERT_THRESHOLD` and readable anywhere as
`ctx.settings.alert_threshold`.

---

## Where things live

| I want to change… | File |
| --- | --- |
| a calculation | `app/indicators/<name>.py` |
| available intervals | `app/core/timeframes.py` |
| where prices come from | `app/feed/<name>.py` |
| how minutes fold up | `app/core/aggregator.py` |
| what is written to disk | `app/core/store.py` |
| the poll loop / payload | `app/core/engine.py` |
| API endpoints | `app/api/routes.py` |
| colours, layout | `web/styles.css` |
| generic chart behaviour | `web/app.js` |

If a change means editing `web/app.js` for one specific indicator, the render
spec is probably missing a field — add the field, keep the chart code generic.

---

## Checklist

- [ ] New file added; no existing file edited (except a one-line `register` for
      a timeframe).
- [ ] `key` is unique.
- [ ] `series` dict keys match the `SeriesSpec` keys.
- [ ] Short series handled — return a "Warming up" stat rather than raising.
- [ ] A test added under `tests/`.
- [ ] `python -m pytest` passes.
- [ ] Loaded the page and confirmed the pane and metrics card appear.
