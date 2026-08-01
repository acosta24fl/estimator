# MNQ Live Dashboard

A local web dashboard for Micro E-mini Nasdaq-100 futures (`MNQ=F`) using free
Yahoo Finance data.

It polls live prices, keeps an append-only log of every 1-minute bar, folds
those minutes into 5m / 10m / 15m / 30m / 1h / 4h / 1d candles, and charts them
with session VWAP, bar-size, MACD and daily higher-high / lower-low structure.

## Quick start

```bash
cd mnq
pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:8765>.

The first run downloads the charting library into `web/vendor/` (about 190 KB,
once). After that the dashboard needs no third-party host to render. If your
network blocks the download, fetch it yourself:

```bash
npm pack lightweight-charts@5.2.0     # then copy
# package/dist/lightweight-charts.standalone.production.js
# to mnq/web/vendor/
```

No network? Run the offline demo feed — same dashboard, synthetic prices:

```bash
MNQ_FEED=synthetic python run.py
```

## What it shows

| Feature | Where |
| --- | --- |
| Live price, change vs previous session close | header |
| Timeframe switcher — 1m, 5m, 10m, 15m, 30m, 1h, 4h, 1d | button row |
| Candles for the selected timeframe | main pane |
| **Session VWAP** + distance in points and % | orange line, metrics panel |
| **Bar size** — range per bar vs its rolling average | own pane, metrics panel |
| **MACD** (12, 26, 9) for the selected timeframe | own pane, metrics panel |
| **Daily higher highs / lower lows** — HH, LH, HL, LL | markers on 1d, level lines on every timeframe |
| **5-minute projection** — where the next 5m bar may close, with an uncertainty cone and a measured track record | ray + cone at the right edge, metrics panel |
| Feed health, bars logged, last bar written | footer |

Every metric is computed for **the timeframe currently on screen**, so the
numbers always describe the candles you are looking at. On the daily timeframe
the swing pivots are labelled directly on the chart as HH / LH / HL / LL.

## How the data flows

```
Yahoo Finance ──poll──▶ 1-minute bars ──▶ append-only log (data/bars/1m/*.jsonl)
                              │
                              └──fold──▶ 5m 10m 15m 30m 1h 4h 1d ──▶ indicators ──▶ chart
```

The 1-minute series is the only thing stored. Every other timeframe is derived
from it on demand, so there is one source of truth and no chance of the
timeframes disagreeing.

Bars are written the moment they close, one file per UTC day, and nothing is
ever rewritten — a corrected bar is appended and supersedes the older record.
The log reloads on startup, so intraday history keeps growing the longer you
run it (Yahoo itself only serves about 7 days of 1-minute data).

## Configuration

Every setting is an environment variable; no file edits required.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MNQ_SYMBOL` | `MNQ=F` | Yahoo ticker. Any symbol works — `ES=F`, `NQ=F`, `AAPL`. |
| `MNQ_FEED` | `yahoo` | `yahoo` or `synthetic`. |
| `MNQ_POLL_SECONDS` | `10` | Seconds between price polls. |
| `MNQ_PORT` / `MNQ_HOST` | `8765` / `127.0.0.1` | Where to serve. |
| `MNQ_INTRADAY_RANGE` | `5d` | History pulled per poll (Yahoo caps 1m at ~7d). |
| `MNQ_DAILY_RANGE` | `1y` | Daily history for market structure. |
| `MNQ_SESSION_TZ` | `America/New_York` | Session timezone. |
| `MNQ_SESSION_OPEN_HOUR` | `18` | Trade date rolls at 18:00 ET (CME Globex). |
| `MNQ_FORECAST_STRENGTH` | `1.0` | Scales the projected move. `0.5` damps it, `0` disables drift. |
| `MNQ_FORECAST_RIDGE_LAMBDA` | `10` | Ridge penalty. Larger shrinks coefficients toward no-move. |
| `MNQ_FORECAST_MIN_SAMPLES` | `200` | Fitted samples required before projecting at all. |
| `MNQ_DATA_DIR` | `mnq/data` | Where bar logs are written. |

## The 5-minute projection

The dashboard projects where the forming 5-minute bar may close, drawn as a
bright ray from the last price with a dashed cone for the likely range and the
target price labelled on the axis. A dotted purple line lays every past
projection over the candle it predicted, so the model's history sits next to
what actually happened.

The projection is a **small fitted linear model, not a trading signal.** It
uses three features you already have:

| Feature | What it measures |
| --- | --- |
| MACD histogram (5m) | momentum |
| Distance from session VWAP | stretch from the session's fair value |
| Daily HH/LL structure | directional bias |

Each is scaled against its *own* recent magnitude (so it stays comparable
across volatility regimes), then regressed on the realised next-bar move by
ridge:

    beta = (XtX + lambda*I)^-1 Xt y

The projection is `sum(beta_k * x_k)`. Coefficients are in **points per unit
feature**, so each term *is* that feature's contribution in points and they sum
exactly to the projection — the panel shows the breakdown.

**Why fitted rather than hand-weighted.** For a predictor with correlation
`rho` to the target, the MSE-optimal coefficient is `rho * sigma_y / sigma_x`;
anything larger provably increases error. An earlier version asserted weights
(0.5 / 0.3 / 0.2) and scaled by volatility, which implicitly assumes
`rho ~ 1`. Measured walk-forward on stored history:

| | direction | mean abs error | skill vs baseline |
| --- | --- | --- | --- |
| hand-weighted | 51.8% | 16.00 pts | **-40.5%** |
| ridge-fitted | 61.6% | 13.07 pts | **+8.1%** |

Fitting also fails safely: with no signal the coefficients shrink toward zero
and the projection degenerates to "no change", which is the baseline. Guessed
weights have no such guarantee. There is no intercept — over five minutes the
unconditional expected move is ~0, and a fitted constant would bias every
projection.

### It grades itself

Every projection is locked once, when its bar opens, using only data available
then — no hindsight — and appended to `data/predictions.jsonl`. Once the bar
closes it is scored. The headline is the **skill score**,
`1 - MSE(model) / MSE(no-move)`: positive means the projection helps, negative
means it is actively worse than assuming price stays put. If it cannot beat the
baseline the panel says *"Not adding value"* outright.

Rates carry **95% Wilson confidence intervals**, and the direction rate is only
coloured as an edge when its interval clears 50%. This matters more than it
sounds: over 100 predictions the interval around a coin flip is roughly +/-10
percentage points, so "56% correct" on a small sample is not evidence of
anything, and the panel now says so.

Measure it against your own history without waiting:

```bash
python -m app.backtest --dry-run              # measure only
python -m app.backtest --dry-run --strength 0.5
python -m app.backtest                        # also populates the chart track
```

On the synthetic feed the fitted model scores ~61% direction and about +9%
skill. **That number will not transfer** — synthetic prices have lag-1 return
autocorrelation of +0.28 where real 5-minute futures are near 0. Run the
backtest on your own MNQ history before trusting anything it draws.

If the projection overshoots, raise `MNQ_FORECAST_RIDGE_LAMBDA` (stronger
shrinkage toward no-move) or lower `MNQ_FORECAST_STRENGTH`.

## Adding to it later

This is built to grow by **adding files, never replacing them**. A new
indicator is one new file in `app/indicators/` — it then appears in the API,
gets its own chart pane and its own metrics automatically, with no changes to
the chart code, the API or any existing indicator.

See **[docs/EXTENDING.md](docs/EXTENDING.md)** for copy-paste recipes, and
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for how the pieces fit.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

216 tests cover bucketing (including the DST-shifted session), aggregation,
every indicator's maths, the append-only log, the Yahoo response parser
(offline, using recorded payload shapes), the library fetch, the forecast and
the ridge solver, the as-of structure series, prediction scoring and the
HTTP + WebSocket API. They need no network.

## Notes and limits

- Yahoo's free endpoint is **delayed** and unofficial; it is fine for charting
  and study, not for execution. There is no rate-limit guarantee — the 10s
  default poll is deliberately gentle.
- 4h and 1d candles are thin on a fresh install because they are folded from
  whatever 1-minute history exists. They fill in as the log grows. The 1d chart
  uses Yahoo's own daily history (1 year) so market structure works immediately.
- Daily bars are keyed by **session open** (18:00 ET the previous calendar day),
  matching the CME trade date.
- Charting uses [lightweight-charts](https://github.com/tradingview/lightweight-charts)
  (Apache-2.0), fetched on first run rather than committed.
