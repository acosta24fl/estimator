# Architecture

The guiding rule: **one source of truth, everything else derived.** The
1-minute series is stored; every timeframe, indicator and chart pane is
computed from it. That is what keeps the parts independent enough to extend
without touching each other.

## Layout

```
mnq/
├── run.py                     entry point
├── app/
│   ├── config.py              every setting, env-overridable
│   ├── models.py              Bar / Quote — the shared vocabulary
│   ├── main.py                assembly; the only file that wires parts together
│   ├── vendor.py              first-run fetch of the charting library
│   ├── feed/                  where data comes from      [plugin registry]
│   │   ├── base.py            PriceFeed interface
│   │   ├── yahoo.py           live Yahoo Finance
│   │   └── synthetic.py       deterministic offline feed
│   ├── core/
│   │   ├── timeframes.py      timeframe registry + bucketing strategies
│   │   ├── aggregator.py      1m ─▶ any timeframe (pure function)
│   │   ├── store.py           in-memory series + append-only minute log
│   │   ├── engine.py          poll loop + snapshot builder
│   │   └── plugins.py         shared auto-discovery helper
│   ├── indicators/            what gets computed         [plugin registry]
│   │   ├── base.py            Indicator interface + RenderSpec
│   │   ├── vwap.py  bar_size.py  macd.py  swings.py
│   └── api/routes.py          HTTP + WebSocket
├── web/                       chart UI (vanilla JS, no build step)
└── tests/
```

## The pipeline

```
        ┌────────────┐  1m bars   ┌──────────┐  close ┌──────────────────┐
Yahoo ─▶│  PriceFeed │───────────▶│ BarStore │───────▶│ data/bars/1m/*.jsonl │
        └────────────┘            └────┬─────┘        └──────────────────┘
                                       │ minute series
                                       ▼
                                 ┌────────────┐
                                 │ aggregator │  fold into 5m…1d
                                 └─────┬──────┘
                                       ▼
                                 ┌────────────┐
                                 │ indicators │  each returns series+stats+markers
                                 └─────┬──────┘
                                       ▼
                            snapshot ──▶ REST + WebSocket ──▶ browser
```

`Engine.snapshot()` is the single place a payload is built, so `/api/snapshot`
and the WebSocket can never disagree.

## Key decisions

### Aggregation is a pure function, not incremental state

`aggregate(minute_bars, timeframe)` recomputes from scratch every time. With a
few tens of thousands of minutes that is microseconds, and it removes the whole
class of bugs where an incrementally-updated 4h candle drifts from the minutes
underneath it. `AggregationCache` memoises on the series' last bar so repeated
requests are free, and any change to the forming bar invalidates it.

A higher-timeframe bar is marked `complete` only when wall-clock time has
passed its close **and** every 1-minute bar inside it is complete.

### Bucketing is pluggable

`timeframes.py` maps a timestamp to the bar that contains it via a `Bucketer`:

- `EpochBucket(seconds)` — fixed width. Every intraday interval here divides
  evenly into an hour or a day, so UTC alignment lands on the boundaries
  traders expect (:00, :05, :15, :30).
- `SessionBucket(tz, open_hour)` — one bucket per futures trade date. The CME
  day rolls at 18:00 America/New_York, so the daily bar spans 18:00 ET through
  17:00 ET the next day. Using `zoneinfo` keeps this right across daylight
  saving, where a session is 23 or 25 hours rather than 24.

Adding a calendar-aware interval later means writing a `Bucketer`, not
special-casing the aggregator.

### The minute log is append-only

`data/bars/1m/YYYY-MM-DD.jsonl`, one JSON object per line, written when a bar
closes. Nothing is ever rewritten in place:

- a partial write can only ever corrupt the final line, and the loader skips
  unparseable lines rather than failing;
- a revised bar is appended again and the loader takes the last record for a
  timestamp;
- the file is safe to `tail -f`, diff, or load into anything else.

The forming bar lives in memory only, so the log contains closed bars alone.

### The 1d timeframe uses real daily history

Folding 1d candles from 1-minute data would only reach as far back as the
minute log. Instead the daily series is Yahoo's own 1-year history, with any
session that *began* inside our minute coverage recomputed from those minutes
so today's bar updates live. Feed daily timestamps are re-keyed through
`SessionBucket` because feeds label them inconsistently (midnight, session open,
cash open); that puts them all on the same trade date the rest of the app uses.

### Indicators declare how they are drawn

An `Indicator` returns data *and* a `RenderSpec` describing its panes, series
types and colours. The browser builds the chart from that declaration, which is
why `web/app.js` contains no mention of VWAP, MACD or anything else specific.
Adding an indicator on the Python side is enough to make it appear.

`RenderSpec.pane` is `"price"` (overlay on the candles) or `"own"` (a new pane
underneath). `SeriesSpec.autoscale=False` keeps reference levels — like the
daily swing lines, which can sit far from price — from stretching the price
scale and squashing the candles.

### The charting library is vendored, but not committed

`web/` loads the chart library from `web/vendor/` rather than a CDN, so the
page renders with no third-party host involved at load time. The 190 KB
minified build is not kept in git; `app/vendor.py` downloads it on first run
(npm registry first, CDN as fallback), verifies it actually is the library
before saving, and does nothing on subsequent runs. `web/vendor/` is
gitignored, and the tests skip rather than fail if it has not been fetched.

### Failures are contained

A feed error is recorded in the status strip and retried on the next poll; the
chart keeps serving what it has. An indicator that raises is caught, reported
in its own metrics card, and the rest of the page still renders. A broken
plugin module is logged and skipped at import rather than taking down startup.

## Session and time conventions

- All timestamps are **UTC epoch seconds**, everywhere, and a bar's timestamp is
  its **start**.
- The browser is the only place times are localised.
- The futures trade date rolls at `MNQ_SESSION_OPEN_HOUR` (18:00) in
  `MNQ_SESSION_TZ` (America/New_York). Session VWAP resets there, and daily bars
  are keyed to it. A daily bar is therefore labelled with its session-open date
  — the evening before the CME trade date it belongs to.
