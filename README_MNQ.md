# MNQ Futures Signal System

A machine-learning signal engine for Micro E-mini Nasdaq-100 (MNQ) futures. It
builds multi-timeframe features, trains a per-direction model ensemble, prices
the result in a cost-aware backtest, and — once you are satisfied with the
numbers — sends entry/stop/target signals to Telegram every 10 minutes while
monitoring open positions minute by minute.

---

## Read this before anything else

**There is a real, small, directional edge. It is not yet shown to be
profitable.** Those are different claims and the distinction is the whole
story here.

The first real-data run — 5m bars over 60 days, price features only — returned
**AUC 0.4981 long / 0.5075 short**: a coin flip. Widening to 1h bars over 730
days and adding cross-asset context lifted that to roughly 0.56. A sweep then
found a configuration making \$4,140 at a 0.62 entry threshold, and it did
**not** survive validation: **p = 0.18** over 300 permutations, **97% of the
profit from a single quarter**, three of six quarters losing money.

That failure was arithmetic, not bad luck — ~180 features and a 270-cell sweep
against ~3,500 bars and 83 trades. The one encouraging detail was a *monotonic*
profit-factor curve across seven thresholds (0.96 → 1.40), the shape a small
real edge makes.

The decisive test came next. Trained on **ES, YM and RTY only — never MNQ** —
the model still predicted MNQ:

```
   long: AUC 0.5782 on 6,400 bars     short: AUC 0.5522 on 6,400 bars
   all 8 folds above chance (min 0.5176, max 0.6340)
   across folds: long p=0.011, short p=0.055
```

No shared bars, one pre-specified test, no tuning. A pattern that transfers
between instruments is a property of index futures, not memorised MNQ noise —
and that is much harder to fake than any time split.

So the constraint really was sample size. Training now
[pools all four instruments](#pooled-training) for ~55,000 rows instead of
13,700, and every backtest reports
[per-trade expectancy against its own error bar](#is-it-profitable).

**What remains unproven is profit.** AUC 0.578 is a small edge, and small
edges die to costs. Read the PROFITABILITY block, not the backtest total.

What has been verified is the machinery: 292 tests pass, including the lookahead
tests that decide whether any performance number can be believed at all.

Two synthetic controls bracket the pipeline's behaviour, and together they are
worth more than either alone:

- On a **random walk**, the models score AUC ≈ 0.50 and lose money after costs.
  Nothing is leaking future information — a pipeline that "worked" here would be
  broken.
- On synthetic data with **deliberately planted structure**, the same pipeline
  recovers AUC ≈ 0.57.

So the pipeline can find edge when edge exists. That makes the real-data 0.4981
more meaningful than it first looks: the problem is not that the machinery
cannot detect a signal, it is that there was no signal in that sample.

Two possible outcomes when you run it on real bars, and you should be prepared
for either:

- **AUC lands near 0.50 and the backtest is negative after costs.** Intraday
  index futures are close to efficient, and this is the common result. Do not
  trade it. Use `sweep` and `discover` to look for a subset of conditions that
  works, and accept that there may not be one.
- **AUC clears ~0.55 with a profitable backtest that holds in both halves of the
  sample.** Then paper-trade it live for several weeks before risking money.

The 25-250 point objective is enforced as a *filter* (`min_edge_points`), not a
promise. It rejects setups too small to be worth the risk; it cannot conjure
moves the market does not offer. 25 points is \$50 a contract against \$1.74 of
round-turn cost — 3.5% of the move, versus 8.7% at 10 points.

---

## Setup

**Windows:** double-click `MNQ.bat`. It installs what it needs on first run and
then shows a menu. There is no separate setup step.

**Everything else:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then fill in the three values
```

### Keep your data outside the code folder

Set `MNQ_HOME` and the cached price history, trained models and reports live
there instead of inside the checkout:

```bash
export MNQ_HOME="$HOME/mnq-data"        # Windows: MNQ.bat does this for you
```

This matters more than it sounds. Downloading two years of hourly bars for
thirteen instruments and training the ensemble takes real time, and without
`MNQ_HOME` every code update throws all of it away. With it set, the code
directory is disposable: replace it, rerun, and the expensive artifacts are
still there. `MNQ.bat` defaults it to `%USERPROFILE%\mnq-data`.

Secrets are read from the environment only, never from the YAML config, so
`config/default.yaml` stays safe to commit.

| Variable | Purpose |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/botfather). |
| `TELEGRAM_CHAT_ID` | From `https://api.telegram.org/bot<TOKEN>/getUpdates` — message your bot once first, or it cannot reply to you. |
| `TV_WEBHOOK_SECRET` | Any long random string. TradingView echoes it back on every alert. |

---

## Getting to a verdict

```bash
python -m mnq.cli fetch        # MNQ bars + the cross-asset basket
python -m mnq.cli experiment   # the decisive test: 4 configurations, one table
python -m mnq.cli train        # purged walk-forward evaluation, then final models
python -m mnq.cli backtest     # price those out-of-sample predictions
python -m mnq.cli sweep        # search entry gates and management rules
python -m mnq.cli discover     # mine the feature space for new patterns
```

**Start with `experiment`.** It runs the two-by-two that separates the two
variables that matter, so neither can be confused with the other:

|  | price only | + cross-asset |
| --- | --- | --- |
| **intraday** 5m, ~60 days | one regime, arbitraged inputs | one regime, better inputs |
| **wide** 1h, ~730 days | many regimes, arbitraged inputs | many regimes, better inputs |

### Why the wide profile exists

Yahoo caps intraday history at ~60 days. That is **one market regime** — and it
shows: a real run on 5m/60d produced base rates of 27.2% long against 35.8%
short, which is a falling market, not a property of the strategy. Nothing
validated on a single regime generalises, and worse, sweeping on it will happily
"discover" a short bias that is really just those two months.

The wide profile uses 1h bars over ~730 days. That is a similar *bar count* but
spans many regimes, which is what makes a walk-forward result mean something.
Hourly ATR also runs ~100 points, so 2×ATR targets land naturally inside the
25–250 point range.

### Why cross-asset context exists

EMAs, RSI and MACD on MNQ are the most widely computed numbers in markets. The
relationship between Nasdaq and bonds, the dollar, credit and volatility is not.
The basket (`mnq/data/context.py`) pulls from the same free Yahoo endpoint — no
new account, key, or rate limit:

| Group | Symbols | What it carries |
| --- | --- | --- |
| Index complex | ES, YM, RTY | rotation and relative strength |
| Rates | ZN | the core risk-off bid |
| Commodities / FX | GC, CL, 6E, 6J | inflation impulse, carry unwind |
| Risk appetite | HYG, VIX | credit stress, volatility regime |
| Leadership / breadth | SOXX, RSP | semis lead Nasdaq; equal-weight is a free breadth proxy |

From these come relative strength, rolling correlations (the *regime*, not the
level), a risk-appetite composite, VIX percentile, and a breadth divergence —
about 94 `ctx_` features. Individual symbols are allowed to fail; a partial
basket still works, and `fetch` prints exactly what loaded.

`train` prints the number that matters first:

```
  long: OOS AUC 0.5310 on 8,584 bars (base rate 31.3%) -> weak
 short: OOS AUC 0.4937 on 8,584 bars (base rate 31.6%) -> no edge
```

AUC below ~0.52 means the model has found nothing. No amount of threshold tuning
downstream will fix that, because there is no signal to tune. Stop there and
reconsider the features or the labels rather than sweeping until something looks
good.

Every command takes `--synthetic` to run without network access. That path
exercises the plumbing and proves nothing about the strategy.

### Accumulating more than 60 days

Yahoo serves only ~60 days of 5-minute bars per request. `fetch` merges each
download into a CSV cache, so running it weekly grows your usable history well
past that cap. More history is the single highest-leverage improvement available
here: the 4h EMA-200 alone needs ~33 days of 4h bars before it is even defined.

---

## Real history: `ingest`

Free Yahoo data caps this system at roughly **3,500 usable hourly bars**, which
produce **~83 trades** at a 0.62 threshold. Against ~180 features and a 270-cell
sweep, that sample cannot answer the question being asked of it — a permutation
test on a promising configuration returned p = 0.18, with one quarter carrying
97% of the profit. That is the arithmetic of the sample, not bad luck.

Two facts fix it:

**NQ is MNQ.** Both track the Nasdaq-100; NQ is \$20/point, MNQ is \$2/point.
MNQ only launched in 2019, but NQ has traded since 1999. For signal purposes NQ
history *is* MNQ history, and it is ~25 years deep instead of two.

**Yahoo's `NQ=F` is a naive splice.** It concatenates front-month contracts with
no adjustment, so each quarterly roll injects a price gap that is not a return.
In a representative run that gap is a **4.46σ** move — larger than almost
anything the market actually does — and it lands four times a year, corrupting
momentum features, inflating ATR (which sets your barriers), and letting
triple-barrier labels resolve on an event that never traded.

`ingest` solves both. It reads contract-level files, detects where volume
actually migrated between contracts, and back-adjusts the splice:

```bash
python -m mnq.cli ingest ~/nq-data --vendor firstrate --root NQ \
    --interval 1m --resample 1h
```

```
=== building continuous series (volume, ratio) ===
  25,890 bars  2022-01-01 .. 2024-12-14  11 rolls  median |gap| 20.0pt

  roll schedule (last 8):
    2023-12-08    NQZ23 -> NQH24   gap   +20.00pt  ratio 1.001360
    2024-03-08    NQH24 -> NQM24   gap   +20.00pt  ratio 1.001389
```

Measured at the roll boundaries, adjustment takes the shock from **4.46σ down
to 1.16σ** — the scale of an ordinary bar (median 0.67σ). The gap is removed;
genuine movement across the boundary is kept.

### Choosing the settings

| Flag | Default | When to change it |
| --- | --- | --- |
| `--method` | `volume` | `calendar` when your files have no volume; `open_interest` if you have OI and prefer it |
| `--adjust` | `ratio` | `difference` to keep point moves exact locally — but over a long history the accumulated offset can drive early prices to zero |
| `--confirm-sessions` | `2` | Raise it if a noisy day flips the roll back and forth |
| `--vendor` | `firstrate` | `databento` for raw exports (ns epochs, fixed-point prices), `generic` for plain UTC CSV |

`ratio` is the default because nearly every feature here is return- or
volatility-relative, and scaling prices by a constant cannot change a return.
The trade-off is that historical prices no longer match what printed.

### Where to buy it

| Vendor | What you get | Notes |
| --- | --- | --- |
| **Databento** | CME direct (GLBX.MDP3), contract-level, ~2010→ | Best quality; you control the roll |
| **FirstRate Data** | 1m continuous + per-contract, one-time fee | Cheapest credible option |
| **IQFeed** | Deep intraday history, subscription | Needs a running client |
| **CME DataMine** | Authoritative source | Expensive |

Add an unlisted vendor by *describing* it rather than writing a parser — see
`VendorSpec` in `mnq/data/vendor.py`.

### One thing to change after ingesting

`labels.min_target_points = 20` is not comparable across a long history. NQ
traded near 1,500 in 2003 and above 20,000 in 2025, so 20 points means 1.3%
then and 0.1% now. `ingest` prints a warning when it detects this. Express
targets in ATR or percentage terms before training on decades of data.

The archive lives outside the Yahoo cache (`artifacts/archive/`) because bought
data is not re-downloadable for free:

```
contracts/NQ/1m/NQZ24.parquet      per-contract raw bars (source of truth)
continuous/NQ_1h_ratio.parquet     the spliced, adjusted series
continuous/NQ_1h_ratio.rolls.csv   the roll schedule, openable in Excel
```

Contract files are kept so the continuous series can always be rebuilt with
different roll or adjustment settings.

---

## Pooled training

The cross-instrument result makes MNQ's own ~13,700 hourly bars an arbitrary
limit: if the pattern belongs to index futures generally, then ES, YM and RTY
bars are training data too.

```bash
python -m mnq.cli train --pooled          # MNQ + ES + YM + RTY, ~55,000 rows
```

Roughly **4x the sample at zero cost**, from symbols already downloaded.

The rule that makes it legitimate: **every fold trains only on bars preceding
its test window, for all four instruments including MNQ itself.** Index futures
are ~90% correlated, so training on ES during the hours being tested on MNQ
would leak the answer straight through that correlation — and it would read as
a spectacular result rather than a bug. Tests stub the models out and inspect
the exact rows reaching `fit`, including that MNQ's own later bars never appear.

The control is what makes the feature trustworthy: pooled training on four
synthetic **random walks** returns AUC 0.46 / 0.49 — *no edge*. Pooling does
not manufacture signal from noise.

---

## Is it profitable?

A positive backtest total does not answer that. Every `backtest` now prints a
PROFITABILITY block that asks three harder questions:

**Does the win rate clear its own breakeven?** A 35% win rate is excellent at
3:1 payoff and ruinous at 1:1. The report derives the breakeven win rate from
the *observed* payoff ratio and shows the margin.

**Is the edge bigger than its error bar?** Expectancy per trade with a 95%
confidence interval and a one-sided t-test. A positive mean whose interval
spans zero is reported **NOT ESTABLISHED** — that is the single most common way
a backtest misleads, and it is invisible in the total. A test pins the case:
59 small losers plus one large winner sums positive, means positive, and is
correctly rejected.

**What do costs take?** Round-turn cost as a share of the gross move.

Shape of the output (**illustrative figures, not a result**):

```
  Win rate            : NN.N%
  Breakeven win rate  : NN.N%  (implied by the observed payoff ratio)
  Margin              : +N.N%
  Expectancy          : $+N.NN per trade (+N.N points)
  95% interval        : [$+N.NN, $+NN.NN]
  VERDICT: PROFITABLE | NOT ESTABLISHED
```

Fewer than 30 trades is reported as *unmeasured*, not as a result.

This prices **one** configuration. It does not correct for how many were tried
to find it — that is `validate`'s job, and skipping it is what made the 0.62
result look real.

---

## Dashboard

```bash
python -m mnq.cli dashboard --open      # or MNQ.bat -> option 7
```

Serves a page on `http://localhost:8000` showing the candle chart with EMA
overlays, the model's direction and expected move in points, realised trend
over several windows, the ATR target and stop levels, live indicator values,
and engine state. It refreshes every 15 seconds.

Everything is served from the same process — no CDN, no external scripts, no
API keys — so it works offline and nothing about what you are watching leaves
the machine. The chart is drawn directly to a canvas; there is no charting
library to break.

The overlays are *read back from the feature matrix* rather than recomputed.
A dashboard that calculated its own EMAs would eventually disagree with the
model, and the disagreement would be silent.

### How "by how many points" is calculated

Not from the regressor. `SharedRegressor` minimises squared error on forward
percent return, and financial returns are mostly noise, so the loss-minimising
prediction is heavily shrunk toward zero — a genuinely informative model still
outputs tiny numbers, because predicting the true magnitude is punished on the
many occasions the move does not arrive. Reporting that as "expected points"
would understate every tradeable move, by construction.

Instead the projection is **empirically calibrated**. Out-of-sample
walk-forward predictions are bucketed by model confidence, and the realised
move after each bucket is measured:

| Side | Confidence | n | Hit rate | 25th | Median | 75th |
| --- | --- | --- | --- | --- | --- | --- |
| long | 0.55–0.60 | 118 | 69.5% | −6 | +15 | +34 |
| long | 0.60–0.65 | 123 | 53.7% | −21 | +4 | +26 |

A live score is matched to its bucket and reported with that bucket's median
*and its interquartile range*. When the 25th percentile is negative, the page
says so — "the middle half ran −21 to +26 points, so the range still spans a
losing outcome" — rather than drawing a confident arrow over an uncertain call.

Below the entry gate the projection reads FLAT and states why, so the direction
is never mistaken for a trade signal.

---

## Going live

```bash
python -m mnq.cli serve --seed-yahoo    # seeds 7 days of 1m bars, then listens
ngrok http 8000
```

`--seed-yahoo` matters. Without it the store needs roughly 17 hours of live
alerts before the slowest indicator is defined, and the system sits silent
meanwhile.

Then create a TradingView alert on MNQ, 1-minute, "Once Per Bar Close", pointed
at `https://<your-ngrok>.ngrok.io/webhook`, with this message body:

```json
{
  "secret": "YOUR_TV_WEBHOOK_SECRET",
  "time": "{{timenow}}",
  "open": {{open}}, "high": {{high}}, "low": {{low}},
  "close": {{close}}, "volume": {{volume}}
}
```

A close-only body (`{"secret": "...", "close": {{close}}}`) also works, and
`/webhook/raw` accepts plain text like `21050.25` for alerts that will not
produce valid JSON.

| Endpoint | Purpose |
| --- | --- |
| `POST /webhook` | Price ingestion. Requires the secret. |
| `POST /webhook/raw` | Same, for plain-text alert bodies. |
| `GET /status` | Bars stored, open trades, last evaluation. |
| `GET /health` | Liveness. |
| `POST /evaluate` | Force a signal evaluation now, ignoring cooldown. |
| `POST /flatten` | Close every open position immediately. |

**The secret is not optional in practice.** An ngrok URL is public; without it,
anyone who finds the URL can inject fake prices and drive your trade monitor.

### What actually happens while it runs

- **Every minute**, an arriving bar is stored and every open trade is re-checked
  against it — stops, targets, breakeven, trailing. Trade management runs on the
  minute cadence deliberately: a stop inspected only every 10 minutes is not a
  stop.
- **Every 10 minutes**, aligned to the wall clock, both direction models score
  the latest completed 5m bar and may fire a signal.

A signal arrives looking like this:

```
🟢 LONG  MNQ
2026-01-15 14:30 UTC

Entry   21,050.25
Stop    21,025.75  (24.5 pts / $49)
Target  21,099.25  (49.0 pts / $98)

R:R 2.00   Confidence 64.3%   ATR 24.5

Models: p_xgb=0.610  p_lgbm=0.660  p_meta=0.643  fwd_pred=+0.180
Monitoring for early exit.
```

---

## How it works

### Features (109 columns)

Computed per timeframe and prefixed `tf5_`, `tf15_`, `tf4h_`:

| Group | Features |
| --- | --- |
| Trend | EMA 8/9/21/50/200 — distance from each (in ATR units) and slope of each; EMA spreads 9-21, 8-21, 50-200 divided by ATR; Slope(20), Slope(50); Momentum(10); higher-highs and lower-lows counts over 5 bars, and their balance |
| Volatility | ATR(14); ATR acceleration (ATR / ATR 12 bars ago); Bollinger width (4σ / midline) |
| Oscillators | RSI(14); MACD(12,26,9) value/signal/histogram; ADX(14); +DI−DI; trend strength = \|slope20\| × ADX/100, signed and unsigned |
| Candle | Body ratio; wick ratio; signed wick skew; bar delta (close−open); relative volume vs SMA(20) |
| Session | Cyclical time-of-day, day of week, RTH flag |

Two decisions in here are load-bearing:

**Everything is scale-free.** Features are ATR-normalised or expressed in
percent, never raw price. A model fed raw EMA values memorises the price range of
its training window and collapses the moment the index trades outside it.

**Higher timeframes are merged on bar *close*, not bar open.** A 15m bar stamped
10:00 spans 10:00-10:15, so it first becomes visible to the 5m bar closing at
10:15 — never to the 5m bars at 10:00 or 10:05. Getting this backwards is the
most common way an intraday backtest ends up reading the future, and it is
covered by an explicit test.

### Labels

Triple-barrier, per direction: from the *next* bar's open, does price reach
`tp_atr_mult × ATR` before `sl_atr_mult × ATR`, within `horizon_bars`?

- Entry is the next bar's open because features are only known once the bar
  closes. Labelling from the close builds a small, consistent and entirely
  fictional edge into everything downstream.
- When one bar's range spans both barriers, **the stop is assumed to hit first**.
  5m OHLC does not record the intra-bar path, and assuming the win would inflate
  the hit rate precisely on the violent bars where it matters most.

### Ensemble, per direction

```
XGBoost classifier  ─┐
LightGBM classifier ─┼─→ Logistic meta ─→ P(win)
XGB regressor (fwd) ─┘
```

The forward-return regressor is shared between directions because forward return
is a directional quantity: a predicted +0.4% is bullish evidence that should
raise the long probability and lower the short one.

The meta learner is fitted on **out-of-fold** base predictions. This is the most
important detail in the model code. Trained on in-sample base output it would see
near-perfect inputs, conclude both boosters are infallible, and emit a wildly
overconfident live probability.

### Validation

Plain k-fold is invalid here, and so is a naive `TimeSeriesSplit`. Triple-barrier
labels overlap — the label at bar `t` depends on bars up to `t + horizon` — so if
bar `t` is in train and `t+3` in test, the training label already encodes the test
period. Training therefore uses expanding-window walk-forward with **purging and
an embargo** (`embargo_bars`, which must exceed `horizon_bars`).

The backtester consumes only these out-of-sample predictions. It never fits
anything.

### Trade management

One module, `mnq/trade/manager.py`, runs both the backtest and the live position.
Rules validated in a backtest are literally the same code that manages real
money — forking the logic is the usual way a strategy comes to behave nothing
like its backtest.

Exit precedence within a bar is fixed and pessimistic: stop, then target, then
discretionary. A bar that gaps through the stop fills at the open, not at the
stop price.

- **Breakeven** — after +1R, stop moves to entry + 1 tick.
- **Trailing** — after +1.5R, stop trails 1.5 ATR behind the extreme, ratcheting
  only toward profit.
- **Early exit** — this is the "is the trend continuing?" judgement. Each minute
  the model re-scores the open trade's direction; if continuation confidence
  falls below `early_exit_prob`, the trade closes at market. No score disables
  the rule rather than guessing.
- **Time stop** — after `max_hold_bars`, flatten if not yet profitable.

### Self-backtesting and pattern discovery

`sweep` re-runs the backtest across combinations of gates and management rules,
reusing one set of walk-forward predictions so hundreds of configurations cost
seconds. `sweep --barriers` searches TP/SL/horizon instead, retraining at every
point (slow).

`discover` ignores the models entirely and mines the feature matrix for
conditions where one direction beats its base rate — single features and pairs,
which is where non-obvious combinations show up.

**Both are multiple hypothesis testing on a finite sample.** Sweep 500
configurations and one will look excellent; mine thousands of buckets and a
striking one will always appear, even in pure noise. So:

- Sweep results are ranked on the **weaker of the two sample halves**, not on
  peak profit. A configuration with `is_net_usd` and `oos_net_usd` not both
  positive is a curve fit, and the CLI says so.
- Patterns are mined on the first 60% of history and re-measured on the rest.
  Only those keeping most of their edge are marked `holds=True`.

On synthetic noise, `discover` proposes ~1,900 candidate patterns with in-sample
lifts up to +8.7% and **the holdout rejects every one**. That rejection is the
feature.

---

## Configuration

`config/default.yaml` documents every tunable. The ones worth knowing:

| Setting | Default | Why it matters |
| --- | --- | --- |
| `labels.tp_atr_mult` / `sl_atr_mult` | 2.0 / 1.0 | Reward:risk. At 2:1 you need >33% wins just to break even. |
| `labels.horizon_bars` | 24 | Two hours to resolve. |
| `model.embargo_bars` | 36 | **Must exceed `horizon_bars`** or labels leak across folds. |
| `trade.min_probability` | 0.58 | Entry gate. Set from `sweep`, not intuition. |
| `trade.min_edge_points` | 20.0 | Enforces the lower end of the target range. |
| `trade.early_exit_prob` | 0.35 | Continuation floor for the monitor. |
| `trade.slippage_ticks` | 1.0 | **Do not set to zero.** A frictionless backtest will approve a losing strategy. |

---

## Layout

```
mnq/
  config.py            all tunables; secrets from env only
  indicators.py        EMA, ATR, RSI, MACD, ADX, Bollinger, slope, momentum
  labeling.py          triple-barrier labels + forward-return target
  cli.py               fetch / ingest / train / backtest / sweep / dashboard
  data/
    yahoo.py           Yahoo loader with a merging cache
    store.py           live 1m bar store with resampling
    synthetic.py       offline generator (pipeline validation only)
    contracts.py       futures symbology: codes, expiries, quarterly cycles
    vendor.py          purchased-data ingestion; add a vendor by describing it
    roll.py            roll detection + back-adjusted continuous series
    archive.py         Parquet store for contract and continuous history
  features/builder.py  multi-timeframe matrix, leak-free HTF merge
  models/
    ensemble.py        XGB + LGBM + shared regressor + LR meta
    train.py           walk-forward evaluation and persistence
    validation.py      purged/embargoed splitters
  backtest/
    engine.py          bar-by-bar simulation
    metrics.py         points and dollars
    sweep.py           parameter search + pattern discovery
  trade/
    manager.py         stops, targets, trailing, early exit (shared live/backtest)
    signals.py         entry/stop/target construction and gating
  notify/telegram.py   message formatting and delivery
  models/projection.py calibrated direction + expected move in points
  server/
    engine.py          live signal generation and monitoring
    app.py             FastAPI webhook + 10-minute scheduler + dashboard API
    dashboard.py       snapshot payloads for the local page
    static/dashboard.html  self-contained page, canvas chart, no CDN
tests/                 292 tests
```

## Tests

```bash
python -m pytest tests/ -q
```

The ones that matter most are in `tests/test_no_lookahead.py`. The strongest is
the truncation test: rebuild the features using only bars available at time T and
confirm every value matches the run where the whole future was present. If any
feature peeks ahead, the two disagree.

---

## Honest limitations

- **No real-data validation yet.** The headline caveat, repeated because it is
  the one that matters.
- **Yahoo intraday data is not exchange-grade.** Gaps, revisions and occasional
  bad prints. Fine for research, worth replacing with a real feed before trading
  size.
- **The backtest assumes your order fills.** One tick of slippage per side is
  modelled, but a resting stop in a fast market can do considerably worse.
- **A signal every 10 minutes is a ceiling, not a quota.** With `min_probability`
  at a sensible level most evaluations produce nothing, and that is correct
  behaviour.
- **Regimes change.** Retrain regularly and re-check that walk-forward AUC has
  not decayed.
- Nothing here is financial advice. Futures are leveraged and can lose more than
  the initial margin.
