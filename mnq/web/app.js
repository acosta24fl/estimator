/*
 * MNQ dashboard front end.
 *
 * The important idea here: this file knows nothing about VWAP, MACD, bar size
 * or market structure.  It reads /api/config, builds a pane + series for every
 * indicator the server declares, then blindly maps snapshot data onto them.
 * Adding a Python indicator makes it appear here with no changes to this file.
 */

const LWC = window.LightweightCharts;
const {
  createChart,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  createSeriesMarkers,
} = LWC;

const SERIES_TYPES = { line: LineSeries, histogram: HistogramSeries };

const state = {
  config: null,
  timeframe: null,
  chart: null,
  candles: null,
  markers: null,
  /** "indicatorKey.seriesKey" -> { api, spec } */
  series: new Map(),
  socket: null,
  reconnectDelay: 1000,
  lastSnapshotAt: 0,
};

const el = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ chart */

function chartOptions() {
  const css = getComputedStyle(document.documentElement);
  const grid = css.getPropertyValue("--border").trim();
  const text = css.getPropertyValue("--muted").trim();
  return {
    layout: {
      background: { color: "transparent" },
      textColor: text,
      attributionLogo: false,
      panes: { separatorColor: grid, separatorHoverColor: grid },
    },
    grid: {
      vertLines: { color: grid, style: 1 },
      horzLines: { color: grid, style: 1 },
    },
    rightPriceScale: { borderColor: grid },
    timeScale: {
      borderColor: grid,
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 4,
      // Axis labels in the viewer's local time rather than UTC.  The chart
      // tells us what granularity each tick represents, which is what makes
      // daily bars show dates and intraday bars show clock times.
      tickMarkFormatter: formatTickMark,
    },
    crosshair: { mode: 0 },
    localization: {
      timeFormatter: (time) => formatTime(time, true),
    },
    autoSize: true,
  };
}

function formatTickMark(time, tickMarkType) {
  const d = new Date(time * 1000);
  const types = LWC.TickMarkType;
  switch (tickMarkType) {
    case types.Year:
      return String(d.getFullYear());
    case types.Month:
      return d.toLocaleDateString([], { month: "short", year: "2-digit" });
    case types.DayOfMonth:
      return d.toLocaleDateString([], { month: "short", day: "numeric" });
    case types.TimeWithSeconds:
      return d.toLocaleTimeString();
    default:
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
}

function formatTime(epochSeconds, withDate) {
  const d = new Date(epochSeconds * 1000);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (!withDate) return time;
  return `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${time}`;
}

function buildChart(config) {
  const container = el("chart");
  container.innerHTML = "";
  state.series.clear();

  const chart = createChart(container, chartOptions());
  state.chart = chart;

  state.candles = chart.addSeries(
    CandlestickSeries,
    {
      upColor: "#2e9e6b",
      downColor: "#d1495b",
      borderUpColor: "#2e9e6b",
      borderDownColor: "#d1495b",
      wickUpColor: "#2e9e6b",
      wickDownColor: "#d1495b",
      priceLineWidth: 1,
    },
    0
  );
  state.markers = createSeriesMarkers(state.candles, []);

  let nextPane = 1;
  const paneHeights = [];
  for (const indicator of config.indicators) {
    const render = indicator.render;
    if (!render.series.length) continue;
    const paneIndex = render.pane === "own" ? nextPane++ : 0;

    for (const spec of render.series) {
      const ctor = SERIES_TYPES[spec.type] || LineSeries;
      const options = {
        color: spec.color,
        lineWidth: spec.line_width,
        lineStyle: spec.line_style,
        priceLineVisible: false,
        lastValueVisible: paneIndex === 0,
        crosshairMarkerVisible: spec.type === "line",
        priceFormat: { type: "price", precision: render.precision, minMove: 0.01 },
      };
      if (spec.type === "histogram") {
        options.base = 0;
        delete options.lineWidth;
        delete options.lineStyle;
      }
      if (spec.autoscale === false) {
        // Reference levels must not stretch the price scale.
        options.autoscaleInfoProvider = () => null;
      }
      const api = chart.addSeries(ctor, options, paneIndex);
      state.series.set(`${indicator.key}.${spec.key}`, { api, spec });
    }

    if (paneIndex > 0) paneHeights.push([paneIndex, render.height]);
  }

  // Applied only once every pane exists — see applyPaneLayout.
  applyPaneLayout(chart, paneHeights);
  renderLegend(config);
}

/**
 * Size the panes proportionally.
 *
 * Stretch factors rather than setHeight: a setHeight call rebalances every
 * other pane to keep the total constant, so setting panes one by one lets each
 * new pane steal space from the ones already sized (with three indicators the
 * first two collapse to slivers). Stretch factors are declarative, so all
 * panes settle at the requested proportions at once.
 *
 * Indicator panes are capped at 55% of the canvas combined, keeping the price
 * chart readable no matter how many indicators get added later.
 */
function applyPaneLayout(chart, requests) {
  if (!requests.length) return;
  const panes = chart.panes();
  const total = el("chart").clientHeight || 600;
  const wanted = requests.reduce((sum, [, h]) => sum + h, 0);
  if (!wanted) return;

  const indicatorShare = Math.min(wanted / total, 0.55);
  const scale = (indicatorShare * total) / wanted;

  if (panes[0]) panes[0].setStretchFactor(1 - indicatorShare);
  for (const [index, height] of requests) {
    if (panes[index]) panes[index].setStretchFactor((height * scale) / total);
  }
}

function renderLegend(config) {
  const rows = ['<div class="row"><span class="swatch" style="border-top-color:#8a93a5"></span>Price</div>'];
  for (const indicator of config.indicators) {
    for (const spec of indicator.render.series) {
      if (!spec.visible_in_legend) continue;
      const color = spec.color || spec.up_color || "#8a93a5";
      rows.push(
        `<div class="row"><span class="swatch" style="border-top-color:${color}"></span>${escapeHtml(spec.label)}</div>`
      );
    }
  }
  el("legend").innerHTML = rows.join("");
}

/* --------------------------------------------------------------- snapshot */

function applySnapshot(snap) {
  state.lastSnapshotAt = Date.now();

  state.candles.setData(
    snap.bars.map((b) => ({
      time: b.time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))
  );

  const markers = [];
  for (const [indicatorKey, result] of Object.entries(snap.indicators)) {
    for (const [seriesKey, points] of Object.entries(result.series || {})) {
      const entry = state.series.get(`${indicatorKey}.${seriesKey}`);
      if (!entry) continue;
      entry.api.setData(points.map((p) => toPoint(p, entry.spec)));
    }
    for (const m of result.markers || []) {
      markers.push({
        time: m.time,
        position: m.position,
        shape: m.shape,
        color: m.color,
        text: m.text,
      });
    }
  }

  // Series that got no data this snapshot must be cleared, or stale points
  // from the previous timeframe linger on the chart.
  for (const [key, entry] of state.series) {
    const [indicatorKey, seriesKey] = key.split(".");
    const result = snap.indicators[indicatorKey];
    if (!result || !result.series || !(seriesKey in result.series)) {
      entry.api.setData([]);
    }
  }

  markers.sort((a, b) => a.time - b.time);
  state.markers.setMarkers(markers);

  renderQuote(snap);
  renderMetrics(snap);
  renderStatus(snap.status);
}

function toPoint(p, spec) {
  if (spec.type !== "histogram") return { time: p.time, value: p.value };
  const up = spec.up_color || spec.color;
  const down = spec.down_color || spec.color;
  return { time: p.time, value: p.value, color: p.value >= 0 ? up : down };
}

/* ---------------------------------------------------------------- panels */

function renderQuote(snap) {
  el("symbol").textContent = snap.symbol;
  el("displayName").textContent = snap.display_name;
  el("price").textContent = snap.price == null ? "—" : formatNumber(snap.price, 2);

  const change = el("change");
  if (snap.change == null) {
    change.textContent = "—";
    change.className = "change";
    return;
  }
  const sign = snap.change > 0 ? "+" : "";
  const pct = snap.change_pct == null ? "" : ` (${sign}${snap.change_pct.toFixed(2)}%)`;
  change.textContent = `${sign}${formatNumber(snap.change, 2)}${pct}`;
  change.className = "change " + (snap.change > 0 ? "up" : snap.change < 0 ? "down" : "");
}

function renderMetrics(snap) {
  const parts = [];
  for (const indicator of state.config.indicators) {
    const result = snap.indicators[indicator.key];
    if (!result) continue;
    if (!result.stats.length && !result.error) continue;

    parts.push('<div class="metric-group">');
    parts.push(`<h3>${escapeHtml(indicator.name)}</h3>`);
    if (result.error) {
      parts.push(`<div class="err">${escapeHtml(result.error)}</div>`);
    }
    for (const stat of result.stats) {
      const value = stat.value == null ? "—" : formatStat(stat);
      const unit = stat.unit ? `<span class="unit">${escapeHtml(stat.unit)}</span>` : "";
      const title = stat.hint ? ` title="${escapeHtml(stat.hint)}"` : "";
      parts.push(
        `<div class="stat"${title}>` +
          `<span class="label">${escapeHtml(stat.label)}</span>` +
          `<span class="value ${stat.tone}">${escapeHtml(value)}${unit}</span>` +
        `</div>`
      );
    }
    parts.push("</div>");
  }
  el("metrics").innerHTML = parts.join("");
}

function formatStat(stat) {
  if (typeof stat.value !== "number") return String(stat.value);
  const text = formatNumber(stat.value, stat.precision);
  return stat.signed && stat.value > 0 ? `+${text}` : text;
}

function renderStatus(status) {
  if (!status) return;
  const age = status.last_poll ? Math.round(Date.now() / 1000 - status.last_poll) : null;
  const logged = status.last_logged_ts
    ? new Date(status.last_logged_ts * 1000).toLocaleTimeString()
    : "—";

  const bits = [
    `Feed <b>${escapeHtml(status.feed_name)}</b>`,
    `Polls <b>${status.poll_count}</b>${age == null ? "" : ` (${age}s ago)`}`,
    `1m bars <b>${status.minute_bars}</b>`,
    `Daily bars <b>${status.daily_bars}</b>`,
    `Logged this run <b>${status.bars_logged}</b>`,
    `Last close logged <b>${logged}</b>`,
  ];
  if (status.error_count) bits.push(`<span class="bad">Errors <b>${status.error_count}</b></span>`);
  if (status.last_error) bits.push(`<span class="bad">${escapeHtml(status.last_error)}</span>`);
  el("statusbar").innerHTML = bits.join(" · ");

  const dot = el("statusDot");
  const text = el("statusText");
  if (status.last_error) {
    dot.className = "dot error";
    text.textContent = "feed error";
  } else if (age != null && age > status.poll_seconds * 3) {
    dot.className = "dot stale";
    text.textContent = `stale (${age}s)`;
  } else {
    dot.className = "dot live";
    text.textContent = `live · ${status.poll_seconds}s poll`;
  }
}

/* ----------------------------------------------------------- timeframes */

function renderTimeframes() {
  const nav = el("timeframes");
  nav.innerHTML = "";
  for (const tf of state.config.timeframes) {
    const button = document.createElement("button");
    button.textContent = tf.key;
    button.title = tf.label;
    button.setAttribute("aria-pressed", String(tf.key === state.timeframe));
    button.addEventListener("click", () => selectTimeframe(tf.key));
    nav.appendChild(button);
  }
}

function selectTimeframe(key) {
  if (key === state.timeframe) return;
  state.timeframe = key;
  localStorage.setItem("mnq.timeframe", key);
  renderTimeframes();
  send({ type: "subscribe", tf: key });
  fetchSnapshot(key); // instant repaint; the socket keeps it updated
}

/* -------------------------------------------------------------- transport */

async function fetchSnapshot(tf) {
  try {
    const resp = await fetch(`/api/snapshot?tf=${encodeURIComponent(tf)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const snap = await resp.json();
    if (snap.timeframe === state.timeframe) applySnapshot(snap);
  } catch (err) {
    console.error("snapshot fetch failed", err);
  }
}

function send(message) {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify(message));
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${proto}://${location.host}/ws`);
  state.socket = socket;

  socket.addEventListener("open", () => {
    state.reconnectDelay = 1000;
    send({ type: "subscribe", tf: state.timeframe });
  });

  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "snapshot" && message.data.timeframe === state.timeframe) {
      applySnapshot(message.data);
    }
  });

  socket.addEventListener("close", () => {
    el("statusDot").className = "dot error";
    el("statusText").textContent = "disconnected";
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 15000);
  });

  socket.addEventListener("error", () => socket.close());
}

/* ------------------------------------------------------------------ utils */

function formatNumber(value, precision = 2) {
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

/* ------------------------------------------------------------------- boot */

async function main() {
  const resp = await fetch("/api/config");
  state.config = await resp.json();

  const saved = localStorage.getItem("mnq.timeframe");
  const known = state.config.timeframes.some((t) => t.key === saved);
  state.timeframe = known ? saved : state.config.default_timeframe;

  buildChart(state.config);
  renderTimeframes();
  await fetchSnapshot(state.timeframe);
  connect();

  // Keep the "seconds ago" readout honest between polls.
  setInterval(() => {
    if (state.lastSnapshotAt) fetch("/api/status").then((r) => r.json()).then(renderStatus).catch(() => {});
  }, 5000);
}

main().catch((err) => {
  console.error(err);
  el("statusText").textContent = "failed to start";
});
