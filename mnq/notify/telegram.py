"""Telegram delivery.

Messages are written for someone glancing at a phone: the actionable numbers
(side, entry, stop, target) first, the reasoning underneath. A signal that
cannot be acted on in five seconds is not much use intraday.

Delivery failures never propagate. A network blip must not kill the trading
loop or, worse, leave a position unmanaged because a message could not be sent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ..config import POINT_VALUE_USD, TelegramConfig
from ..trade.manager import (
    EXIT_EARLY, EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_TRAIL, LONG, Trade,
)
from ..trade.signals import Signal

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
TIMEOUT = 10

EXIT_EMOJI = {
    EXIT_TARGET: "✅",
    EXIT_STOP: "🛑",
    EXIT_TRAIL: "🔒",
    EXIT_EARLY: "⚠️",
    EXIT_TIME: "⏱",
}


class TelegramNotifier:
    """Thin Telegram Bot API client with retry and graceful degradation."""

    def __init__(self, cfg: TelegramConfig):
        self.cfg = cfg
        self._session = requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.cfg.bot_token and self.cfg.chat_id)

    def send(self, text: str, retries: int = 3) -> bool:
        """Send a Markdown message. Returns success; never raises."""
        if not self.cfg.enabled:
            log.debug("telegram disabled; message suppressed")
            return False
        if not self.configured:
            # Loud, because a silently unconfigured notifier looks exactly like
            # a strategy that is not finding trades.
            log.warning(
                "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; message not sent:\n%s",
                text,
            )
            return False

        url = f"{API_BASE}/bot{self.cfg.bot_token}/sendMessage"
        payload = {
            "chat_id": self.cfg.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        for attempt in range(retries):
            try:
                r = self._session.post(url, json=payload, timeout=TIMEOUT)
                if r.status_code == 200:
                    return True
                if r.status_code == 429:
                    wait = int(r.json().get("parameters", {}).get("retry_after", 2))
                    log.warning("telegram rate limited; waiting %ds", wait)
                    time.sleep(wait)
                    continue
                log.error("telegram %s: %s", r.status_code, r.text[:200])
            except requests.RequestException as exc:
                log.warning("telegram attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2**attempt)

        log.error("telegram delivery failed after %d attempts", retries)
        return False

    # ---------------------------------------------------------- formatting

    def send_signal(self, signal: Signal, extra: dict[str, Any] | None = None) -> bool:
        return self.send(format_signal(signal, self.cfg.include_features, extra))

    def send_exit(self, trade: Trade, reason_note: str = "") -> bool:
        return self.send(format_exit(trade, reason_note))

    def send_update(self, trade: Trade, price: float, note: str) -> bool:
        return self.send(format_update(trade, price, note))


def _fmt(price: float) -> str:
    return f"{price:,.2f}"


def format_signal(
    signal: Signal, include_features: bool = True, extra: dict[str, Any] | None = None
) -> str:
    """The entry alert."""
    arrow = "🟢 LONG" if signal.direction == LONG else "🔴 SHORT"
    lines = [
        f"*{arrow}  MNQ*",
        f"`{signal.timestamp:%Y-%m-%d %H:%M} UTC`",
        "",
        f"*Entry*   `{_fmt(signal.entry)}`",
        f"*Stop*    `{_fmt(signal.stop)}`  ({signal.risk_points:.1f} pts / ${signal.risk_usd:,.0f})",
        f"*Target*  `{_fmt(signal.target)}`  ({signal.expected_points:.1f} pts / ${signal.reward_usd:,.0f})",
        "",
        f"R:R `{signal.reward_risk:.2f}`   Confidence `{signal.probability:.1%}`   ATR `{signal.atr:.1f}`",
    ]

    if include_features and signal.components:
        parts = []
        for key in ("p_xgb", "p_lgbm", "p_meta", "fwd_pred"):
            if key in signal.components:
                v = signal.components[key]
                parts.append(f"{key}={v:+.3f}" if key == "fwd_pred" else f"{key}={v:.3f}")
        if parts:
            lines += ["", "_Models:_ `" + "  ".join(parts) + "`"]

    if signal.context:
        ctx = "  ".join(f"{k}={v}" for k, v in list(signal.context.items())[:6])
        lines += [f"_Context:_ `{ctx}`"]

    lines += ["", "_Monitoring for early exit._"]
    return "\n".join(lines)


def format_exit(trade: Trade, note: str = "") -> str:
    """The close alert."""
    emoji = EXIT_EMOJI.get(trade.exit_reason or "", "◻️")
    pts = trade.realised_points
    usd = pts * POINT_VALUE_USD * trade.contracts
    side = "LONG" if trade.direction == LONG else "SHORT"
    r = pts / trade.risk_points if trade.risk_points else 0.0
    verdict = "WIN" if pts > 0 else "LOSS"

    lines = [
        f"{emoji} *{side} CLOSED — {verdict}*",
        f"`{trade.exit_time:%Y-%m-%d %H:%M} UTC`" if trade.exit_time else "",
        "",
        f"Entry `{_fmt(trade.entry_price)}` → Exit `{_fmt(trade.exit_price or 0)}`",
        f"*{pts:+.1f} pts*  (`${usd:+,.0f}`, {r:+.2f}R)",
        f"Reason: `{trade.exit_reason}`   Held: `{trade.bars_held}` bars",
        f"MFE `{trade.mfe_points:.1f}` / MAE `{trade.mae_points:.1f}` pts",
    ]
    if trade.exit_probability is not None:
        lines.append(f"Model confidence at exit: `{trade.exit_probability:.1%}`")
    if note:
        lines += ["", f"_{note}_"]
    return "\n".join(l for l in lines if l != "")


def format_update(trade: Trade, price: float, note: str) -> str:
    """An in-flight status change (breakeven moved, stop trailed)."""
    side = "LONG" if trade.direction == LONG else "SHORT"
    pts = trade.unrealised_points(price)
    return "\n".join(
        [
            f"🔁 *{side} update*",
            f"Price `{_fmt(price)}`   Unrealised *{pts:+.1f} pts* ({trade.r_multiple(price):+.2f}R)",
            f"Stop now `{_fmt(trade.stop)}`   Target `{_fmt(trade.target)}`",
            f"_{note}_",
        ]
    )


def format_heartbeat(state: dict[str, Any]) -> str:
    """Periodic 'still alive' message, so silence can be distinguished from
    a dead process."""
    return "\n".join(
        [
            "💓 *MNQ system heartbeat*",
            f"Bars stored: `{state.get('bars', 0)}`",
            f"Last bar: `{state.get('last_bar', 'n/a')}`",
            f"Last price: `{state.get('last_price', 'n/a')}`",
            f"Open trades: `{state.get('open_trades', 0)}`",
            f"Signals today: `{state.get('signals_today', 0)}`",
        ]
    )
