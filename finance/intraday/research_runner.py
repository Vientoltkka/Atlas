"""Controlled live read-only intraday research runner.

Market data only. No account access, credentials or order execution.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from finance.intraday.collector import IntradayResearchCollector
from finance.intraday.persistence import IntradayResearchLedger
from finance.intraday.service import IntradaySignalService
from finance.market_data.models import Quote
from finance.market_data.quality import MarketDataQualityGate
from finance.market_data.revolut_x import RevolutXPublicMarketDataProvider


DEFAULT_LEDGER = Path(
    ".atlas/finance_intraday/live_research.jsonl"
)


@dataclass
class LiveResearchStats:
    polls: int = 0
    quotes_received: int = 0
    quotes_accepted: int = 0
    quotes_rejected_quality: int = 0
    quotes_rejected_duplicate: int = 0
    signals: int = 0
    candidates: int = 0
    outcomes: int = 0
    provider_errors: int = 0


class LiveIntradayResearchRunner:
    """Runs bounded, read-only live market research."""

    def __init__(
        self,
        *,
        symbol: str,
        duration_seconds: float,
        poll_seconds: float,
        ledger_path: str | Path = DEFAULT_LEDGER,
        provider=None,
        quality_gate: MarketDataQualityGate | None = None,
        collector: IntradayResearchCollector | None = None,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
        now_fn=lambda: datetime.now(timezone.utc),
    ) -> None:
        normalized = symbol.strip().upper().replace("/", "-")

        if not normalized:
            raise ValueError("symbol is required")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

        self._symbol = normalized
        self._duration_seconds = duration_seconds
        self._poll_seconds = poll_seconds

        self._provider = (
            provider
            if provider is not None
            else RevolutXPublicMarketDataProvider()
        )

        self._quality_gate = (
            quality_gate
            if quality_gate is not None
            else MarketDataQualityGate()
        )

        if collector is None:
            ledger = IntradayResearchLedger(ledger_path)
            collector = IntradayResearchCollector(
                signal_service=IntradaySignalService(),
                ledger=ledger,
            )

        self._collector = collector
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._now = now_fn

    def run(self) -> LiveResearchStats:
        stats = LiveResearchStats()
        started = self._monotonic()

        self._provider.connect()

        try:
            self._provider.subscribe([self._symbol])

            while (
                self._monotonic() - started
                < self._duration_seconds
            ):
                stats.polls += 1

                try:
                    events = tuple(self._provider.events())

                    for event in events:
                        if not isinstance(event, Quote):
                            continue

                        stats.quotes_received += 1

                        decision = self._quality_gate.evaluate(
                            event,
                            now=self._now(),
                        )

                        if not decision.accepted:
                            stats.quotes_rejected_quality += 1
                            continue

                        result = self._collector.ingest_quote(event)

                        if not result.accepted_observation:
                            stats.quotes_rejected_duplicate += 1
                            continue

                        stats.quotes_accepted += 1
                        stats.outcomes += result.recorded_outcomes

                        if result.signal is not None:
                            stats.signals += 1

                            if (
                                result.signal.action.value
                                == "CANDIDATE"
                            ):
                                stats.candidates += 1

                except (
                    ConnectionError,
                    TimeoutError,
                    OSError,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    stats.provider_errors += 1
                    print(
                        f"[provider-error] "
                        f"{type(exc).__name__}: {exc}"
                    )

                remaining = (
                    self._duration_seconds
                    - (self._monotonic() - started)
                )

                if remaining <= 0:
                    break

                self._sleep(
                    min(self._poll_seconds, remaining)
                )

        except KeyboardInterrupt:
            print("\nResearch run interrupted by user.")

        finally:
            self._provider.disconnect()

        return stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atlas live read-only intraday research"
        )
    )

    parser.add_argument(
        "--symbol",
        default="BTC-USD",
    )

    parser.add_argument(
        "--minutes",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER),
    )

    return parser


def main() -> int:
    args = _parser().parse_args()

    runner = LiveIntradayResearchRunner(
        symbol=args.symbol,
        duration_seconds=args.minutes * 60,
        poll_seconds=args.poll_seconds,
        ledger_path=args.ledger,
    )

    print(
        f"Atlas Live Research | {args.symbol} | "
        f"{args.minutes:g} min | "
        f"poll={args.poll_seconds:g}s"
    )
    print("READ-ONLY: no orders, no account access.\n")

    stats = runner.run()

    print("\n=== LIVE RESEARCH SUMMARY ===")
    print(f"polls: {stats.polls}")
    print(f"quotes_received: {stats.quotes_received}")
    print(f"quotes_accepted: {stats.quotes_accepted}")
    print(
        "quotes_rejected_quality: "
        f"{stats.quotes_rejected_quality}"
    )
    print(
        "quotes_rejected_duplicate: "
        f"{stats.quotes_rejected_duplicate}"
    )
    print(f"signals: {stats.signals}")
    print(f"candidates: {stats.candidates}")
    print(f"outcomes: {stats.outcomes}")
    print(f"provider_errors: {stats.provider_errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
