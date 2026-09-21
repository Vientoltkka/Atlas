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
from finance.intraday.time_replay import TIME_BASED_STRATEGY_VERSION
from finance.market_data.models import Quote
from finance.market_data.quality import MarketDataQualityGate
from finance.market_data.revolut_x import RevolutXPublicMarketDataProvider


DEFAULT_LEDGER = Path(
    ".atlas/finance_intraday/live_research.jsonl"
)
DEFAULT_V2_LEDGER = Path(
    ".atlas/finance_intraday/time_based_v2_research.jsonl"
)
V1_STRATEGY_VERSION = IntradayResearchLedger.DEFAULT_STRATEGY_VERSION
STRATEGY_VERSIONS = (
    V1_STRATEGY_VERSION,
    TIME_BASED_STRATEGY_VERSION,
)
DEFAULT_PROVIDER_RETRIES = 3
DEFAULT_PROVIDER_RETRY_BACKOFF = (1.0, 2.0, 4.0)
PROVIDER_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
    ValueError,
    RuntimeError,
)
RETRYABLE_PROVIDER_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
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
    provider_errors_finales: int = 0
    provider_retries_recovered: int = 0
    provider_retry_attempts: int = 0


class LiveIntradayResearchRunner:
    """Runs bounded, read-only live market research."""

    def __init__(
        self,
        *,
        symbol: str,
        duration_seconds: float,
        poll_seconds: float,
        ledger_path: str | Path | None = None,
        strategy_version: str = V1_STRATEGY_VERSION,
        provider=None,
        quality_gate: MarketDataQualityGate | None = None,
        collector: IntradayResearchCollector | None = None,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
        now_fn=lambda: datetime.now(timezone.utc),
        max_provider_retries: int = DEFAULT_PROVIDER_RETRIES,
        provider_retry_backoff: tuple[float, ...] = (
            *DEFAULT_PROVIDER_RETRY_BACKOFF,
        ),
    ) -> None:
        normalized = symbol.strip().upper().replace("/", "-")

        if not normalized:
            raise ValueError("symbol is required")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if max_provider_retries < 0:
            raise ValueError("max_provider_retries must not be negative")
        if any(delay < 0 for delay in provider_retry_backoff):
            raise ValueError("provider retry backoff must not be negative")
        if max_provider_retries and not provider_retry_backoff:
            raise ValueError(
                "provider_retry_backoff is required when retries are enabled"
            )
        if strategy_version not in STRATEGY_VERSIONS:
            raise ValueError(
                f"unsupported strategy_version: {strategy_version}"
            )

        self._symbol = normalized
        self._duration_seconds = duration_seconds
        self._poll_seconds = poll_seconds
        self._strategy_version = strategy_version
        self._max_provider_retries = max_provider_retries
        self._provider_retry_backoff = provider_retry_backoff

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
            if ledger_path is None:
                ledger_path = (
                    DEFAULT_V2_LEDGER
                    if strategy_version == TIME_BASED_STRATEGY_VERSION
                    else DEFAULT_LEDGER
                )

            ledger = IntradayResearchLedger(
                ledger_path,
                strategy_version=strategy_version,
            )
            existing_versions = {
                record.get("strategy_version")
                for record in ledger.records()
            }
            incompatible_versions = existing_versions - {
                strategy_version
            }
            if incompatible_versions:
                versions = ", ".join(
                    sorted(str(version) for version in incompatible_versions)
                )
                raise ValueError(
                    f"ledger {ledger.path} contains incompatible "
                    f"strategy_version(s): {versions}; expected "
                    f"{strategy_version}"
                )

            if strategy_version == TIME_BASED_STRATEGY_VERSION:
                collector = IntradayResearchCollector.for_time_based_v2(
                    ledger=ledger,
                )
            else:
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

                events = None
                retry_count = 0
                provider_error = None

                while True:
                    try:
                        events = tuple(self._provider.events())
                        break
                    except RETRYABLE_PROVIDER_ERRORS as exc:
                        provider_error = exc
                        remaining = (
                            self._duration_seconds
                            - (self._monotonic() - started)
                        )

                        if (
                            retry_count >= self._max_provider_retries
                            or remaining <= 0
                        ):
                            break

                        delay_index = min(
                            retry_count,
                            len(self._provider_retry_backoff) - 1,
                        )
                        delay = self._provider_retry_backoff[delay_index]
                        self._sleep(min(delay, remaining))

                        if (
                            self._monotonic() - started
                            >= self._duration_seconds
                        ):
                            break

                        retry_count += 1
                        stats.provider_retry_attempts += 1

                if events is None:
                    stats.provider_errors += 1
                    stats.provider_errors_finales += 1
                    print(
                        f"[provider-error] "
                        f"{type(provider_error).__name__}: {provider_error}"
                    )
                else:
                    if retry_count:
                        stats.provider_retries_recovered += 1
                        print(
                            "[provider-retry-recovered] "
                            f"quote poll recovered after {retry_count} "
                            "retry attempt(s)"
                        )

                    try:
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

                    except RETRYABLE_PROVIDER_ERRORS as exc:
                        stats.provider_errors += 1
                        stats.provider_errors_finales += 1
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
        default=None,
    )

    parser.add_argument(
        "--strategy-version",
        choices=STRATEGY_VERSIONS,
        default=V1_STRATEGY_VERSION,
    )

    return parser


def main() -> int:
    args = _parser().parse_args()

    runner = LiveIntradayResearchRunner(
        symbol=args.symbol,
        duration_seconds=args.minutes * 60,
        poll_seconds=args.poll_seconds,
        ledger_path=args.ledger,
        strategy_version=args.strategy_version,
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
    print(f"provider_errors_finales: {stats.provider_errors_finales}")
    print(
        "provider_retries_recovered: "
        f"{stats.provider_retries_recovered}"
    )
    print(f"provider_retry_attempts: {stats.provider_retry_attempts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
