"""Tests for webhook idempotency stores (Atlas Phase 3, Block 1)."""

from __future__ import annotations

import threading

import pytest

from pathlib import Path

from channels.webhook_idempotency import IdempotencyStore, SqliteIdempotencyStore


@pytest.fixture()
def tmp_dir():
    import shutil
    import tempfile

    path = tempfile.mkdtemp(prefix="atlas-idem-")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TestInMemoryStore:
    def test_first_reserve_accepted(self) -> None:
        store = IdempotencyStore()
        assert store.check_and_reserve("wamid.1") is True

    def test_same_key_is_duplicate(self) -> None:
        store = IdempotencyStore()
        store.check_and_reserve("wamid.1")
        assert store.check_and_reserve("wamid.1") is False


class TestSqliteStore:
    def test_first_reserve_accepted(self, tmp_dir) -> None:
        store = SqliteIdempotencyStore(db_path=Path(tmp_dir) / "idem.db")
        assert store.check_and_reserve("wamid.1") is True

    def test_same_key_is_duplicate(self, tmp_dir) -> None:
        store = SqliteIdempotencyStore(db_path=Path(tmp_dir) / "idem.db")
        store.check_and_reserve("wamid.1")
        assert store.check_and_reserve("wamid.1") is False

    def test_concurrent_same_key_exactly_one_reservation(self, tmp_dir) -> None:
        store = SqliteIdempotencyStore(db_path=Path(tmp_dir) / "idem.db")
        results: list[bool] = []
        lock = threading.Lock()

        def reserve() -> None:
            accepted = store.check_and_reserve("wamid.race")
            with lock:
                results.append(accepted)

        threads = [threading.Thread(target=reserve) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results.count(True) == 1
        assert results.count(False) == 7

    def test_second_instance_recognizes_previous_reservation(self, tmp_dir) -> None:
        db_path = Path(tmp_dir) / "idem.db"
        first = SqliteIdempotencyStore(db_path=db_path)
        assert first.check_and_reserve("wamid.persisted") is True
        second = SqliteIdempotencyStore(db_path=db_path)
        assert second.check_and_reserve("wamid.persisted") is False
        assert second.check_and_reserve("wamid.new") is True

    def test_ttl_expiry_allows_reprocessing(self, tmp_dir) -> None:
        store = SqliteIdempotencyStore(db_path=Path(tmp_dir) / "idem.db", ttl_seconds=1)
        assert store.check_and_reserve("wamid.old") is True
        import sqlite3
        import time

        with sqlite3.connect(Path(tmp_dir) / "idem.db") as connection:
            connection.execute(
                "UPDATE whatsapp_idempotency SET reserved_at = reserved_at - 7200"
            )
            connection.commit()
        time.sleep(0.01)
        assert store.check_and_reserve("wamid.old") is True


class TestCrossImplementationContract:
    def test_memory_store_restart_still_reprocesses_documented_limitation(self) -> None:
        first = IdempotencyStore()
        assert first.check_and_reserve("wamid.x") is True
        second = IdempotencyStore()
        assert second.check_and_reserve("wamid.x") is True

    def test_sqlite_store_rejects_invalid_configuration(self, tmp_dir) -> None:
        with pytest.raises(ValueError):
            SqliteIdempotencyStore(db_path=Path(tmp_dir) / "idem.db", ttl_seconds=0)

    def test_in_memory_store_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            IdempotencyStore(ttl_seconds=0)


def _mp_reserve_worker(db_path: str, barrier, queue) -> None:
    from channels.webhook_idempotency import SqliteIdempotencyStore

    store = SqliteIdempotencyStore(db_path=db_path)
    barrier.wait(timeout=30)
    queue.put(store.check_and_reserve("wamid.multiprocess"))


class TestMultiProcess:
    def test_same_event_id_across_processes_reserves_exactly_once(self, tmp_dir) -> None:
        import multiprocessing

        db_path = str(Path(tmp_dir) / "idem.db")
        # Initialize schema before forking workers.
        SqliteIdempotencyStore(db_path=db_path)

        context = multiprocessing.get_context("spawn")
        processes = 4
        barrier = context.Barrier(processes)
        queue = context.Queue()
        workers = [
            context.Process(
                target=_mp_reserve_worker,
                args=(db_path, barrier, queue),
            )
            for _ in range(processes)
        ]
        for worker in workers:
            worker.start()
        results = [queue.get(timeout=60) for _ in range(processes)]
        for worker in workers:
            worker.join(timeout=30)
        assert all(worker.exitcode == 0 for worker in workers)
        assert results.count(True) == 1
        assert results.count(False) == processes - 1

        # No corruption: the database still answers consistently.
        final = SqliteIdempotencyStore(db_path=db_path)
        assert final.check_and_reserve("wamid.multiprocess") is False
        assert final.check_and_reserve("wamid.other") is True
