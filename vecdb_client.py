"""
vecdb_client.py — Python SDK for the VecDB REST API.

Usage:
    from vecdb_client import VecDBClient

    client = VecDBClient("http://localhost:8080")

    # Insert with optional metadata
    client.insert(42, [0.1, 0.2, ...], metadata={"author": "alice", "year": 2026})

    # Search (with optional equality filter)
    results = client.search([0.1, 0.2, ...], k=5)
    results = client.search([0.1, 0.2, ...], k=5, filter_key="author", filter_value="alice")

    # Soft-delete
    client.delete(42)

    # Server status
    info = client.status()
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    id: int
    distance: float
    metadata: dict = field(default_factory=dict)


@dataclass
class InsertItem:
    id: int
    vector: list[float]
    metadata: Optional[dict] = None


@dataclass
class ServerStatus:
    live_vectors: int
    total_vectors: int
    deleted_vectors: int
    save_in_progress: bool
    dimensions: int


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class VecDBClient:
    """Thread-safe HTTP client for VecDB with connection pooling and retries."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Session with connection pooling and automatic retry on transient errors
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=50,
        )
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # ------------------------------------------------------------------
    # Single insert
    # ------------------------------------------------------------------

    def insert(
        self,
        vector_id: int,
        vector: list[float],
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Insert a single vector with optional JSON metadata.
        Returns True on success.
        """
        payload: dict[str, Any] = {"id": vector_id, "vector": vector}
        if metadata:
            payload["metadata"] = metadata

        try:
            resp = self.session.post(
                f"{self.base_url}/insert",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            print(f"[VecDBClient] insert({vector_id}) failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Batch insert — concurrent, shows progress
    # ------------------------------------------------------------------

    def batch_insert(
        self,
        items: list[InsertItem],
        workers: int = 32,
        show_progress: bool = True,
    ) -> tuple[int, int]:
        """
        Insert a list of InsertItem concurrently.
        Returns (successful_count, total_count).
        """
        total      = len(items)
        successful = 0
        start      = time.time()

        def _insert_one(item: InsertItem) -> bool:
            return self.insert(item.id, item.vector, item.metadata)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_insert_one, item): item for item in items}
            for i, future in enumerate(as_completed(futures), start=1):
                if future.result():
                    successful += 1
                if show_progress and i % max(1, total // 20) == 0:
                    pct     = i / total * 100
                    elapsed = time.time() - start
                    rate    = i / elapsed
                    print(
                        f"  [{pct:5.1f}%] {i}/{total} vectors  |  "
                        f"{rate:.0f} vec/s",
                        end="\r",
                    )

        if show_progress:
            elapsed = time.time() - start
            print(
                f"\n  Done: {successful}/{total} vectors inserted in "
                f"{elapsed:.2f}s ({total/elapsed:.0f} vec/s)"
            )
        return successful, total

    # ------------------------------------------------------------------
    # Search (with optional metadata filter)
    # ------------------------------------------------------------------

    def search(
        self,
        vector: list[float],
        k: int = 10,
        filter_key: Optional[str] = None,
        filter_value: Any = None,
    ) -> list[SearchResult]:
        """
        Return top-k nearest neighbours sorted by distance (closest first).

        Args:
            vector:       Query embedding.
            k:            Number of results to return.
            filter_key:   If set, restrict results to vectors whose metadata
                          contains this key with the given filter_value.
            filter_value: The value to match (any JSON-serialisable type).
        """
        payload: dict[str, Any] = {"vector": vector, "k": k}

        if filter_key is not None and filter_value is not None:
            payload["filter"] = {"key": filter_key, "value": filter_value}

        try:
            resp = self.session.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return [
                SearchResult(
                    id=r["id"],
                    distance=r["distance"],
                    metadata=r.get("metadata", {}),
                )
                for r in resp.json()
            ]
        except requests.RequestException as exc:
            print(f"[VecDBClient] search failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Delete (soft-delete / tombstone)
    # ------------------------------------------------------------------

    def delete(self, vector_id: int) -> bool:
        """
        Soft-delete a vector by ID. The vector is tombstoned — excluded from
        all future searches but graph topology is preserved.
        Returns True on success.
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/delete",
                json={"id": vector_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            print(f"[VecDBClient] delete({vector_id}) failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> bool:
        """
        Trigger an async background save to disk.
        Returns True if the save was initiated, False if one is already running.
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/save", timeout=self.timeout
            )
            if resp.status_code == 202:
                return True
            if resp.status_code == 409:
                print("[VecDBClient] save skipped: a save is already in progress.")
                return False
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            print(f"[VecDBClient] save failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Status & Info
    # ------------------------------------------------------------------

    def status(self) -> Optional[ServerStatus]:
        """Return detailed server status including vector counts and save state."""
        try:
            resp = self.session.get(
                f"{self.base_url}/status", timeout=self.timeout
            )
            resp.raise_for_status()
            d = resp.json()
            return ServerStatus(
                live_vectors    = d["live_vectors"],
                total_vectors   = d["total_vectors"],
                deleted_vectors = d["deleted_vectors"],
                save_in_progress= d["save_in_progress"],
                dimensions      = d["dimensions"],
            )
        except requests.RequestException as exc:
            print(f"[VecDBClient] status() failed: {exc}")
            return None

    def info(self) -> Optional[dict]:
        """Return a brief {live, deleted, total} summary (legacy /info endpoint)."""
        try:
            resp = self.session.get(
                f"{self.base_url}/info", timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"[VecDBClient] info() failed: {exc}")
            return None

    def ping(self) -> bool:
        """
        Quick health check — returns True if the server TCP port is open.
        Uses a raw socket so it's completely dimension-agnostic.
        """
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(self.base_url)
        host   = parsed.hostname or "localhost"
        port   = parsed.port or 8080
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            return False
