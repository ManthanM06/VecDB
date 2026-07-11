"""
vecdb_client.py — Python SDK for the VecDB REST API.

Usage:
    from vecdb_client import VecDBClient

    client = VecDBClient("http://localhost:8080")
    client.insert(42, [0.1, 0.2, ...])
    results = client.search([0.1, 0.2, ...], k=5)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class InsertItem:
    id: int
    vector: list[float]
    metadata: Optional[dict] = None   # reserved for Phase 7


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
            backoff_factor=0.5,          # waits 0.5s, 1s, 2s between retries
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
        """Insert a single vector. Returns True on success."""
        payload: dict = {"id": vector_id, "vector": vector}
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
        total = len(items)
        successful = 0
        start = time.time()

        def _insert_one(item: InsertItem) -> bool:
            return self.insert(item.id, item.vector, item.metadata)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_insert_one, item): item for item in items}
            for i, future in enumerate(as_completed(futures), start=1):
                if future.result():
                    successful += 1
                if show_progress and i % max(1, total // 20) == 0:
                    pct = i / total * 100
                    elapsed = time.time() - start
                    rate = i / elapsed
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
    # Search
    # ------------------------------------------------------------------

    def search(self, vector: list[float], k: int = 10) -> list[SearchResult]:
        """Return top-k nearest neighbours sorted by distance (closest first)."""
        try:
            resp = self.session.post(
                f"{self.base_url}/search",
                json={"vector": vector, "k": k},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return [SearchResult(**r) for r in resp.json()]
        except requests.RequestException as exc:
            print(f"[VecDBClient] search failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> bool:
        """Trigger the server to flush all data to disk."""
        try:
            resp = self.session.post(
                f"{self.base_url}/save", timeout=self.timeout
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            print(f"[VecDBClient] save failed: {exc}")
            return False

    def ping(self) -> bool:
        """Quick health check — returns True if the server is reachable."""
        try:
            # Use a zero-vector search; if the db is empty it returns [] gracefully
            self.session.post(
                f"{self.base_url}/search",
                json={"vector": [0.0], "k": 1},
                timeout=3.0,
            )
            return True
        except requests.RequestException:
            return False
