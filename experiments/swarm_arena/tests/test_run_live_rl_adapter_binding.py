from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from experiments.swarm_arena.scripts import run_live_rl


class _Response:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Client:
    def __init__(self, registries: dict[str, dict[str, object]]) -> None:
        self.registries = registries
        self.posts: list[tuple[str, dict[str, str]]] = []

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str) -> _Response:
        return _Response(self.registries[url])

    async def post(self, url: str, json: dict[str, str]) -> _Response:
        self.posts.append((url, json))
        base_url = url.split("/v1/", 1)[0]
        registry_url = f"{base_url}/v1/models"
        if url.endswith("/load_lora_adapter"):
            self.registries[registry_url] = {
                "data": [{"id": json["lora_name"], "root": json["lora_path"]}]
            }
        return _Response({})


def _patch_client(monkeypatch: Any, client: _Client) -> None:
    monkeypatch.setattr(
        run_live_rl.httpx,
        "AsyncClient",
        lambda **_kwargs: client,
    )


def test_replace_adapter_skips_reload_when_every_registry_matches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    adapter = tmp_path.resolve()
    registries = {
        f"http://server-{index}/v1/models": {
            "data": [{"id": "blue-0", "root": str(adapter)}]
        }
        for index in range(2)
    }
    client = _Client(registries)
    _patch_client(monkeypatch, client)

    asyncio.run(
        run_live_rl.replace_adapter(
            ("http://server-0", "http://server-1"), "blue-0", adapter
        )
    )

    assert client.posts == []


def test_replace_adapter_reloads_when_any_registry_path_differs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    adapter = tmp_path.resolve()
    registries = {
        "http://server-0/v1/models": {
            "data": [{"id": "blue-0", "root": str(adapter)}]
        },
        "http://server-1/v1/models": {
            "data": [{"id": "blue-0", "root": "/wrong/path"}]
        },
    }
    client = _Client(registries)
    _patch_client(monkeypatch, client)

    asyncio.run(
        run_live_rl.replace_adapter(
            ("http://server-0", "http://server-1"), "blue-0", adapter
        )
    )

    assert len(client.posts) == 4
    assert {url.rsplit("/", 1)[-1] for url, _ in client.posts} == {
        "load_lora_adapter",
        "unload_lora_adapter",
    }
