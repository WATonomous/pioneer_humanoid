"""The real aiohttp server: page served, Meta Browser packets queued, other browsers refused."""
import asyncio
import json
import socket
import time

import pytest

aiohttp = pytest.importorskip("aiohttp")

from test_clock_width_tracker import packet  # noqa: E402
from umi_gripper.quest_tracker import QuestTracker  # noqa: E402

QUEST_UA = "Mozilla/5.0 (X11; Linux x86_64; Quest 2) OculusBrowser/35.0"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def stream(port, packets, user_agent=QUEST_UA):
    async with aiohttp.ClientSession(headers={"User-Agent": user_agent}) as client:
        async with client.get(f"http://127.0.0.1:{port}/") as response:
            page = await response.text()
        async with client.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
            replies = []
            for p in packets:
                await ws.send_str(json.dumps(p))
                replies.append(json.loads((await ws.receive()).data))
        return page, replies


def test_server_round_trip():
    port = free_port()
    tracker = QuestTracker("right", port=port).start()
    try:
        tracker.set_status("REC 1.0 s")
        page, replies = asyncio.run(stream(port, [packet(i, buttons={"record": i == 2}) for i in range(1, 4)]))
        assert "navigator.xr" in page
        assert replies[-1] == {"status": "REC 1.0 s"}
        samples = tracker.drain()
        assert [s.client_ms for s in samples] == [10, 20, 30] and all(s.tracked for s in samples)
        assert [e.name for e in tracker.poll_events()] == ["primary"]
        assert samples[0].host_s <= time.monotonic()
        with pytest.raises(aiohttp.WSServerHandshakeError):
            asyncio.run(stream(port, [packet(1)], user_agent="Mozilla/5.0 Chrome/120"))
    finally:
        tracker.stop()
