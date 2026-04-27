"""Tiny in-process pub/sub for live /graph updates.

WebSocket clients subscribe to an `EventBus`; every state-changing endpoint
publishes a small JSON event. Async-friendly: subscribers are bounded queues,
so a slow client never blocks publishers — events get dropped on overflow.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

QUEUE_MAX = 256


@dataclass
class Event:
    type: str  # node.created | node.updated | node.deleted | training.step | training.epoch | proposals.updated
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": self.type, "payload": self.payload, "ts": self.ts}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, event: Event) -> None:
        dead: list[asyncio.Queue] = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            # Slow client — drop them; they'll reconnect.
            self.unsubscribe(q)

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)
