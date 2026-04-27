"""EventBus tests."""

from __future__ import annotations

import asyncio

import pytest

from brain.events import Event, EventBus


def test_subscribe_unsubscribe():
    bus = EventBus()
    q = bus.subscribe()
    assert bus.n_subscribers == 1
    bus.unsubscribe(q)
    assert bus.n_subscribers == 0


def test_publish_routes_to_all_subs():
    async def run():
        bus = EventBus()
        a = bus.subscribe()
        b = bus.subscribe()
        bus.publish(Event(type="ping", payload={"k": 1}))
        ea = await a.get()
        eb = await b.get()
        assert ea.type == "ping"
        assert eb.payload["k"] == 1

    asyncio.run(run())


def test_event_to_dict_roundtrip():
    e = Event(type="x", payload={"k": 1})
    d = e.to_dict()
    assert d["type"] == "x" and d["payload"] == {"k": 1} and "ts" in d


def test_publish_drops_full_subscriber():
    async def run():
        bus = EventBus()
        q = bus.subscribe()
        # Fill up.
        for i in range(q.maxsize):
            bus.publish(Event(type="x", payload={"i": i}))
        assert bus.n_subscribers == 1
        # Overflow drops the slow client.
        bus.publish(Event(type="x", payload={"overflow": True}))
        assert bus.n_subscribers == 0

    asyncio.run(run())
