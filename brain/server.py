"""FastAPI HTTP server. Contract is the surface Jarvis 2.0's vault connector hits."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncio

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from brain.embed import get_embedder, text_for_embedding
from brain.events import Event, EventBus
from brain.graph_html import GRAPH_HTML
from brain.index import IndexedDoc, MemoryIndex
from brain.memory import Memory
from brain.note import Note, now_utc
from brain.rerank import get_reranker
from brain.search import search as search_vault
from brain.slug import InvalidSlugError, validate_slug
from brain.vault import (
    NoteExistsError,
    NoteNotFoundError,
    Vault,
)


class NoteCreate(BaseModel):
    slug: str
    title: str
    body: str = ""
    tags: list[str] = Field(default_factory=list)


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    tags: Optional[list[str]] = None


class NoteOut(BaseModel):
    slug: str
    title: str
    created: datetime
    updated: datetime
    tags: list[str]
    body: str


class NoteSummaryOut(BaseModel):
    slug: str
    title: str
    tags: list[str]
    updated: datetime


class LinksOut(BaseModel):
    outgoing: list[str]
    incoming: list[str]


def _to_out(note: Note) -> NoteOut:
    return NoteOut(
        slug=note.slug,
        title=note.title,
        created=note.created,
        updated=note.updated,
        tags=list(note.tags),
        body=note.body,
    )


def create_app(vault_path: Path) -> FastAPI:
    app = FastAPI(title="brain", version="0.2.0")
    vault = Vault(vault_path)
    embedder = get_embedder()
    index = MemoryIndex(embedder, path=vault_path / ".brain" / "index.npz")
    memory = Memory(vault, index, reranker=get_reranker())
    bus = EventBus()
    app.state.bus = bus

    def _node_payload(note: Note) -> dict:
        return {
            "id": note.slug,
            "title": note.title,
            "tags": list(note.tags),
            "kind": note.kind,
            "importance": note.importance,
            "preview": note.body[:240],
            "ghost": False,
            "xy": list(note.xy) if note.xy else None,
        }
    # Bootstrap index from disk on startup (first-run builds it lazily).
    try:
        index = MemoryIndex.load(embedder, vault_path / ".brain" / "index.npz")
        memory.index = index
    except Exception:
        pass
    if not index.all_slugs():
        memory.reindex_from_vault()
        try:
            index.save()
        except Exception:
            pass

    def _reindex_one(note: Note) -> None:
        memory.index.upsert(
            IndexedDoc(
                slug=note.slug,
                text=text_for_embedding(note.title, note.body, note.tags),
                kind=note.kind,
                tags=list(note.tags),
                importance=note.importance,
                updated_ts=note.updated.timestamp(),
            )
        )

    def get_vault() -> Vault:
        return vault

    def get_memory() -> Memory:
        return memory

    @app.get("/notes", response_model=list[NoteSummaryOut])
    def list_notes(
        tag: Optional[str] = None, v: Vault = Depends(get_vault)
    ) -> list[NoteSummaryOut]:
        return [
            NoteSummaryOut(slug=s.slug, title=s.title, tags=s.tags, updated=s.updated)
            for s in v.list(tag=tag)
        ]

    @app.post("/notes", response_model=NoteOut, status_code=201)
    def create_note(payload: NoteCreate, v: Vault = Depends(get_vault)) -> NoteOut:
        try:
            validate_slug(payload.slug)
        except InvalidSlugError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        ts = now_utc()
        note = Note(
            slug=payload.slug,
            title=payload.title,
            created=ts,
            updated=ts,
            tags=list(payload.tags),
            body=payload.body,
        )
        try:
            v.create(note)
        except NoteExistsError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        _reindex_one(note)
        bus.publish(Event(type="node.created", payload=_node_payload(note)))
        return _to_out(note)

    @app.get("/notes/{slug}", response_model=NoteOut)
    def read_note(slug: str, v: Vault = Depends(get_vault)) -> NoteOut:
        try:
            return _to_out(v.read(slug))
        except NoteNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except InvalidSlugError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.put("/notes/{slug}", response_model=NoteOut)
    def update_note(
        slug: str, payload: NoteUpdate, v: Vault = Depends(get_vault)
    ) -> NoteOut:
        try:
            note = v.update(slug, title=payload.title, body=payload.body, tags=payload.tags)
        except NoteNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except InvalidSlugError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        _reindex_one(note)
        bus.publish(Event(type="node.updated", payload=_node_payload(note)))
        return _to_out(note)

    @app.delete("/notes/{slug}", status_code=204)
    def delete_note(slug: str, v: Vault = Depends(get_vault)) -> None:
        try:
            v.delete(slug)
        except NoteNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except InvalidSlugError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        memory.index.delete(slug)
        bus.publish(Event(type="node.deleted", payload={"id": slug}))

    @app.get("/search", response_model=list[NoteSummaryOut])
    def search_endpoint(
        q: str = Query(default=""),
        tag: Optional[str] = None,
        mode: str = Query(default="substring"),
        kind: Optional[str] = None,
        rerank: bool = False,
        v: Vault = Depends(get_vault),
    ) -> list[NoteSummaryOut]:
        if mode == "substring":
            results = search_vault(v, q, tag=tag)
            if kind:
                results = [s for s in results if s.kind == kind]
            return [
                NoteSummaryOut(slug=s.slug, title=s.title, tags=s.tags, updated=s.updated)
                for s in results
            ]
        # mode in {semantic, hybrid}
        hits = memory.archival(q, kind=kind) if mode == "hybrid" else memory.archival(q, kind=kind)
        if mode == "semantic":
            search_results = memory.index.search(q, mode="semantic", kind=kind)
            slugs = [r.slug for r in search_results]
        else:
            slugs = [h.slug for h in hits]
        out: list[NoteSummaryOut] = []
        for slug in slugs:
            try:
                note = v.read(slug)
            except NoteNotFoundError:
                continue
            if tag and tag not in note.tags:
                continue
            out.append(
                NoteSummaryOut(
                    slug=note.slug,
                    title=note.title,
                    tags=list(note.tags),
                    updated=note.updated,
                )
            )
        return out

    @app.get("/graph.json")
    def graph_json(v: Vault = Depends(get_vault)) -> dict:
        return v.graph()

    @app.get("/graph", response_class=HTMLResponse)
    def graph_page() -> str:
        return GRAPH_HTML

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return GRAPH_HTML

    # Skill registry — discover built-ins on startup, expose over HTTP.
    from brain.skill import discover_builtins, discover_user, get_registry

    discover_builtins()
    discover_user(vault_path)

    @app.get("/skills")
    def list_skills_endpoint() -> dict:
        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "schema": s.schema,
                    "trusted": s.trusted,
                }
                for s in get_registry().list_skills()
            ]
        }

    @app.post("/skills/{name}")
    def invoke_skill_endpoint(name: str, payload: dict = None) -> dict:
        try:
            result = get_registry().invoke(name, **(payload or {}))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except TypeError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return {"result": result}

    @app.post("/reindex")
    def reindex_endpoint(m: Memory = Depends(get_memory)) -> dict:
        n = m.reindex_from_vault()
        try:
            m.index.save()
        except Exception:
            pass
        return {"indexed": n}

    @app.post("/train")
    def train_endpoint(epochs: int = 1) -> dict:
        from brain.loops.train_brain import BrainTrainer

        trainer = BrainTrainer(memory)
        summaries = trainer.train_full(epochs=epochs)
        # Push fresh xy positions to all connected clients.
        slugs, _ = memory.index.vectors_matrix()
        moved = []
        for slug in slugs:
            try:
                note = vault.read(slug)
            except NoteNotFoundError:
                continue
            if note.xy is not None:
                moved.append({"id": slug, "xy": list(note.xy)})
        bus.publish(Event(type="training.epoch", payload={
            "summaries": [
                {
                    "avg_recon": s.avg_recon,
                    "avg_link": s.avg_link,
                    "n_pairs": s.n_pairs,
                    "n_docs": s.n_docs,
                }
                for s in summaries
            ],
            "positions": moved,
        }))
        # Surface new edge proposals.
        proposals_path = vault_path / ".brain" / "edge_proposals.jsonl"
        if proposals_path.exists():
            import json as _json
            props = []
            for line in proposals_path.read_text(encoding="utf-8").splitlines():
                try:
                    props.append(_json.loads(line))
                except Exception:
                    continue
            bus.publish(Event(type="proposals.updated", payload={"proposals": props}))
        return {
            "epochs": [
                {
                    "avg_recon": s.avg_recon,
                    "avg_link": s.avg_link,
                    "n_pairs": s.n_pairs,
                    "n_docs": s.n_docs,
                }
                for s in summaries
            ]
        }

    @app.websocket("/graph/ws")
    async def graph_ws(ws: WebSocket) -> None:
        await ws.accept()
        q = bus.subscribe()
        try:
            # Send initial graph snapshot so clients pick up cold.
            await ws.send_json({"type": "graph.snapshot", "payload": vault.graph()})
            while True:
                event: Event = await q.get()
                await ws.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            bus.unsubscribe(q)

    @app.get("/training_history")
    def training_history_endpoint(tail: int = 200) -> dict:
        from brain.nn import read_history

        return {"history": read_history(vault_path / ".brain" / "training_history.jsonl", tail=tail)}

    @app.get("/edge_proposals")
    def edge_proposals_endpoint() -> dict:
        path = vault_path / ".brain" / "edge_proposals.jsonl"
        if not path.exists():
            return {"proposals": []}
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(__import__("json").loads(line))
            except Exception:
                continue
        return {"proposals": out}

    @app.get("/notes/{slug}/links", response_model=LinksOut)
    def links_endpoint(slug: str, v: Vault = Depends(get_vault)) -> LinksOut:
        try:
            return LinksOut(
                outgoing=v.outgoing_links(slug), incoming=v.incoming_links(slug)
            )
        except NoteNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except InvalidSlugError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app
