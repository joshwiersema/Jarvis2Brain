"""Brain daemon — sleep, cron, and event loops in a single supervised process.

The daemon's job is to *quietly* keep the brain healthy: train at idle, classify
inbox notes once enough signal exists, watch the vault for outside edits, and
fire scheduled skills.

Threads (each a tight loop with a stop event):

- SleepExecutor — every `sleep_interval_s` runs the consolidation pass:
  reindex, train one epoch, auto-classify inbox, write daily reflection.
- CronExecutor — reads `<vault>/.brain/cron.toml`, fires skills at scheduled
  times. Cron format is keep-it-simple: `[every]` blocks with `seconds: N`.
- EventExecutor — polls the vault dir for .md mtimes, reindexes touched
  files. Avoids the watchdog dep — at single-user scale a 5s poll is fine.
"""

from __future__ import annotations

import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from brain.embed import get_embedder, text_for_embedding
from brain.index import IndexedDoc, MemoryIndex
from brain.loops.auto_classify import auto_classify_inbox
from brain.loops.train_brain import BrainTrainer
from brain.memory import Memory
from brain.skill import discover_builtins, get_registry
from brain.vault import Vault


@dataclass
class DaemonStatus:
    running: bool
    threads: list[str] = field(default_factory=list)
    last_sleep_run: Optional[float] = None
    last_cron_run: Optional[float] = None
    last_event_run: Optional[float] = None


@dataclass
class DaemonConfig:
    sleep_interval_s: float = 30 * 60.0  # 30 min
    event_poll_s: float = 5.0
    cron_check_s: float = 5.0
    train_epochs: int = 1


class BrainDaemon:
    def __init__(
        self,
        vault_path: Path,
        config: Optional[DaemonConfig] = None,
    ) -> None:
        self.vault_path = Path(vault_path)
        self.config = config or DaemonConfig()
        self.vault = Vault(self.vault_path)
        self.index = MemoryIndex.load(get_embedder(), self.vault_path / ".brain" / "index.npz")
        self.memory = Memory(self.vault, self.index)
        if not self.memory.index.all_slugs():
            self.memory.reindex_from_vault()
            self.memory.index.save()
        self.trainer = BrainTrainer(self.memory)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.status = DaemonStatus(running=False)

    # ----- lifecycle -----

    def start(self) -> None:
        if self.status.running:
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._sleep_loop, name="sleep", daemon=True),
            threading.Thread(target=self._cron_loop, name="cron", daemon=True),
            threading.Thread(target=self._event_loop, name="event", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self.status = DaemonStatus(running=True, threads=[t.name for t in self._threads])

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self.status = DaemonStatus(running=False)

    # ----- sleep loop -----

    def _sleep_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_sleep_cycle()
            except Exception:
                pass
            self.status.last_sleep_run = time.time()
            self._stop.wait(self.config.sleep_interval_s)

    def run_sleep_cycle(self) -> dict:
        """Public entrypoint so tests + CLI can invoke a single cycle."""
        from brain.loops.constitution import ensure_constitution
        from brain.loops.skill_synthesis import propose_skills, write_proposals

        ensure_constitution(self.vault)
        n = self.memory.reindex_from_vault()
        self.memory.index.save()
        summaries = self.trainer.train_full(epochs=self.config.train_epochs)
        classifications = auto_classify_inbox(self.memory)
        skill_proposals = propose_skills(self.memory)
        n_proposals = write_proposals(self.memory, skill_proposals)
        return {
            "indexed": n,
            "epochs": [s.avg_recon for s in summaries],
            "classified": [
                {"slug": c.slug, "kind": c.chosen_kind, "confidence": c.confidence}
                for c in classifications
            ],
            "skill_proposals": n_proposals,
        }

    # ----- cron loop -----

    def _cron_loop(self) -> None:
        cron_path = self.vault_path / ".brain" / "cron.toml"
        last_fire: dict[str, float] = {}
        while not self._stop.is_set():
            try:
                if cron_path.exists():
                    self._fire_due(cron_path, last_fire)
            except Exception:
                pass
            self.status.last_cron_run = time.time()
            self._stop.wait(self.config.cron_check_s)

    def _fire_due(self, cron_path: Path, last_fire: dict) -> None:
        config = tomllib.loads(cron_path.read_text(encoding="utf-8"))
        every = config.get("every", [])
        if not isinstance(every, list):
            return
        now = time.time()
        for entry in every:
            name = entry.get("name") or entry.get("skill") or "anon"
            seconds = float(entry.get("seconds", 0))
            if seconds <= 0:
                continue
            if now - last_fire.get(name, 0.0) < seconds:
                continue
            skill_name = entry.get("skill")
            args = entry.get("args") or {}
            try:
                discover_builtins()
                get_registry().invoke(skill_name, **args)
            except Exception:
                pass
            last_fire[name] = now

    # ----- event loop -----

    def _event_loop(self) -> None:
        seen: dict[Path, float] = {}
        while not self._stop.is_set():
            try:
                self._scan_vault(seen)
            except Exception:
                pass
            self.status.last_event_run = time.time()
            self._stop.wait(self.config.event_poll_s)

    def _scan_vault(self, seen: dict[Path, float]) -> None:
        for path in self.vault.path.rglob("*.md"):
            try:
                rel = path.relative_to(self.vault.path)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == ".brain":
                continue
            mt = path.stat().st_mtime
            if seen.get(path) == mt:
                continue
            seen[path] = mt
            slug = path.stem
            try:
                note = self.vault.read(slug)
            except Exception:
                continue
            self.memory.index.upsert(
                IndexedDoc(
                    slug=slug,
                    text=text_for_embedding(note.title, note.body, note.tags),
                    kind=note.kind,
                    tags=list(note.tags),
                    importance=note.importance,
                    updated_ts=note.updated.timestamp(),
                )
            )
