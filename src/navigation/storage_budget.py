"""Outer execution resource guard; never imported by the camera inference policy."""

import json
import shutil
import threading
import time
from pathlib import Path

from src.navigation.runtime import RunLock, atomic_json

ROOT = Path(__file__).resolve().parents[2]
BUDGET = ROOT / "outputs/navigation_compact_execution_20261020/storage_guard"


def read(path):
    return json.loads(Path(path).read_text())


def require_budget(budget=BUDGET):
    budget = Path(budget)
    state = read(budget / "progress.json")
    if state["status"] != "running" or time.time() - state["updated_at"] > 20:
        raise InterruptedError("A live approved storage guardian is required")
    if shutil.disk_usage(ROOT).free < state["effective_floor_bytes"] + 64 * 1024 * 1024:
        raise InterruptedError("Approved storage cap/reserve reached")


def register_output(root, budget=BUDGET):
    root, budget = Path(root).resolve(), Path(budget).resolve()
    if not root.is_relative_to(ROOT / "outputs") or root == ROOT / "outputs":
        raise ValueError("Only owned project output roots can receive budget STOP files")
    require_budget(budget)
    with RunLock(budget / ".registration.lock"):
        registry = read(budget / "watched_roots.json")
        registry["roots"] = sorted(set(registry["roots"]) | {str(root)})
        registry["updated_at"] = time.time()
        atomic_json(budget / "watched_roots.json", registry)


class BudgetWatch:
    """Keep STOP protection active even if the separate guardian stops updating."""

    def __init__(self, root, budget=BUDGET, poll_seconds=5):
        self.root, self.budget = Path(root).resolve(), Path(budget).resolve()
        self.poll_seconds = poll_seconds
        self.done = threading.Event()
        self.thread = None

    def monitor(self):
        while not self.done.wait(self.poll_seconds):
            try:
                require_budget(self.budget)
            except Exception as error:
                (self.root / "STOP").touch()
                atomic_json(self.root / "storage_stop.json", dict(
                    reason=str(error), mechanism="Owned queue STOP file only",
                    guardian=str(self.budget), occurred_at=time.time(),
                    every_artifact_retained=True,
                ))
                return

    def __enter__(self):
        if not self.root.is_dir():
            raise FileNotFoundError("Create the owned result root before registering its budget")
        register_output(self.root, self.budget)
        self.thread = threading.Thread(
            target=self.monitor, name="navigation-storage-watch", daemon=True
        )
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=self.poll_seconds + 2)
