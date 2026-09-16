"""Import shim for the Project 2 extraction harness.

Project 2 (`offline-slm-assistant`) was written as a benchmark study, not a
library: its modules live under a plain `src/` package with relative imports and
no packaging metadata. Rather than fork or vendor that code -- which would let
the two copies drift and quietly invalidate Project 2's published report -- it is
imported by path from the sibling checkout.

That is a deliberate trade-off, and it has a real cost worth stating: this
service is coupled to another repo's internal layout. The alternative was
copying `structured.py`, `schemas.py`, and the 52 gold tasks into this repo,
which is worse -- the whole point of the regression gate is that it grades
against *the same* labels Project 2 published.

Everything this service is allowed to touch from Project 2 is re-exported here,
so the coupling is one file wide.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from .settings import SLM_REPO


class SLMRepoMissing(RuntimeError):
    pass


def _load() -> ModuleType:
    root = SLM_REPO
    if not (root / "src" / "structured.py").exists():
        raise SLMRepoMissing(
            f"Project 2 checkout not found at {root}. Clone offline-slm-assistant "
            f"next to this repo, or set SLM_REPO=/path/to/offline-slm-assistant."
        )
    if str(root) not in sys.path:
        # Appended, not prepended: this repo's own modules must always win a
        # name collision. Project 2 exposes a package literally named `src`,
        # which is exactly the kind of name that collides by accident.
        sys.path.append(str(root))
    return importlib.import_module("src")


_load()

_structured = importlib.import_module("src.structured")
_schemas = importlib.import_module("src.schemas")
_tasks = importlib.import_module("src.tasks")
_client = importlib.import_module("src.ollama_client")

# --- generation ---
OllamaClient = _client.OllamaClient
GenerationResult = _client.GenerationResult
StructuredGenerator = _structured.StructuredGenerator
StructuredOutcome = _structured.StructuredOutcome
StructuredAttempt = _structured.StructuredAttempt
build_system_prompt = _structured.build_system_prompt

# --- schemas and scoring ---
SCHEMAS = _schemas.SCHEMAS
score_record = _schemas.score_record
score_field = _schemas.score_field
UNSCORED_FIELDS = _schemas.UNSCORED_FIELDS

# --- the graded task suite (reused as this project's golden dataset) ---
ALL_TASKS = _tasks.ALL_TASKS
Task = _tasks.Task
REFERENCE_DATE = _tasks.REFERENCE_DATE

TASKS_BY_ID: dict[str, "Task"] = {t.id: t for t in ALL_TASKS}

__all__ = [
    "OllamaClient", "GenerationResult", "StructuredGenerator", "StructuredOutcome",
    "StructuredAttempt", "build_system_prompt", "SCHEMAS", "score_record",
    "score_field", "UNSCORED_FIELDS", "ALL_TASKS", "Task", "TASKS_BY_ID",
    "REFERENCE_DATE", "SLMRepoMissing",
]
