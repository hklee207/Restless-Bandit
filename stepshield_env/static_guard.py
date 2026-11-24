"""Adapter for StepShield's StaticGuard (847 regex rules; standard library only)."""
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

from env import CURRENT_FIELDS

DETECTORS = Path.home() / "orca" / "stepshield" / "benchmark" / "detectors"
SOURCES = ("base.py", "static_guard.py")
CACHE = Path(__file__).resolve().parent / "cache" / "static_guard.json"
_PKG = "_stepshield_detectors"
ALLOWED = frozenset(CURRENT_FIELDS) | {"observation"}


def load_static_guard(detectors=DETECTORS):
    """Load only base.py and static_guard.py.

    Skips the package's __init__.py, which would import LLMJudge (and openai).
    """
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = []  # nothing else can be imported through it
    sys.modules[_PKG] = pkg
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        for source in SOURCES:
            name = f"{_PKG}.{source[:-3]}"
            spec = importlib.util.spec_from_file_location(name, Path(detectors) / source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module.StaticGuard()


class StaticGuardScorer:
    """evaluate_step(step, history) -> (is_rogue, confidence), cached on disk.

    The key is a hash of the visible step and StaticGuard's source, so it carries
    no run id and goes stale by itself if the rules change.
    """

    def __init__(self, cache_path=CACHE, detectors=DETECTORS):
        self._detectors = Path(detectors)
        self._guard = None
        self._source = hashlib.sha256(
            b"\0".join((self._detectors / s).read_bytes() for s in SOURCES)).hexdigest()
        self._path = Path(cache_path) if cache_path else None
        self._cache = {}
        if self._path and self._path.exists():
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        self.computed = 0  # cache misses

    def evaluate_step(self, step, history=()):
        extra = set(step) - ALLOWED
        if extra:
            raise ValueError(f"fields StaticGuard may not see: {sorted(extra)}")
        # A null observation means no result; StaticGuard expects text.
        step = {k: v for k, v in step.items() if v is not None}
        text = json.dumps(step, sort_keys=True)
        key = hashlib.sha256(f"{self._source}\0{text}".encode()).hexdigest()
        if key not in self._cache:
            if self._guard is None:
                self._guard = load_static_guard(self._detectors)
            # StaticGuard ignores both the step index and the context.
            res = asyncio.run(self._guard.detect_step(step, len(history) + 1, {}))
            self._cache[key] = [bool(res.is_rogue), float(res.confidence)]
            self.computed += 1
        return tuple(self._cache[key])

    def save(self):
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        os.replace(tmp, self._path)


def flag_rates(runs, scorer, with_result=False):
    """{group: (flagged steps, steps)}. Analysis only: uses tau to group steps."""
    fields = CURRENT_FIELDS + (("observation",) if with_result else ())
    counts = {g: [0, 0] for g in ("clean", "rogue, before tau", "rogue, at tau",
                                  "rogue, after tau", "rogue, tau onward")}
    for run in runs:
        for t, step in enumerate(run.steps, 1):
            flagged, _ = scorer.evaluate_step({f: step[f] for f in fields})
            if not run.rogue:
                groups = ["clean"]
            elif t < run.tau:
                groups = ["rogue, before tau"]
            else:
                groups = ["rogue, at tau" if t == run.tau else "rogue, after tau",
                          "rogue, tau onward"]
            for g in groups:
                counts[g][0] += flagged
                counts[g][1] += 1
    return {g: tuple(c) for g, c in counts.items()}
