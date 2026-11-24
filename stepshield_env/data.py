"""Load the StepShield training recordings and pair rogue/clean twins."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_DATA = Path.home() / "orca" / "stepshield" / "data" / "train"


@dataclass(frozen=True)
class Run:
    id: str               # shared by both twins, e.g. "SEC-L1-001"
    category: str         # folder name, e.g. "SEC"
    rogue: bool           # from the file name
    task: dict
    steps: list
    n: int                # number of steps
    tau: Optional[int]    # first labeled step (1-indexed); None if unlabeled


def load_run(path):
    path = Path(path)
    pair_id, kind = path.stem.rsplit("-", 1)
    if kind not in ("ROGUE", "CLEAN"):
        raise ValueError(f"unexpected file name: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)  # one JSON object per file, despite .jsonl
    steps = raw["steps"]
    tau = next((i for i, s in enumerate(steps, 1) if s["label"] is not None), None)
    return Run(pair_id, path.parent.name, kind == "ROGUE", raw["task"], steps,
               len(steps), tau)


def load_runs(data_dir=DEFAULT_DATA):
    paths = sorted(Path(data_dir).glob("*/*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl files under {data_dir}")
    return [load_run(p) for p in paths]


def pair_runs(runs, exclude_task_mismatch=False):
    """Return {id: (rogue, clean)}.

    exclude_task_mismatch drops pairs whose task text differs between twins.
    """
    sides = {}
    for r in runs:
        twins = sides.setdefault(r.id, {})
        if r.rogue in twins:
            raise ValueError(f"duplicate run: {r.id} rogue={r.rogue}")
        twins[r.rogue] = r
    pairs = {}
    for pair_id, twins in sorted(sides.items()):
        if len(twins) != 2:
            raise ValueError(f"run has no twin: {pair_id}")
        rogue, clean = twins[True], twins[False]
        if exclude_task_mismatch and rogue.task != clean.task:
            continue
        pairs[pair_id] = (rogue, clean)
    return pairs
