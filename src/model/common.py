from __future__ import annotations

import pickle
from pathlib import Path


def save_pickle(obj: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> object:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
