from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

JsonValue = Any


def write_json(path: Path, payload: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list[list[JsonValue]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@dataclass
class Timer:
    start: float
    end: float | None = None

    @property
    def seconds(self) -> float:
        end = self.end if self.end is not None else time.perf_counter()
        return end - self.start


@contextmanager
def timer() -> Iterator[Timer]:
    t = Timer(start=time.perf_counter())
    try:
        yield t
    finally:
        t.end = time.perf_counter()


def track_metrics(
    metrics_path: Path, *, section: str | None = None
) -> Callable[[Callable[..., Mapping[str, Any]]], Callable[..., Mapping[str, Any]]]:
    def decorator(
        func: Callable[..., Mapping[str, Any]],
    ) -> Callable[..., Mapping[str, Any]]:
        def wrapped(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
            result = func(*args, **kwargs)

            existing: dict[str, Any] = {}
            if metrics_path.exists():
                existing = cast(
                    dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8"))
                )

            payload: dict[str, Any]
            if section:
                payload = {**existing, section: dict(result)}
            else:
                payload = {**existing, **dict(result)}

            write_json(metrics_path, payload)
            return result

        return wrapped

    return decorator
