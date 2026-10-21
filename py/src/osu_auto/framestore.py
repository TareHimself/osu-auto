"""Compressed frame store: zstd blocks of consecutive frames, indexed in meta so any range can be read without touching the rest."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zstandard

FILE = "frames.zst"
LEVEL = 3


class FrameWriter:
    def __init__(self, path: Path, height: int, width: int):
        self.path = path
        self.hw = (height, width)
        self.file = open(path, "wb")
        self.blocks: list[list[int]] = []
        self.offset = 0
        self.cctx = zstandard.ZstdCompressor(level=LEVEL)

    def append(self, frames: np.ndarray) -> None:
        data = self.cctx.compress(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
        self.file.write(data)
        self.blocks.append([self.offset, len(data), int(len(frames))])
        self.offset += len(data)

    def close(self) -> dict:
        self.file.close()
        return {"path": self.path.name, "codec": "zstd", "blocks": self.blocks}


def _decode(path: Path, block: list[int], hw: tuple[int, int]) -> np.ndarray:
    offset, length, count = block
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(length)
    out = zstandard.ZstdDecompressor().decompress(data, max_output_size=count * hw[0] * hw[1])
    return np.frombuffer(out, dtype=np.uint8).reshape(count, *hw)


def read_all(path: Path, meta: dict, into: np.ndarray | None = None, threads: int = 4) -> np.ndarray:
    """Decode every block, in parallel, into `into` (count, h, w) or a fresh array."""
    hw = (int(meta["height"]), int(meta["width"]))
    blocks = meta["blocks"]
    total = sum(b[2] for b in blocks)
    out = np.empty((total, *hw), dtype=np.uint8) if into is None else into
    starts = np.cumsum([0] + [b[2] for b in blocks[:-1]])

    def fill(k: int) -> None:
        out[starts[k]:starts[k] + blocks[k][2]] = _decode(path, blocks[k], hw)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(fill, range(len(blocks))))
    return out


def read_range(path: Path, meta: dict, start: int, stop: int) -> np.ndarray:
    hw = (int(meta["height"]), int(meta["width"]))
    pieces = []
    base = 0
    for block in meta["blocks"]:
        count = block[2]
        if base + count > start and base < stop:
            dec = _decode(path, block, hw)
            pieces.append(dec[max(start - base, 0):min(stop - base, count)])
        base += count
        if base >= stop:
            break
    return np.concatenate(pieces) if pieces else np.empty((0, *hw), dtype=np.uint8)


def frames_meta(export_dir: Path) -> dict:
    return json.loads((export_dir / "meta.json").read_text())["frames"]
