#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a SHA-256 manifest for a package directory.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("manifests/SHA256SUMS.txt"))
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    return parser.parse_args()


def file_hash(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() != output.resolve()
    )
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(root).as_posix()
            handle.write(f"{file_hash(path, args.chunk_size)}  {relative}\n")
            if index % 25 == 0 or index == len(files):
                print(f"[manifest] {index}/{len(files)} files", flush=True)
    print(output)


if __name__ == "__main__":
    main()
