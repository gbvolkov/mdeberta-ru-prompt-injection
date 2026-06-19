#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SHA256SUMS.txt.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("manifests/SHA256SUMS.txt"))
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
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    failures: list[str] = []
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, line in enumerate(lines, start=1):
        expected, relative = line.split("  ", 1)
        path = root / Path(relative)
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif file_hash(path, args.chunk_size) != expected:
            failures.append(f"hash_mismatch:{relative}")
        if index % 25 == 0 or index == len(lines):
            print(f"[verify] {index}/{len(lines)} files failures={len(failures)}", flush=True)
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"PASS: {len(lines)} files")


if __name__ == "__main__":
    main()
