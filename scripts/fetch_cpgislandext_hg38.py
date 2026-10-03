"""Deterministic acquisition of the UCSC hg38 cpgIslandExt track used by biological_validation_v2.

The file is deliberately NOT versioned in git (redistribution policy of UCSC is per-track; the
track is public and stable). This script downloads it and refuses to write it unless the sha256
matches the value frozen in data/derived/bioval_v2/MANIFEST_checksums.json.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz"
SHA256 = "2339f8bad0ec9993211287b319b8d59ddeff30a5010505f649817332ed8222f8"
SIZE = 717984
UPSTREAM_LAST_MODIFIED = "Wed, 19 Oct 2022 06:39:38 GMT"
DEST = Path("data/bio_annotations/cpgIslandExt.hg38.txt.gz")


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    if args.dest.exists() and sha256_bytes(args.dest.read_bytes()) == SHA256:
        print(f"ok: {args.dest} already matches sha256 {SHA256}")
        return 0
    with urllib.request.urlopen(URL, timeout=120) as resp:
        blob = resp.read()
    got = sha256_bytes(blob)
    if got != SHA256 or len(blob) != SIZE:
        print(f"error: upstream changed (sha256 {got}, {len(blob)} bytes); expected {SHA256}, {SIZE}", file=sys.stderr)
        return 1
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    args.dest.write_bytes(blob)
    print(f"wrote {args.dest} ({len(blob)} bytes, sha256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
