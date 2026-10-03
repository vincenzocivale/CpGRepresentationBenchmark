"""Write data/derived/bioval_v2/MANIFEST_checksums.json: sha256 + size of EVERY derived file.

Complements the per-axis FROZEN*/MANIFEST* files (which do not cover every derived table) and records
the external island-track source. Pure bookkeeping: reads bytes, never opens embeddings.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("data/derived/bioval_v2")
OUT = ROOT / "MANIFEST_checksums.json"
SKIP_NAMES = {OUT.name}


def digest(path: Path) -> tuple[str, str, int]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return path.relative_to(ROOT).as_posix(), h.hexdigest(), path.stat().st_size


def main() -> None:
    files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.name not in SKIP_NAMES and not p.name.endswith(".tmp")
    )
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(digest, files))
    manifest = {
        "purpose": "sha256 of every derived bioval_v2 file at protocol freeze v1 (files themselves are gitignored)",
        "produced_before_any_embedding_metric": True,
        "root": ROOT.as_posix(),
        "files": {rel: {"sha256": sha, "bytes": size} for rel, sha, size in rows},
        "external_sources_not_versioned": {
            "data/bio_annotations/cpgIslandExt.hg38.txt.gz": {
                "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz",
                "sha256": "2339f8bad0ec9993211287b319b8d59ddeff30a5010505f649817332ed8222f8",
                "bytes": 717984,
                "upstream_last_modified": "Wed, 19 Oct 2022 06:39:38 GMT",
                "retrieved": "2026-10-02 (re-downloaded 2026-10-03: byte-identical)",
                "acquisition": "python scripts/fetch_cpgislandext_hg38.py (verifies sha256, refuses mismatch)",
                "license_note": "UCSC: data files freely usable; redistribution limits apply only to some third-party tracks "
                                "(https://genome.ucsc.edu/license/). Not committed out of caution.",
                "used_by": ["coordinates.map_genomic_context (8 CpGs without legacy context)", "scripts/build_bioval_v2_covariates.py"],
            }
        },
    }
    OUT.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}: {len(rows)} files")


if __name__ == "__main__":
    main()
