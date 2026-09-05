#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import zipfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"stream_betti_curves-{a.version}-{a.label}"
    staging = a.output_dir / stem
    staging.mkdir(exist_ok=True)
    shutil.copy2(a.binary, staging / a.binary.name)
    for name in ('README.md', 'LICENSE', 'CHANGELOG.md'):
        source = Path(name)
        if source.exists():
            shutil.copy2(source, staging / name)
    if 'windows' in a.label:
        archive = a.output_dir / f"{stem}.zip"
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in staging.iterdir():
                zf.write(f, f"{stem}/{f.name}")
    else:
        archive = a.output_dir / f"{stem}.tar.gz"
        with tarfile.open(archive, 'w:gz') as tf:
            tf.add(staging, arcname=stem)
    (archive.with_suffix(archive.suffix + '.sha256')).write_text(f"{sha256(archive)}  {archive.name}\n")
    shutil.rmtree(staging)
    print(archive)


if __name__ == '__main__':
    main()
