#!/usr/bin/env python3
"""Materialize a server-matched image subset from an existing dataset tree."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True,help='Prepared source tree with the dataset directory names documented in docs/DATASETS.md.')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--dataset',required=True,help='Dataset directory name, for example imagenet, sun397, or UCF101.')
    parser.add_argument('--dry-run',action='store_true',help='Verify source images without copying them.')
    args=parser.parse_args()
    source=args.source_root.resolve(); target=args.output_root.resolve()
    if source==target or source.is_relative_to(target) or target.is_relative_to(source):
        raise SystemExit('Source and output roots must be separate non-nested directories')
    checked=0
    with gzip.open(ROOT/'datasets/manifests/reference_data_files.csv.gz','rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            rel=Path(row['path'])
            if rel.parts[0]!=args.dataset or rel.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}: continue
            src=(source/rel).resolve(); dst=(target/rel).resolve()
            if not src.is_relative_to(source) or not dst.is_relative_to(target): raise SystemExit('Unsafe manifest path')
            if not src.is_file() or src.stat().st_size!=int(row['bytes']) or digest(src)!=row['sha256']:
                raise SystemExit(f'Source image is missing or differs from the reference: {rel.as_posix()}')
            if not args.dry_run:
                if dst.exists():
                    if not dst.is_file() or digest(dst)!=row['sha256']:
                        raise SystemExit(f'Existing output differs from the reference: {rel.as_posix()}')
                else:
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(src,dst)
            checked+=1
    if not checked:raise SystemExit('No image files matched the requested dataset')
    print(f'Reference images '+('verified' if args.dry_run else 'prepared')+f': {checked} files')

if __name__=='__main__':main()
