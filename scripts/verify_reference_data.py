#!/usr/bin/env python3
"""Compare a local dataset tree with the server reference file manifest."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'datasets/manifests/reference_data_files.csv.gz'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'data')
    parser.add_argument('--dataset',help='Restrict verification to one directory name, for example imagenet or UCF101.')
    parser.add_argument('--sizes-only',action='store_true',help='Compare paths and sizes without hashing image contents.')
    parser.add_argument('--include-metadata',action='store_true',help='Also verify original annotations, preparation records, and cached metadata from the server snapshot.')
    args=parser.parse_args()
    root=args.root.resolve(); checked=0; failures=[]
    with gzip.open(MANIFEST,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if args.dataset and row['path'].split('/',1)[0]!=args.dataset: continue
            if not args.include_metadata and Path(row['path']).suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}: continue
            path=(root/row['path']).resolve()
            if not path.is_relative_to(root): raise SystemExit('Unsafe manifest path')
            checked+=1
            if not path.is_file(): failures.append(f"Missing: {row['path']}");continue
            if path.stat().st_size!=int(row['bytes']): failures.append(f"Size differs: {row['path']}");continue
            if not args.sizes_only:
                h=hashlib.sha256()
                with path.open('rb') as image:
                    for block in iter(lambda:image.read(1024*1024),b''):h.update(block)
                if h.hexdigest()!=row['sha256']:failures.append(f"SHA-256 differs: {row['path']}")
    if not checked: raise SystemExit('No manifest files matched the requested dataset')
    if failures:
        raise SystemExit(f'{len(failures)} reference-file mismatches among {checked} files.\n'+'\n'.join(failures[:25]))
    print(f'Reference dataset verification passed: {checked} files; '+('paths and sizes' if args.sizes_only else 'paths, sizes, and SHA-256'))

if __name__=='__main__':main()
