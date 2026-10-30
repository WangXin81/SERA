#!/usr/bin/env python3
"""Prepare the reference CLIP checkpoints and archived text-feature caches."""
from __future__ import annotations
import argparse
import ast
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ASSETS = {'clip/ViT-B-16.pt': '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f', 'clip/ViT-B-32.pt': '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af', 'clip/ViT-L-14.pt': 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836', 'clip/text_feats/caltech101_ViT_B_16.pt': 'c503340e00006168ea23b7f08e294ae5087c634d7bc1cd09bc9c4de9d8383dc0', 'clip/text_feats/caltech101_ViT_B_32.pt': 'adef55749a1e7654c78e267c9d00224e661fcbbc5819e9b17d24cd8694276d91', 'clip/text_feats/caltech101_ViT_L_14.pt': 'f9b4d149062ce76018e1d5a65ca5ec6af8aabf2ba683b56213efacc55d83d1be', 'clip/text_feats/dtd_ViT_B_16.pt': 'f5f6449210343814aa9c1334abfdab834d105f8fe395e24a8ae9a60534364345', 'clip/text_feats/dtd_ViT_B_32.pt': '9c4f6dcc79ad1aedfa425b01ec925b3f17f41379280366d7de6a001accbfdb06', 'clip/text_feats/dtd_ViT_L_14.pt': 'b9298ac0fe76d3477740b0f53af09ea991a7c84c2c599f1a9e9ab1267d6a6879', 'clip/text_feats/eurosat_ViT_B_16.pt': '8108014f2d3a941683f6bc4e341237dc65b656b3caad05bbef98b5e08f4eb52f', 'clip/text_feats/eurosat_ViT_B_32.pt': 'c3224b24abd889a6a19abaa04ba5d53780e22dd6443ec0666d827564c2efd915', 'clip/text_feats/eurosat_ViT_L_14.pt': 'f2ce1283c891f314e6eae9c77f161649d10fbe2f2f53445274ca89374cd3a1a3', 'clip/text_feats/fgvc_ViT_B_16.pt': '04a6f8101eab111df875096da761eabebece4951a55be0f34ec2c0599958216f', 'clip/text_feats/fgvc_ViT_B_32.pt': 'fe74c407d1e2f489055389645629ced1e180bde336385a13db9233e8246386c0', 'clip/text_feats/fgvc_ViT_L_14.pt': '3eccb7f6b8e58a5bcde247f32aa9764d1e319b66bb1f6704c8c95fbae6e7561e', 'clip/text_feats/fgvc_aircraft_ViT_B_16.pt': '523448adf95b2d3ae7497464b065f06ada3dd4a2f62fd711596c9c524fe5b41d', 'clip/text_feats/fgvc_aircraft_ViT_B_32.pt': '41a06bde63ceb78dc318ec16bf82d1440ae0844974b32bc0481a4a49fbe6882c', 'clip/text_feats/fgvc_aircraft_ViT_L_14.pt': '2631a908c18fc37ebe1d1316a9b260fcf8deff0a456ec6e9a416b88f5e62bc8e', 'clip/text_feats/food101_ViT_B_16.pt': '94aa24e0b7b07fe90bba10f9c98a41dfdf6a6bf1f976dba64c134ff7d045b805', 'clip/text_feats/food101_ViT_B_32.pt': 'dd0f92035d058609d77ecfa655871ad2e34ea3a306192320003a1413c6a1c228', 'clip/text_feats/food101_ViT_L_14.pt': '5c151a48381acc8b8a1c8e0ab948ed44dbd0e72433a82720506fadf449a203a2', 'clip/text_feats/imagenet_ViT_B_16.pt': '3effd2bfc988edcf6525c3f9e66da129f795d4f7565daa0c53b6893f2087b0d2', 'clip/text_feats/imagenet_ViT_B_32.pt': 'ad61e6c1813e5281d648d48146a22bc291ec5d5ebf2aab3aa979a1326e291d9b', 'clip/text_feats/imagenet_ViT_L_14.pt': '900b4bd51d414ecc9e12116e73956f29ef1278f37d8ac1f987dcee27895f5e17', 'clip/text_feats/oxford_flowers_ViT_B_16.pt': '936a9e3b90f5bc669dc2d1063fba10e74882b4861b240223cf17b673c6b853f3', 'clip/text_feats/oxford_flowers_ViT_B_32.pt': '02e6bb1f7f53c1a6f8161e6baa86c2ee4ac4e818a21539fa7287ede65a2e44fa', 'clip/text_feats/oxford_flowers_ViT_L_14.pt': 'e9fc5668e854066128cb1e131d80101f79ebce7cd5996eec39c7d5637698447d', 'clip/text_feats/oxford_pets_ViT_B_16.pt': '409c1677befe77fadda83b383f27414e732de098838b2a7f9ec743b60f51af85', 'clip/text_feats/oxford_pets_ViT_B_32.pt': '1e86fadd76b6afa92238a46e2ea73a508acbf24b7ca55865e4483e63293c81da', 'clip/text_feats/oxford_pets_ViT_L_14.pt': '9ac86045e8360cdeef191b1e019ffaa25a896adef879816cc91991bd10ad3a54', 'clip/text_feats/stanford_cars_ViT_B_16.pt': '680953a2c96685d7955c7ecf85ae94ee9d82a2aa8525bd066ce4822dc9d92d66', 'clip/text_feats/stanford_cars_ViT_B_32.pt': 'b27617bf2a75d7f367882237791978cd0ef657d08890edcb105e6f1569753f35', 'clip/text_feats/stanford_cars_ViT_L_14.pt': '8882ce33804b7b121f20878d8798afb05173dcf1e2d99229ca9351b0a66932a1', 'clip/text_feats/sun397_ViT_B_16.pt': 'f48f19f3cfbb10b22c0161ebe3e7745440d5e6f7a1cf0a382702e977e9c6e4ce', 'clip/text_feats/sun397_ViT_B_32.pt': '51597ea6a957e610fe08bb88918a3c12d37ae5595a6986b70cd2c1e20398af0c', 'clip/text_feats/sun397_ViT_L_14.pt': '5cd8b0462ac9f073f3a0e8c404bb857678c3e92d003cf2948edf758ca54525dd', 'clip/text_feats/ucf101_ViT_B_16.pt': '30eb0e8f14f8bf0b4f96a3f78b9b8a6a29d80e50872743dc8c5fb98c4c0b40b8', 'clip/text_feats/ucf101_ViT_B_32.pt': 'bf814f564e2cb28e1e0ed50e90503cd47bcb1a26d294d1e04b6d5f4ab60d4c07', 'clip/text_feats/ucf101_ViT_L_14.pt': '145a2cda4b61ffcf1367dff22727c27904e080ec352201b77867139176ab911d', 'clip/text_feats/vocab_free_imagenet_text_ensemble_ViT_B_16.pt': '659a148ac75aaecf95653ee5cbe2879560caf6aee3a54a92860b023626f8f4bf', 'clip/text_feats/vocab_free_imagenet_text_ensemble_ViT_B_32.pt': '912f71a0a23e8e75b2ea4dd9c502338d5ca2292df1a34393f19f375919cce5b6', 'clip/text_feats/vocab_free_imagenet_text_ensemble_ViT_L_14.pt': 'ccd684074330b0aa84f9fe2ee09e36bda28991baea057be82783678f3119e7e8'}

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def model_urls():
    tree=ast.parse((ROOT/'sera/clip_encoder/clip.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MODELS' for t in node.targets):
            urls=ast.literal_eval(node.value)
            return {name:urls[name] for name in ('ViT-B/32','ViT-B/16','ViT-L/14')}
    raise RuntimeError('CLIP checkpoint URLs are absent')

def prepare(destination, download=True):
    destination=destination.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    entries=EXPECTED_ASSETS
    if download:
        for name,url in model_urls().items():
            target=destination/'clip'/url.rsplit('/',1)[1]
            target.parent.mkdir(parents=True,exist_ok=True)
            expected=entries[target.relative_to(destination).as_posix()]
            if target.is_file() and sha256(target)==expected:
                print(f'Checkpoint verified: {name}')
                continue
            if target.exists(): raise SystemExit(f'Existing checkpoint has a different checksum: {target}')
            temporary=target.with_suffix('.pt.download')
            try:
                with urllib.request.urlopen(url,timeout=120) as response, temporary.open('wb') as out:
                    shutil.copyfileobj(response,out)
                if sha256(temporary)!=expected: raise SystemExit(f'Checkpoint checksum mismatch: {name}')
                temporary.replace(target)
            finally:
                if temporary.exists(): temporary.unlink()
            print(f'Checkpoint prepared: {name}')
    expected_caches={name for name in entries if name.startswith('clip/text_feats/')}
    restored_caches=set()
    with tarfile.open(ROOT/'resources/SERA_text_features.tar.gz','r:gz') as archive:
        for item in archive.getmembers():
            if not item.isfile() or item.name not in entries or not item.name.startswith('clip/text_feats/'):
                raise SystemExit(f'Unexpected cache archive entry: {item.name}')
            if item.name in restored_caches:
                raise SystemExit(f'Duplicate cache archive entry: {item.name}')
            target=(destination/item.name).resolve()
            if not target.is_relative_to(destination): raise SystemExit('Unsafe archive path')
            payload=archive.extractfile(item).read()
            expected=entries[item.name]
            if hashlib.sha256(payload).hexdigest()!=expected: raise SystemExit(f'Cache checksum mismatch: {item.name}')
            if target.exists():
                if not target.is_file() or sha256(target)!=expected:
                    raise SystemExit(f'Existing cache differs from the reference: {target}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(payload)
            restored_caches.add(item.name)
    missing_caches=expected_caches-restored_caches
    if missing_caches:
        raise SystemExit('Reference cache archive is incomplete: '+', '.join(sorted(missing_caches)))
    print('Reference text-feature caches verified and prepared: 39 files')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'pretrained')
    parser.add_argument('--text-features-only',action='store_true',help='Restore caches without downloading the three CLIP checkpoints.')
    args=parser.parse_args()
    prepare(args.output_root,not args.text_features_only)

if __name__=='__main__': main()
