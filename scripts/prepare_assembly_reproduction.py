#!/usr/bin/env python3
"""Fetch the versioned assembly assets and the pinned, patched SAM-6D runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'reproducibility/assembly_20260916'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and digest(path) == expected:
        return
    temporary = path.with_name(path.name + '.download')
    print(f'Downloading {path.name}', flush=True)
    if 'drive.google.com' in url:
        subprocess.run([sys.executable, '-m', 'gdown', '--fuzzy', url, '-O', str(temporary)], check=True)
    else:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output, 8 * 1024 * 1024)
    if digest(temporary) != expected:
        raise ValueError(f'SHA256 mismatch: {temporary}; retained for diagnosis')
    if path.exists():
        raise ValueError(f'Refusing to replace different existing file: {path}')
    temporary.rename(path)


def assets(archive_path):
    manifest = json.loads((META / 'runtime_assets.json').read_text())
    path = archive_path or ROOT / '.deps/downloads' / manifest['archive_name']
    if archive_path is None:
        url = ('https://github.com/dongtian12138/kcgtest1/releases/download/'
               + manifest['release_tag'] + '/' + manifest['archive_name'])
        download(url, path, manifest['archive_sha256'])
    if digest(path) != manifest['archive_sha256']:
        raise ValueError('Runtime archive SHA256 mismatch')
    expected = {row['path']: row for row in manifest['files']}
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        if {member.name for member in members} != set(expected) or len(members) != len(expected):
            raise ValueError('Archive members differ from the versioned manifest')
        for member in members:
            destination = ROOT / member.name
            if not member.isfile() or not destination.resolve().is_relative_to(ROOT):
                raise ValueError(f'Invalid archive member: {member.name}')
            row = expected[member.name]
            if destination.exists():
                if digest(destination) != row['sha256']:
                    raise ValueError(f'Refusing to overwrite changed input: {destination}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open('xb') as output:
                shutil.copyfileobj(source, output)
            if digest(destination) != row['sha256']:
                raise ValueError(f'Extracted file differs: {destination}')
    print(f'Runtime assets verified: {len(expected)} files')
    supplement = ROOT / 'reproducibility/four_camera_baseline_20260920/extra_runtime_inputs.json'
    if supplement.is_file():
        rows = json.loads(supplement.read_text())['files']
        for row in rows:
            source = ROOT / row['path']
            if not source.is_file() or digest(source) != row['sha256']:
                raise ValueError(f'Missing or changed Git-tracked high-camera input: {source}')
        print(f'High-camera inputs verified from Git: {len(rows)} files')


def sam6d(reuse):
    manifest = json.loads((META / 'sam6d.json').read_text())
    checkout = ROOT / '.deps/SAM-6D'
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not checkout.exists():
        subprocess.run(['git', 'clone', '--no-hardlinks', str(reuse) if reuse else manifest['upstream'], str(checkout)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', manifest['commit']], check=True)
        subprocess.run(['git', '-C', str(checkout), 'remote', 'set-url', 'origin', manifest['upstream']], check=True)
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if head != manifest['commit']:
        raise ValueError('Existing SAM-6D checkout is not the pinned commit')
    patch = META / manifest['patch']
    reversed_check = subprocess.run(['git', '-C', str(checkout), 'apply', '--reverse', '--check', str(patch)], capture_output=True)
    if reversed_check.returncode:
        subprocess.run(['git', '-C', str(checkout), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'apply', str(patch)], check=True)
    runtime = checkout / 'SAM-6D'
    for row in manifest['weights']:
        destination = runtime / row['path']
        source = reuse / 'SAM-6D' / row['path'] if reuse else None
        if not destination.exists() and source is not None and source.is_file():
            if digest(source) != row['sha256']:
                raise ValueError(f'Cached weight differs: {source}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        download(row['url'], destination, row['sha256'])
    print('SAM-6D pinned source, inference patch and all four weights verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', action='store_true')
    parser.add_argument('--archive', type=Path, help='Use an already downloaded release archive')
    parser.add_argument('--sam6d', action='store_true')
    parser.add_argument('--reuse-sam6d-cache', type=Path, help='Optional local source checkout and weights, still hash-checked')
    args = parser.parse_args()
    if not (args.assets or args.sam6d):
        parser.error('Choose --assets and/or --sam6d')
    if args.assets:
        assets(args.archive)
    if args.sam6d:
        sam6d(args.reuse_sam6d_cache)


if __name__ == '__main__':
    main()
