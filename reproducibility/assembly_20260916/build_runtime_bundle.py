#!/usr/bin/env python3
"""Maintainer tool: copy audited inputs, changing only repository path bindings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    meta = Path(__file__).resolve().parent
    source = args.source.resolve()
    inventory = json.loads(args.inventory.read_text())
    rows = []
    for item in inventory['repo_files']:
        relative = item['relative_path']
        original = source / relative
        destination = root / relative
        original_sha = sha(original)
        if not relative.startswith('artifacts/'):
            if sha(destination) != original_sha:
                raise ValueError(f'Tracked input differs: {relative}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = original.read_bytes()
        if original.name in ('two_nail_all_reserves_recipe.json',
                             'wrist_all_reserves_no_angle_gate_recipe.json',
                             'selected_two_nail_geometry.json',
                             'geometry_only_manifest.json', 'geometry_manifest.json'):
            data = data.decode().replace(str(source) + '/', '').encode()
        elif original.suffix == '.usda':
            content = data.decode()
            def asset(match):
                target = root / Path(match.group(1)).relative_to(source)
                return '@' + os.path.relpath(target, destination.parent) + '@'
            content = re.sub(r'@(' + re.escape(str(source)) + r'/[^@]+)@', asset, content)
            data = content.encode()
        destination.write_bytes(data)
        rows.append({'path': relative, 'original_sha256': original_sha})
    # The cooked geometry is unchanged. Its manifest binds the relocated USD
    # wrapper, whose only change is a relative reference to the same base layer.
    for row in rows:
        path = root / row['path']
        if path.name == 'geometry_only_manifest.json':
            record = json.loads(path.read_text())
            record['source_robot_asset_sha256'] = sha(root / record['source_robot_asset'])
            path.write_text(json.dumps(record, indent=2) + '\n')
    for row in rows:
        path = root / row['path']
        row.update(sha256=sha(path), bytes=path.stat().st_size)
        row['path_binding_changed'] = row['sha256'] != row['original_sha256']
        if row['path_binding_changed']:
            preserved = meta / 'original_path_metadata' / row['path']
            preserved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row['path'], preserved)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, 'w:gz', compresslevel=3) as archive:
        for row in rows:
            archive.add(root / row['path'], arcname=row['path'], recursive=False)
    report = {'schema': 'assembly_runtime_assets_v1',
              'release_tag': 'assembly-baseline-20260916',
              'archive_name': args.archive.name, 'archive_sha256': sha(args.archive),
              'archive_bytes': args.archive.stat().st_size, 'files': rows,
              'original_source_commit': '30d98391531ad8e2f15d82898bfa8bea02e5b3db',
              'changes': 'Repository path strings, relative USD references, and the matching top-level USD wrapper hash only. Geometry, physics and numerical settings unchanged.'}
    (meta / 'runtime_assets.json').write_text(json.dumps(report, indent=2) + '\n')
    sam = inventory['external_sam6d']
    weights = []
    for item in sam['weights']:
        path = Path(item['path'])
        weights.append({'path': str(path.relative_to(Path(sam['root']))),
                        'bytes': path.stat().st_size, 'sha256': sha(path),
                        'url': item['upstream_download_url_from_local_source']})
    (meta / 'sam6d.json').write_text(json.dumps({
        'upstream': sam['upstream'], 'commit': sam['commit'], 'weights': weights,
        'patch': 'sam6d_inference_local_changes.patch'}, indent=2) + '\n')
    shutil.copy2(sam['local_inference_patch'], meta / 'sam6d_inference_local_changes.patch')
    print(json.dumps({'files': len(rows), 'changed_path_metadata': sum(row['path_binding_changed'] for row in rows),
                      'archive_bytes': args.archive.stat().st_size, 'archive_sha256': report['archive_sha256']}, indent=2))


if __name__ == '__main__':
    main()
