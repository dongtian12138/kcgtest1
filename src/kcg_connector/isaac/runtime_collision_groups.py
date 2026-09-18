"""Compile explicit leaf-shape exclusion pairs into equivalent USD groups.

Only enabled collision leaves without rigid-body/articulation APIs participate.
Actor, articulation, disabled-shape and all other pair relationships remain
unchanged. A complete pair matrix checks the generated group rules before any
original relationship is edited. Nothing is saved to source asset files.
"""
from collections import defaultdict


def _parsed_ownership(stage, workdir):
    """Read ownership with the existing planner's USD, isolated from Kit's USD.

    The bundled Kit 25.11 parser crashed on this composed stage before physics
    started. The project's existing standalone USD 26.8 parser accepts it.
    The child reads a composed USD snapshot and returns only ownership data.
    """
    import json
    import os
    from pathlib import Path
    import subprocess
    from pxr import Usd, UsdPhysics

    if Usd.GetVersion() < (0, 26, 8):
        if workdir is None:
            raise ValueError('Kit USD ownership parsing requires an isolated parser work directory')
        folder = Path(workdir)
        folder.mkdir(parents=True, exist_ok=False)
        snapshot, result = folder/'composed_scene.usdc', folder/'ownership.json'
        stage.Export(str(snapshot))
        parser = os.environ.get('KCG_PLANNER_PYTHON')
        if not parser:
            parser = str(Path(__file__).resolve().parents[3]/'.venv/bin/python')
        code = '''from pxr import Usd,UsdPhysics
import json,sys
s=Usd.Stage.Open(sys.argv[1]);parsed=UsdPhysics.LoadUsdPhysicsFromRange(s,['/'])
r={'usd_version':list(Usd.GetVersion()),'shapes':[],'bodies':{},'articulations':{}}
for _,descs in parsed.values():
 for d in descs:
  if hasattr(d,'collisionEnabled') and hasattr(d,'rigidBody'):
   r['shapes'].append({'path':str(d.primPath),'valid':bool(d.isValid),'enabled':bool(d.collisionEnabled)})
  elif hasattr(d,'rigidBodyEnabled'):r['bodies'][str(d.primPath)]=list(map(str,d.collisions))
  elif hasattr(d,'articulatedBodies'):r['articulations'][str(d.primPath)]=list(map(str,d.articulatedBodies))
with open(sys.argv[2],'w') as f:json.dump(r,f)
'''
        env = {k:v for k,v in os.environ.items()
               if k not in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD')}
        run = subprocess.run([parser,'-I','-c',code,str(snapshot),str(result)],
                             env=env,capture_output=True,text=True,timeout=90)
        (folder/'parser.log').write_text(run.stdout+run.stderr)
        run.check_returncode()
        return json.loads(result.read_text())
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ['/'])
    result = {'usd_version': list(Usd.GetVersion()), 'shapes': [], 'bodies': {}, 'articulations': {}}
    for _, descriptors in parsed.values():
        for desc in descriptors:
            if hasattr(desc, 'collisionEnabled') and hasattr(desc, 'rigidBody'):
                result['shapes'].append({'path': str(desc.primPath),
                                        'valid': bool(desc.isValid), 'enabled': bool(desc.collisionEnabled)})
            elif hasattr(desc, 'rigidBodyEnabled'):
                result['bodies'][str(desc.primPath)] = list(map(str,desc.collisions))
            elif hasattr(desc, 'articulatedBodies'):
                result['articulations'][str(desc.primPath)] = list(map(str,desc.articulatedBodies))
    return result


def compile_leaf_collision_pairs(stage, *, include_actor_pairs=False, ownership_parser_workdir=None):
    import numpy as np
    from pxr import Sdf, Usd, UsdPhysics

    if any(p.IsA(UsdPhysics.CollisionGroup) for p in stage.Traverse()):
        raise ValueError('This exact compiler requires a scene without existing collision groups')
    parent = '/World/PerformanceCollisionGroups'
    if stage.GetPrimAtPath(parent):
        raise ValueError('Runtime collision group namespace already exists')
    prims = [p for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.CollisionAPI)
             and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
             and (include_actor_pairs or (
                 not p.HasAPI(UsdPhysics.RigidBodyAPI)
                 and not p.HasAPI(UsdPhysics.ArticulationRootAPI)
                 and not p.GetChildren()))]
    paths = [str(p.GetPath()) for p in prims]
    index = {path: i for i, path in enumerate(paths)}
    size = len(paths)
    excluded = np.zeros((size, size), dtype=bool)
    relations = []
    entries = {path: {i} for i, path in enumerate(paths)}
    if include_actor_pairs:
        # Match PhysX's collectFilteredObjects: a body refers to its actual
        # owned shapes, an articulation to all of its links' owned shapes.
        # Use USD's parser rather than assuming ownership from path prefixes.
        parsed = _parsed_ownership(stage, ownership_parser_workdir)
        bodies, parsed_shapes = {}, set()
        for desc in parsed['shapes']:
            path = desc['path']
            if path in index:
                if not desc['valid'] or not desc['enabled']:
                    raise RuntimeError('USD shape parsing did not preserve enabled geometry')
                parsed_shapes.add(path)
        for path, collisions in parsed['bodies'].items():
            bodies[path] = {index[p] for p in collisions if p in index}
        if parsed_shapes != set(paths):
            raise RuntimeError('Collision filter compilation requires every enabled shape to be parsed')
        for path, members in bodies.items():
            entries.setdefault(path, set()).update(members)
        for path, links in parsed['articulations'].items():
            members = set().union(*(bodies.get(p, set()) for p in links))
            entries.setdefault(path, set()).update(members)

    def resolve_target(path):
        direct = entries.get(str(path))
        if direct is not None:
            return direct
        if not include_actor_pairs:
            return set()
        # PhysX's collectFilteredPairs traverses a target scope if it has no
        # object, pruning at the first body/shape. This also handles an old
        # disabled parent mesh that still contains enabled collision children.
        prim = stage.GetPrimAtPath(path)
        pending = list(prim.GetChildren()) if prim else []
        found = set()
        while pending:
            child = pending.pop()
            key = str(child.GetPath())
            if key in index or key in bodies:
                found.update(entries[key])
            else:
                pending.extend(child.GetChildren())
        return found

    source_prims = list(stage.Traverse()) if include_actor_pairs else prims
    for prim in source_prims:
        rel = prim.GetRelationship('physics:filteredPairs')
        if not rel:
            continue
        source = entries.get(str(prim.GetPath()), set())
        if not source:
            continue
        original = rel.GetTargets()
        remaining = []
        for target in original:
            target_members = resolve_target(target)
            if not target_members and not include_actor_pairs:
                remaining.append(target)
            else:
                for i in source:
                    for j in target_members:
                        excluded[i, j] = excluded[j, i] = True
        if remaining != original:
            relations.append((rel, original, remaining))
    # A shape never collides with itself. Treat its diagonal as a don't-care
    # when finding equal neighborhoods; all distinct pairs are checked below.
    np.fill_diagonal(excluded, False)
    independent, cliques = defaultdict(list), defaultdict(list)
    for i, row in enumerate(excluded):
        independent[row.tobytes()].append(i)
        row = row.copy()
        row[i] = True
        cliques[row.tobytes()].append(i)
    groups, assigned = [], set()
    for i, row in enumerate(excluded):
        if i in assigned:
            continue
        diagonal = row.copy()
        diagonal[i] = True
        a = [j for j in independent[row.tobytes()] if j not in assigned]
        b = [j for j in cliques[diagonal.tobytes()] if j not in assigned]
        members = a if len(a) > len(b) else b
        groups.append(members)
        assigned.update(members)
    ids = np.empty(size, dtype=np.int32)
    for group, members in enumerate(groups):
        ids[members] = group
    rules = np.zeros((len(groups), len(groups)), dtype=bool)
    for i, a in enumerate(groups):
        for j, b in enumerate(groups):
            if i != j:
                rules[i, j] = excluded[a[0], b[0]]
            elif len(a) > 1:
                rules[i, j] = excluded[a[0], a[1]]
    represented = rules[ids[:, None], ids[None, :]]
    np.fill_diagonal(represented, False)
    if not np.array_equal(represented, excluded):
        raise RuntimeError('Collision group partition changes a shape-pair exclusion')
    group_paths = [Sdf.Path(f'{parent}/G{i:03d}') for i in range(len(groups))]
    for i, members in enumerate(groups):
        group = UsdPhysics.CollisionGroup.Define(stage, group_paths[i])
        collection = group.GetCollidersCollectionAPI()
        collection.CreateExpansionRuleAttr().Set(Usd.Tokens.explicitOnly)
        collection.CreateIncludesRel().SetTargets([Sdf.Path(paths[j]) for j in members])
        group.CreateFilteredGroupsRel().SetTargets(
            [path for j, path in enumerate(group_paths) if rules[i, j]])
    # Use USD's own filter-table implementation as a separate authoring check.
    table = UsdPhysics.CollisionGroup.ComputeCollisionGroupTable(stage)
    for i, first in enumerate(group_paths):
        for j, second in enumerate(group_paths):
            if table.IsCollisionEnabled(first, second) == bool(rules[i, j]):
                raise RuntimeError('USD collision group table differs from the original pair graph')
    for rel, original, remaining in relations:
        if rel.GetTargets() != original:
            raise RuntimeError('Pair relationships changed while the compiler was running')
        rel.SetTargets(remaining)
        if rel.GetTargets() != remaining:
            raise RuntimeError('Original uncompiled relationships were not retained')
    return {'scope': 'EXACT_RUNTIME_PAIR_FILTER_COMPILATION',
            'include_actor_pairs': include_actor_pairs,
            'enabled_leaf_count': size, 'group_count': len(groups),
            'distinct_excluded_shape_pairs': int(np.count_nonzero(excluded)//2),
            'replaced_directed_relationship_targets': sum(len(a)-len(b) for _, a, b in relations),
            'all_distinct_leaf_pairs_checked': size*(size-1)//2,
            'pair_graph_mismatches': 0, 'usd_group_table_checked': True,
            'actor_pairs': 'EXPANDED_USING_USD_SHAPE_OWNERSHIP' if include_actor_pairs else 'RETAINED',
            'target_scope_resolution': 'PHYSX_DESCENDANT_TRAVERSAL_PRUNED_AT_BODY_OR_SHAPE',
            'geometry_material_mass_pose_or_drive_changes': False,
            'source_asset_saved': False,
            'groups': [{'path': str(group_paths[i]), 'colliders': [paths[j] for j in members],
                        'excluded_groups': [str(p) for j,p in enumerate(group_paths) if rules[i,j]]}
                       for i,members in enumerate(groups)]}
