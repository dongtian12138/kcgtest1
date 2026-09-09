"""Install frozen connector collision/materials BEFORE physics initialization.

No simulator is started here. The caller builds its ordinary Body/Nut/Socket
scene, calls install_model(stage), then creates tensor readers and resets World.
Only connector collision subtrees, private material copies, binding strengths
and its existing passive internal joint are authored. Actor pose/mass/velocity,
robot commands, world solver and laboratory apparatus are not copied.
"""
from pathlib import Path
import copy
import re
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

BODY='/World/TE_J35FreeSplitPlug/Body'
NUT='/World/TE_J35FreeSplitPlug/CouplingNut'
SOCKET='/World/TEVisualHandoff/FixedReceptaclePose'
JOINT='/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute'
MASS_FIELDS=('physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes')


def _assert_before_physics():
    try:
        import omni.timeline
    except ImportError:
        return 'OFFLINE_USD_NO_TIMELINE'
    timeline=omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time()!=0.:
        raise RuntimeError('Frozen connector must be installed before physics starts/reset; no hot replacement')
    try:
        from isaacsim.core.simulation_manager import SimulationManager
        if SimulationManager._physics_sim_view__warp is not None:
            raise RuntimeError('Physics tensor view already exists; install before World.reset()')
    except ImportError:
        pass
    return 'ISAAC_TIMELINE_ZERO_AND_NO_PHYSICS_TENSOR_VIEW'


def _snapshot(stage,paths):
    record={}
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        record[path]={
            'world_transform':str(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
            'pose_mass_velocity':{a.GetName():str(a.Get()) for a in prim.GetAttributes()
                                 if a.GetName().startswith('xformOp') or a.GetName() in MASS_FIELDS+('physics:velocity','physics:angularVelocity')},
        }
    return record


def _under(path,root):
    return path==root or path.startswith(root+'/')


def _physics_values(material):
    return {a.GetName():str(a.Get()) for a in material.GetPrim().GetAttributes()
            if a.HasAuthoredValueOpinion() and a.GetName().startswith(('physics:','physxMaterial:'))}


def install_model(stage, model_path=None, *, body_path=BODY, nut_path=NUT,
                  socket_root_path=SOCKET, passive_joint_path=JOINT,
                  material_root='/World/ConnectorFrozenModelMaterials', prepared=None):
    """Install exactly the frozen model; do not save or reset the caller's stage.

    model_path defaults to connector_model.usdc beside this adapter. Optional
    path arguments relocate the same connector actors without moving them.
    Call AFTER ordinary robot/connector material and geometry setup, but BEFORE
    tensor reader construction/World.reset. State:q/dq remains caller-owned.
    Returns concrete changed paths and invariant checks, not an assembly PASS.
    """
    guard=_assert_before_physics()
    model_path=Path(model_path or Path(__file__).with_name('connector_model.usdc')).resolve()
    if not model_path.is_file():raise FileNotFoundError(model_path)
    frozen=Usd.Stage.Open(str(model_path))
    if not frozen:raise ValueError('Could not open frozen connector USD')
    source_layer=frozen.Flatten();source=Usd.Stage.Open(source_layer)
    target_layer=stage.GetEditTarget().GetLayer()
    roots={BODY:str(body_path),NUT:str(nut_path),SOCKET:str(socket_root_path)}
    if len(set(roots.values()))!=3:raise ValueError('Connector actors must be distinct')
    for src,dst in roots.items():
        if not source.GetPrimAtPath(src) or not stage.GetPrimAtPath(dst):
            raise ValueError(f'Existing source/target connector root missing: {src} -> {dst}')
    for path in (body_path,nut_path):
        prim=stage.GetPrimAtPath(path)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
            raise ValueError('Target Body and Nut must be existing dynamic rigid actors')
    before=_snapshot(stage,tuple(roots.values()))
    for src,dst in ((BODY,body_path),(NUT,nut_path)):
        for name in MASS_FIELDS:
            if str(source.GetPrimAtPath(src).GetAttribute(name).Get())!=str(stage.GetPrimAtPath(dst).GetAttribute(name).Get()):
                raise ValueError(f'Caller mass/inertia differs from frozen model: {dst}.{name}')
    old_joint=stage.GetPrimAtPath(passive_joint_path)
    source_joint=source.GetPrimAtPath(JOINT)
    if not old_joint or not source_joint or not old_joint.IsA(UsdPhysics.RevoluteJoint) or not source_joint.IsA(UsdPhysics.RevoluteJoint):
        raise ValueError('The existing original Body/Nut revolute joint is required')
    src_joint=UsdPhysics.Joint(source_joint)
    if list(map(str,src_joint.GetBody0Rel().GetTargets()))!=[BODY] or list(map(str,src_joint.GetBody1Rel().GetTargets()))!=[NUT]:
        raise ValueError('Frozen internal joint is not the permitted Body/Nut topology')
    old_joint_api=UsdPhysics.Joint(old_joint)
    if list(map(str,old_joint_api.GetBody0Rel().GetTargets()))!=[body_path] or list(map(str,old_joint_api.GetBody1Rel().GetTargets()))!=[nut_path]:
        raise ValueError('Target internal joint does not connect these Body and Nut actors')
    old_joint_schemas=old_joint.GetMetadata('apiSchemas')
    old_joint_tokens=list(old_joint_schemas.GetAddedOrExplicitItems()) if old_joint_schemas else []
    old_state={a.GetName():a.Get() for a in old_joint.GetAttributes() if a.GetName().startswith('state:')}
    old_state_types={a.GetName():a.GetTypeName() for a in old_joint.GetAttributes() if a.GetName().startswith('state:')}
    # A stronger ancestor outside the connector must not be silently changed.
    # Bindings inside connector roots can be weakened without affecting robot.
    colliders=[];materials={};remap=dict(roots)
    for src_root in roots:
        for prim in Usd.PrimRange(source.GetPrimAtPath(src_root)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Frozen collider subtree must not itself create another actor')
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                raise ValueError(f'Frozen collision has no explicit physics material: {prim.GetPath()}')
            path=str(material.GetPath())
            if path not in materials:
                name=re.sub('[^A-Za-z0-9_]','_',material.GetPrim().GetName())
                destination=f'{material_root}/M{len(materials):02d}_{name}'
                materials[path]=(destination,material)
                remap[path]=destination
            colliders.append((prim,src_root,path))
    if not colliders:raise ValueError('Frozen connector has no collision specs')
    def mapped(path):
        text=str(path)
        for src,dst in sorted(remap.items(),key=lambda x:-len(x[0])):
            if _under(text,src):return Sdf.Path(dst+text[len(src):])
        return Sdf.Path(text)
    def copy_spec(src_path,dst_path):
        Sdf.CreatePrimInLayer(target_layer,Sdf.Path(dst_path).GetParentPath())
        if not Sdf.CopySpec(source_layer,Sdf.Path(src_path),target_layer,Sdf.Path(dst_path)):
            raise RuntimeError(f'Frozen USD copy failed: {src_path}')
        prim=stage.GetPrimAtPath(dst_path)
        for child in Usd.PrimRange(prim):
            for rel in child.GetRelationships():
                targets=rel.GetTargets()
                if targets:rel.SetTargets([mapped(x) for x in targets])
            for attr in child.GetAttributes():
                connections=attr.GetConnections()
                if connections:attr.SetConnections([mapped(x) for x in connections])
        return prim
    expected_paths={str(mapped(prim.GetPath())) for prim,_,_ in colliders}
    extra_colliders=[]
    for dst_root in roots.values():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(dst_root)):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and str(prim.GetPath()) not in expected_paths:
                extra_colliders.append(str(prim.GetPath()))
    if extra_colliders:
        raise ValueError('Target has undeclared connector colliders; compatibility review required: '+repr(extra_colliders))
    # Isolate PhysicsMaterials so a shared robot material is never overwritten.
    UsdGeom.Scope.Define(stage,material_root)
    for src,(dst,_) in materials.items():copy_spec(src,dst)
    binding_changes=[];installed=[]
    for prim,src_root,material_path in colliders:
        dst=str(mapped(prim.GetPath()));dst_root=roots[src_root]
        # Copy only relative intermediate transforms: never actor/root poses.
        ancestors=[];parent=prim.GetParent()
        while str(parent.GetPath())!=src_root:
            ancestors.append(parent);parent=parent.GetParent()
        for ancestor in reversed(ancestors):
            path=str(mapped(ancestor.GetPath()))
            target=stage.GetPrimAtPath(path)
            if not target:target=stage.DefinePrim(path,ancestor.GetTypeName())
            for attr in ancestor.GetAttributes():
                if attr.GetName().startswith('xformOp') and attr.HasAuthoredValueOpinion():
                    Sdf.CopySpec(source_layer,attr.GetPath(),target_layer,Sdf.Path(path).AppendProperty(attr.GetName()))
        copied=copy_spec(str(prim.GetPath()),dst)
        material=UsdShade.Material.Get(stage,materials[material_path][0])
        # Freeze the resolved local physics material independently of visual materials.
        UsdShade.MaterialBindingAPI.Apply(copied).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
        parent=copied.GetParent()
        while _under(str(parent.GetPath()),dst_root):
            api=UsdShade.MaterialBindingAPI(parent)
            rel=api.GetDirectBindingRel('physics')
            if rel and rel.GetTargets() and api.GetMaterialBindingStrength(rel)==UsdShade.Tokens.strongerThanDescendants:
                api.SetMaterialBindingStrength(rel,UsdShade.Tokens.weakerThanDescendants)
                binding_changes.append(str(rel.GetPath()))
            parent=parent.GetParent()
        resolved,_=UsdShade.MaterialBindingAPI(copied).ComputeBoundMaterial('physics')
        if not resolved or resolved.GetPath()!=material.GetPath():
            raise RuntimeError(f'External ancestor overrides frozen PhysicsMaterial at {dst}; move connector material setup before installer')
        if _physics_values(resolved)!=_physics_values(materials[material_path][1]):
            raise RuntimeError(f'Frozen physics material readback differs: {dst}')
        installed.append({'collider':dst,'collision_enabled':bool(UsdPhysics.CollisionAPI(copied).GetCollisionEnabledAttr().Get()),
                          'material':str(material.GetPath())})
    # Preserve the physical model of the existing passive internal joint, not
    # the initial articulation state captured by the model verification run.
    joint=copy_spec(JOINT,str(passive_joint_path))
    joint_schemas=joint.GetMetadata('apiSchemas')
    joint_tokens=list(joint_schemas.GetAddedOrExplicitItems()) if joint_schemas else []
    joint_tokens=[x for x in joint_tokens if 'StateAPI' not in x]+[x for x in old_joint_tokens if 'StateAPI' in x]
    joint.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(list(dict.fromkeys(joint_tokens))))
    for attr in list(joint.GetAttributes()):
        if attr.GetName().startswith('state:'):joint.RemoveProperty(attr.GetName())
    for name,value in old_state.items():joint.CreateAttribute(name,old_state_types[name]).Set(value)
    # Carry only the frozen reduced-coordinate representation/solver properties
    # on connector roots; source root poses/mass/velocity are never copied.
    articulation_properties=[]
    for src,dst in roots.items():
        source_prim=source.GetPrimAtPath(src);target=stage.GetPrimAtPath(dst)
        source_schemas=source_prim.GetMetadata('apiSchemas')
        source_tokens=list(source_schemas.GetAddedOrExplicitItems()) if source_schemas else []
        allowed=('PhysicsArticulationRootAPI','PhysxArticulationAPI')
        current=target.GetMetadata('apiSchemas')
        tokens=list(current.GetAddedOrExplicitItems()) if current else []
        tokens=[x for x in tokens if x not in allowed]+[x for x in source_tokens if x in allowed]
        target.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(list(dict.fromkeys(tokens))))
        for attr in source_prim.GetAttributes():
            if (attr.GetName().startswith('physxArticulation:') or attr.GetName() in ('physxRigidBody:solverPositionIterationCount','physxRigidBody:solverVelocityIterationCount','physxRigidBody:sleepThreshold')) and attr.HasAuthoredValueOpinion():
                target.CreateAttribute(attr.GetName(),attr.GetTypeName()).Set(attr.Get())
                articulation_properties.append(str(target.GetPath().AppendProperty(attr.GetName())))
    after=_snapshot(stage,tuple(roots.values()))
    if before!=after:raise RuntimeError('Installing frozen colliders changed caller pose, mass/inertia or velocity')
    final_joint=UsdPhysics.Joint(stage.GetPrimAtPath(passive_joint_path))
    if list(map(str,final_joint.GetBody0Rel().GetTargets()))!=[body_path] or list(map(str,final_joint.GetBody1Rel().GetTargets()))!=[nut_path]:
        raise RuntimeError('Frozen joint path remapping failed')
    if {a.GetName():a.Get() for a in joint.GetAttributes() if a.GetName().startswith('state:')}!=old_state:
        raise RuntimeError('Caller passive-joint initial state changed')
    record={'scope':'FROZEN_CONNECTOR_COLLISION_AND_PASSIVE_MODEL_INSTALL_NOT_ASSEMBLY_SUCCESS',
            'model_path':str(model_path),'initialization_guard':guard,'installed_colliders':installed,
            'private_materials':[{ 'source':src,'installed':dst,'parameters':_physics_values(mat)} for src,(dst,mat) in materials.items()],
            'extra_existing_colliders_disabled':[],'weakened_connector_binding_paths':list(dict.fromkeys(binding_changes)),
            'passive_joint_path':str(passive_joint_path),'caller_joint_initial_state_preserved':True,
            'articulation_properties':articulation_properties,'pose_mass_velocity_before':before,
            'pose_mass_velocity_after':after,'actor_pose_mass_com_inertia_velocity_unchanged':True,
            'robot_or_drive_caps_modified':False,'laboratory_joint_or_external_drive_installed':False,
            'world_solver_configuration_modified':False,'requires_manifest_solver_configuration_before_physics':True}
    socket_contact_paths=[entry['collider'] for entry in installed
                          if entry['collision_enabled'] and _under(entry['collider'],socket_root_path)]
    record['socket_contact_filter_paths_for_recording']=socket_contact_paths
    if prepared is not None:
        prepared.setdefault('report',{})['frozen_connector_model']=copy.deepcopy(record)
        if 'contact_recording' in prepared:
            recording=prepared['contact_recording']
            recording['additional_required_contact_filter_paths']=list(dict.fromkeys([
                *recording.get('additional_required_contact_filter_paths',[]),*socket_contact_paths]))
            recording['minimum_contact_records']=max(int(recording.get('minimum_contact_records',0)),8192)
    return record
