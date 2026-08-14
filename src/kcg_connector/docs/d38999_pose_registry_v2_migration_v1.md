# D38999 pose-registry v2 migration v1

This is a disabled, side-by-side migration design.  It does not edit
`connector_pose_observation_v1.yaml`, the active E2E runner, or any USD/OBJ
asset, and it cannot authorize robot control.

## Confirmed mismatch

The active registry labels both D38999 endpoint models `keyed_order_1`.  The
exact hash-bound FoundationPose proxy OBJs do not support that claim:

- the loose-body vertex multiset is invariant under a 180-degree local-Z
  rotation;
- the fixed-receptacle vertex multiset is invariant under the same rotation;
- the independently rotating coupling-nut mesh is invariant at all divisors
  of order 24, including order 24;
- none of these meshes contains a unique polarization key.

The proxy configuration's `polarization: N` value records selected identity.
It is not geometry and cannot make yaw observable.  A FoundationPose
quaternion chosen from two equivalent endpoint hypotheses is therefore not a
unique physical key angle.

## Versioned correction

The proposed `kcg_connector_pose_contract_v2` runs beside v1.  It does not
reinterpret an old observation in place.  Each current endpoint is registered
with a discrete axial order-2 group around object-frame +Z and pose semantics
`equivalence_class`.  The nut is a separate order-24 component because its
absolute tooth yaw is not the loose body's key yaw.

With the existing `parent_T_child` convention, equivalent poses are

```text
parent_T_object(k) = parent_T_object * object_Rz(2*pi*k/n)
```

The v1-to-v2 bridge is diagnostic only.  It must reject D38999 pair
publication and control because `keyed_order_1` cannot losslessly encode an
order-2 equivalence class.

## Required source for a keyed revision

A keyed proxy must use new model IDs.  Reusing the current model IDs would
make observations of different geometry indistinguishable.  Acceptable
primary sources are either:

1. licensed vendor CAD for the exact orderable plug/receptacle part numbers
   and polarization; or
2. calibrated metrology/scan data from traceable physical mates.

The source record must bind license, identity, units, coordinate frame,
original and derived hashes, conversion, and uncertainty.  The derived model
must contain the asymmetric feature on each mate, object-frame key-reference
directions, the mating key angle/tolerance, and calibrated object-to-assembly
transforms.  Public shell-outline drawings, a part-number string, or a
manually guessed notch are insufficient.

## FoundationPose evaluation

For endpoint order `n`, compare orientation using

```text
min_k angle(truth_q^-1 * estimated_q * Rz(2*pi*k/n))
```

Report 3D translation and unsigned connector-axis error separately.  Retain
all loose/fixed hypothesis combinations; for two order-2 endpoints the pair
still has two distinct relative-yaw branches.  A temporal tracker must keep
branch identity and flag unexplained branch flips instead of silently
canonicalizing them.  Evaluate the coupling nut separately modulo order 24.
Raw yaw of a chosen representative is never keyed-yaw accuracy.

## Authorization boundary

There are three separate gates:

- Evaluation/preflight pair publication can begin only after a content-
  addressed v2 parser is active and withheld-truth metrics pass modulo the
  declared symmetry.
- Pick-only visual control may be possible without a unique key only if every
  grasp and collision result is valid for every retained symmetry hypothesis,
  the controller is stage-limited, and repeated closed-loop pick tests pass.
- Insertion/twist/full-workflow visual control requires new hash-bound keyed
  geometry, observed or independently calibrated mate key frames, rejection
  of all false equivalent hypotheses, qualified target transforms, withheld
  relative-yaw accuracy, collision/FT/repeatability gates, and no truth-filled
  pose fields.

A separately validated force-guided key-search strategy could provide another
route, but it is intentionally outside this registry migration contract.

Run the current CPU-only audit:

```bash
PYTHONPATH=src/kcg_connector python3 -m \
  kcg_connector.d38999_pose_registry_v2_migration \
  --repository "$PWD"
```

The expected exit code is zero for a valid audit.  The report status remains
`AUDIT_CONFIRMED_V2_DESIGNED_NOT_ACTIVATED`; a successful audit is not runtime
or control readiness.
