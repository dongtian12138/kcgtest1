#!/usr/bin/env python3
"""Compare a finished stage's wrist load with its recorded hand-side contacts.

The observer is offline. It writes no controller input and does not calibrate or
zero the loaded sensor. The existing hand gravity and frozen-hand inertia models
are retained, so instantaneous finger dynamics are not independently identified.
"""
import argparse
import gzip
import json
from pathlib import Path
from collections import Counter

import ijson
import numpy as np


def evaluate(directory, stage_name, last_seconds=.5):
    if Path(stage_name).name != stage_name or not 0 < last_seconds <= 2.:
        raise ValueError("one local stage and a positive window of at most two seconds are required")
    stage=directory/"socket_transport"/stage_name
    controls=list(stage.glob("*controller_result.json"))
    if len(controls)!=1:
        raise ValueError("the stage must have one finished controller record")
    control=json.loads(controls[0].read_text())
    end=int(control["last_step"])-1
    sensors={};previous=None;dt=None;first=None
    with gzip.open(stage/"joint_ft_samples.json.gz","rb") as stream:
        for row in ijson.items(stream,"item",use_float=True):
            if previous is not None and dt is None:
                dt=(row["simulation_time_s"]-previous["simulation_time_s"])/(row["step"]-previous["step"])
                if not np.isfinite(dt) or dt<=0:raise ValueError("invalid sensor time interval")
                first=max(int(control["first_step"]),end-round(last_seconds/dt)+1)
                if previous["step"]>=first:sensors[int(previous["step"])]=previous
            if first is not None and row["step"]>=first:sensors[int(row["step"])]=row
            previous=row
    if first is None or sorted(sensors)!=list(range(first,end+1)):
        raise ValueError("the sealed sensor archive does not cover the requested window")
    rows=[];partners=set();phases=Counter()
    with (directory/"truth_samples.jsonl").open("rb") as stream:
        for line in stream:
            # The current JSONL writer puts step first. Skip unrelated samples
            # before decoding their large contact arrays; retain the old-layout fallback.
            if line.startswith(b'{"step":'):
                step=int(line[8:line.index(b',')])
                if step<first:continue
                if step>end:break
            actual=json.loads(line);step=int(actual["step"])
            if step<first:continue
            if step>end:break
            sensor=sensors[step];origin=np.asarray(sensor["handbase_position_world_m"])
            R=np.asarray(sensor["handbase_rotation_world_row_major"]).reshape(3,3)
            contact=np.zeros(6)
            for field,impulse_field in (("tensor_headers","impulse_n_s"),
                                         ("friction_headers","tangential_impulse_n_s")):
                for header in actual["contacts"][field]:
                    if "/handbase_link" not in header["paths"][0]:continue
                    partners.add(header["paths"][1])
                    for hit in header["contacts"]:
                        force=np.asarray(hit[impulse_field])/dt
                        contact[:3]+=force
                        contact[3:]+=np.cross(np.asarray(hit["position_m"])-origin,force)
            contact=np.r_[R.T@contact[:3],R.T@contact[3:]]
            bias=np.asarray(sensor["run_specific_tare_canonical_sensor_wrench"])-np.asarray(sensor["run_specific_tare_modeled_gravity_sensor_wrench"])
            residual=np.asarray(sensor["hand2arm_canonical_sensor_wrench"])-np.asarray(sensor["modeled_hand_gravity_sensor_wrench"])-bias
            prediction=sensor.get("dynamic_inertia_prediction") or {}
            inertia=np.asarray(prediction.get("predicted_positive_inertia_wrench_sensor") or [0.]*6)
            defect=residual+inertia-contact
            rows.append({"step":step,"phase":actual["phase"],"contact_sensor_n_nm":contact.tolist(),
                         "residual_sensor_n_nm":residual.tolist(),"existing_inertia_prediction_n_nm":inertia.tolist(),
                         "balance_defect_sensor_n_nm":defect.tolist()})
            phases[actual["phase"]]+=1
    if [x["step"] for x in rows]!=list(range(first,end+1)):
        raise ValueError("raw contact and wrist intervals do not coincide")
    keys=("contact_sensor_n_nm","residual_sensor_n_nm","existing_inertia_prediction_n_nm","balance_defect_sensor_n_nm")
    result={"scope":"POST_MOTION_WRIST_CONTACT_BALANCE_NOT_ONLINE_COMPENSATION", "source_run":str(directory),
            "stage_name":stage_name,"interval_steps":[first,end],"physics_dt_s":dt,"sample_count":len(rows),
            "phases":dict(phases),"all_samples_are_hold":all("hold" in phase for phase in phases),
            "hand_side_impulse_vectors_only":True,"contact_partners":sorted(partners),
            "existing_free_space_tare_unchanged":True,"loaded_rezeroing_performed":False,
            "gravity_uses_existing_joint_FK_model":True,"inertia_uses_existing_frozen_hand_prediction":True,
            "exact_finger_inertia_not_established":True,
            "mean":{key:np.mean([x[key] for x in rows],axis=0).tolist() for key in keys},
            "standard_deviation":{key:np.std([x[key] for x in rows],axis=0).tolist() for key in keys}}
    prefix=directory/(stage_name+"_hand_wrench_balance")
    for path in [prefix.with_name(prefix.name+"_posthoc_v1.json"),prefix.with_name(prefix.name+"_samples_v1.json")]:
        if path.exists():raise FileExistsError(path)
    prefix.with_name(prefix.name+"_posthoc_v1.json").write_text(json.dumps(result,indent=2)+"\n")
    prefix.with_name(prefix.name+"_samples_v1.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--stage-name",default="nut_rotation")
    parser.add_argument("--last-seconds",type=float,default=.5)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.directory.resolve(),args.stage_name,args.last_seconds),indent=2))
