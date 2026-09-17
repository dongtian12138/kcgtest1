#!/usr/bin/env python3
"""Run an existing review on one ended episode, never on live physics.

The visibility decision still requires inspecting this episode's actual frames.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src/kcg_connector'), str(ROOT/'src/kcg_connector/isaac'),
               str(Path(__file__).resolve().parent/'postreview')]


def sealed(run):
    archive = run/'truth_samples.msgpack.gz'
    index = json.loads(Path(str(archive)+'.index.json').read_text())
    if not (run/'motion_timing.json').is_file() or not index['blocks'] or index['blocks'][-1]['end'] != archive.stat().st_size:
        raise ValueError('Requires ended full-episode motion and a matching sealed archive')
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--part', choices=('body','nut','key','turns','terminal','band','whole'), required=True)
    parser.add_argument('--output', type=Path, help='Required outside the run for the independent terminal supplement')
    args = parser.parse_args();run=args.run.resolve();sealed(run)
    if args.part in ('body','nut'):
        from selective_postrun_reader import review_rows
        if args.part=='body':
            import evaluate_visual_body_grasp as module
            module.iter_truth_samples=lambda directory: review_rows(directory,hand_contacts=True)
            result=module.review(run,ROOT)
        else:
            import evaluate_source_nut_pad as module
            module.iter_truth_samples=lambda directory: review_rows(directory,hand_contacts=True)
            result=module.review(run,geometry_plan=ROOT/'artifacts/grasp_capacity_20260914/selected_two_nail_geometry.json')
    elif args.part=='key':
        from evaluate_source_key_containment import review
        result=review(run)
    elif args.part=='turns':
        from review_episode_turn_progress import review
        result=review(run)
    elif args.part=='terminal':
        if args.output is None or args.output.resolve().is_relative_to(run):
            raise ValueError('Terminal supplement output must be explicitly outside the source run')
        from supplement_terminal_release import review
        from trace_metadata import without_cyclic_gc
        result=without_cyclic_gc(review,run)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    elif args.part=='band':
        from review_source_band_occlusion import review
        result=review(run)
    else:
        from evaluate_visual_assembly_v1 import review
        from trace_metadata import without_cyclic_gc
        result=without_cyclic_gc(review,run)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
