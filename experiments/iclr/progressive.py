"""Independent Original/Rectified 4->2->1 experiment; GPU commands default to dry run."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.iclr.progressive_protocol import (ARMS, check, prepare, read,
    completed_pair, require_review, checkpoint_for)
from experiments.iclr.protocol import REPO, atomic_json, sha256_file


def export_folder(root, phase):
    return root/'final' if phase == 'final' else root/'previews'/phase


def torchrun(python, module, *args):
    return [python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            '--module', module, *map(str,args)]


def commands(args, m):
    root = Path(args.output).resolve()
    if args.action == 'train':
        return [torchrun(args.python,'experiments.iclr.progressive_train','--output',root,
                         '--stage',args.stage,'--arm',arm) for arm in ARMS]
    phase = 'final' if args.action == 'evaluate' else args.phase
    result = [torchrun(args.python,'experiments.iclr.progressive_export','--output',root,'--phase',phase)]
    if args.action == 'evaluate':
        for arm in ARMS:
            result.append([args.vbench_python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(REPO/'scripts/run_vbench.py'),'--videos_path',str(root/'final'/arm/'videos'),
                '--full_info_path',m['full_info'],'--output_dir',str(root/'final'/arm/'vbench'),
                '--name',arm,'--device','cuda','--samples_per_prompt','1','--mode','vbench_standard'])
    return result


def preflight(args, m):
    root = Path(args.output).resolve()
    if args.action == 'train':
        previous = 'source' if args.stage == '4to2' else '4to2'
        require_review(root, previous)
        if previous == '4to2': completed_pair(root, previous)
        if (root/'training'/args.stage).exists(): raise FileExistsError('Preserve existing training output')
    else:
        phase = 'final' if args.action == 'evaluate' else args.phase
        if phase != 'source':
            completed_pair(root, '2to1' if phase == 'final' else phase)
        if phase == 'final': require_review(root,'2to1')
        if export_folder(root,phase).exists(): raise FileExistsError('Preserve existing export output')
    for arm in ARMS:
        if sha256_file(m['sources']['arms'][arm]['checkpoint']) != m['sources']['arms'][arm]['sha256']:
            raise ValueError('Source teacher changed')


def finish_export(root, phase):
    from experiments.iclr.progressive_export import cases_for, modes_for, video_path, validate_video
    m = check(root)
    folder = export_folder(root,phase)
    expected = {str(video_path(root,phase,arm,mode,case)):(arm,mode,case,steps,legacy)
        for arm in ARMS for mode,steps,legacy in modes_for(phase) for case in cases_for(m,phase)}
    videos, runtime = [], []
    stage = '2to1' if phase == 'final' else phase
    checkpoints = {arm: sha256_file(checkpoint_for(root,arm,stage)) for arm in ARMS}
    for rank in range(8):
        receipt = read(folder/f'rank{rank}.json')
        if receipt['rank'] != rank or receipt['world'] != 8 or receipt['phase'] != phase or receipt['manifest_sha256'] != sha256_file(root/'manifest.json'):
            raise ValueError('Export rank receipt mismatch')
        if any(receipt['checkpoints'][arm]['sha256'] != checkpoints[arm] for arm in ARMS):
            raise ValueError('Export used a different checkpoint')
        videos.extend(receipt['videos'])
        runtime.append(receipt['runtime'])
    if len(videos) != len(expected) or {x['path'] for x in videos} != set(expected):
        raise ValueError('Incomplete or duplicate paired video export')
    for row in videos:
        arm,mode,case,steps,legacy = expected[row['path']]
        controls = {'arm':arm,'mode':mode,'case':case,'steps_every_frame':steps,
                    'legacy_renoise':legacy,'ffe':False,'use_ema':False}
        if any(row.get(k) != v for k,v in controls.items()) or sha256_file(row['path']) != row['sha256']:
            raise ValueError('Video protocol/hash mismatch')
        if phase != 'final': validate_video(row['path'])
    atomic_json(folder/'export.done.json', {'phase':phase,'manifest_sha256':sha256_file(root/'manifest.json'),
        'checkpoints':checkpoints,'runtime':runtime,'videos':videos,'visual_inspection':'pending'})
    if phase != 'final':
        atomic_json(folder/'review.template.json', {'export_sha256':sha256_file(folder/'export.done.json'),
            'dynamic_playback_pending':True,'inspected':[
                {'path':x['path'],'usable':False,'method':'unreviewed','notes':''} for x in videos]})
        print(f'Inspect paired MP4s and temporal sheets; complete reviews/{phase}.json before the next stage.',flush=True)
    return videos


def report(root):
    from experiments.rebuttal.consolidate_results import load_vbench_results, official_totals
    m = check(root)
    stages = {stage: completed_pair(root,stage) for stage in ('4to2','2to1')}
    reviews = {phase:require_review(root,phase) for phase in ('source','4to2','2to1')}
    exported = read(root/'final/export.done.json')
    if len(exported['videos']) != 1888: raise ValueError('Require 944x1 per arm')
    results = {}
    for arm in ARMS:
        path = root/'final'/arm/'vbench'/f'{arm}_eval_results.json'
        scores = load_vbench_results(path)
        total = official_totals(scores)
        if len(scores) != 16 or total is None: raise ValueError('All sixteen VBench dimensions required')
        results[arm] = {'dimensions':scores, 'scores':total, 'result_sha256':sha256_file(path),
                        'checkpoint_sha256':stages['2to1'][arm]['checkpoint_sha256']}
        if exported['checkpoints'][arm] != results[arm]['checkpoint_sha256']:
            raise ValueError('Final checkpoint lineage mismatch')
    for row in exported['videos']:
        if sha256_file(row['path']) != row['sha256']: raise ValueError('Final video changed')
    result = {'profile':m['profile'],'seed':2026,'updates_total':1200,'weight_source':'raw',
        'sampling':'all1_no_ffe','prompt_protocol':'original_944x1','results':results,
        'dynamic_playback_pending':any(x.get('dynamic_playback_pending',True) for x in reviews.values()),
        'stages':stages,'interpretation':'Report both arms as observed; do not select checkpoints by score.'}
    atomic_json(root/'report.json',result)
    return result


def inventory(roots):
    candidates = []
    for root in roots:
        # rg is read-only, does not follow symlink loops, and does not load multi-GB weights.
        proc = subprocess.run(['rg','--files','--hidden','--no-ignore',str(Path(root).resolve())],
                              text=True,capture_output=True)
        if proc.returncode not in (0,1): raise RuntimeError(proc.stderr)
        for line in proc.stdout.splitlines():
            path = Path(line)
            if path.name == 'model.pt' and path.parent.name == 'checkpoint_model_000300':
                candidates.append(str(path))
    return {'candidate_paths_only': sorted(set(candidates)), 'verified_four_step_sources':False,
        'next':'Inspect provenance/initialization/training records, then prepare and source previews. Never substitute 1000-step weights.',
        'if_missing':{'recovery':'Restore the historical 9-frame adjacent-CD pair after agreeing source and cost.',
            'extra_updates':600,'extra_training_runs':2,
            'trajectory_generation':'Zero if a verified shared LMDB exists; otherwise quote separately before generating.',
            'progressive_updates':1200,'automatic_recovery':False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('inventory','prepare','preview','train','evaluate','report'))
    parser.add_argument('--output',default='')
    parser.add_argument('--source-pair')
    parser.add_argument('--data')
    parser.add_argument('--roots',nargs='+',default=[])
    parser.add_argument('--stage',choices=('4to2','2to1'))
    parser.add_argument('--phase',choices=('source','4to2','2to1'))
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--vbench-python',default=sys.executable)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--session',default='iclr_progressive')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == 'inventory':
        if not args.roots: parser.error('inventory requires --roots')
        print(json.dumps(inventory(args.roots),indent=2)); return
    if not args.output: parser.error('--output required')
    root = Path(args.output).resolve()
    if args.action == 'prepare':
        if not args.source_pair or not args.data: parser.error('--source-pair and --data required')
        prepare(args.source_pair,args.data,root)
        print(f'Prepared {root}; source quality remains UNVERIFIED.'); return
    if args.action == 'report':
        print(json.dumps(report(root),indent=2)); return
    if args.action == 'train' and not args.stage: parser.error('train requires --stage')
    if args.action == 'preview' and not args.phase: parser.error('preview requires --phase')
    m = check(root)
    planned = commands(args,m)
    if not args.execute and not args.worker:
        for command in planned: print(shlex.join(command))
        print('DRY RUN: no GPU queries or model loading. Execution requires provenance and video review gates.')
        return
    preflight(args,m)
    from experiments.iclr.launch import parse_gpus, require_idle, start_tmux
    gpus = parse_gpus(args.gpus)
    require_idle(gpus)
    if not args.worker:
        argv = ['env',f'CUDA_VISIBLE_DEVICES={args.gpus}','CAUSAL_DISABLE_FLEX_ATTENTION=1',
            args.python,'-m','experiments.iclr.progressive',args.action,'--output',str(root),
            '--python',args.python,'--vbench-python',args.vbench_python,'--gpus',args.gpus,'--worker']
        if args.stage: argv.extend(['--stage',args.stage])
        if args.phase: argv.extend(['--phase',args.phase])
        # Each new launch gets its own log; no tmux-wide environment mutations.
        log = root/'logs'/f'{args.session}.log'
        if log.exists(): raise FileExistsError(log)
        start_tmux(args.session,argv,log); return
    if not os.environ.get('TMUX'): raise RuntimeError('Worker requires independent tmux')
    phase = 'final' if args.action == 'evaluate' else args.phase
    intent = root/'training'/args.stage if args.action == 'train' else export_folder(root,phase)
    # For training, each torchrun creates its own arm subdirectory after the rank barrier.
    atomic_json(intent/'launch.json',{'commands':planned,'manifest_sha256':sha256_file(root/'manifest.json')})
    environment = {**os.environ,'CUDA_VISIBLE_DEVICES':args.gpus,'CAUSAL_DISABLE_FLEX_ATTENTION':'1'}
    for index,command in enumerate(planned):
        require_idle(gpus)
        print(shlex.join(command),flush=True)
        subprocess.run(command,cwd=REPO,env=environment,check=True)
        if args.action != 'train' and index == 0: finish_export(root,phase)
    if args.action == 'train':
        completed_pair(root,args.stage)
        print('Both arms completed. Export and inspect heldout videos before advancing.',flush=True)
    elif args.action == 'evaluate':
        print(json.dumps(report(root),indent=2),flush=True)


if __name__ == '__main__': main()
