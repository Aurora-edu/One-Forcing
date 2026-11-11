"""Provenance and quality gates for the independent progressive experiment."""
import json
import math
from pathlib import Path

from omegaconf import OmegaConf
from experiments.iclr.protocol import REPO, atomic_json, sha256_file

CONFIG = REPO / 'experiments/iclr/configs/cd_progressive.yaml'
ARMS = ('original', 'rectified')
PROFILE = 'original_rectified_progressive_v1'
SOURCE_EXPERIMENT = 'curvature_adjacent_consistency_distillation'


def read(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    paths = [*sorted((REPO / 'experiments/iclr').glob('progressive_*.py')),
             REPO / 'experiments/iclr/progressive.py', CONFIG,
             *[REPO / p for p in ('utils/wan_wrapper.py', 'utils/distributed.py',
                 'utils/dataset.py', 'wan/modules/causal_model.py', 'wan/modules/model.py',
                 'wan/modules/attention.py', 'pipeline/causal_inference.py',
                 'scripts/run_vbench.py', 'experiments/rebuttal/consolidate_results.py')]]
    return {str(p.relative_to(REPO)): sha256_file(p) for p in paths}


def raw_state(path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = payload.get('generator')
    if not isinstance(state, dict) or not state:
        raise ValueError('Explicit raw generator required; no EMA fallback')
    fixed = {k.replace('model._fsdp_wrapped_module.', 'model.', 1): v for k, v in state.items()}
    if len(fixed) != len(state) or any(not torch.isfinite(v).all() for v in fixed.values()):
        raise ValueError('Invalid raw parameters')
    return payload, fixed


def audit_source_pair(spec):
    """A filename, aggregate score, or endpoint-regression checkpoint is insufficient."""
    if spec.get('origin') not in ('historical_four_step_cd', 'recovered_four_step_cd'):
        raise ValueError('Declare historical or separately recovered four-step CD provenance')
    initial = Path(spec['initial_checkpoint']).resolve()
    data_manifest = Path(spec['trajectory_manifest']).resolve()
    initial_hash, data_hash = sha256_file(initial), sha256_file(data_manifest)
    dm = read(data_manifest)
    if dm.get('generator_checkpoint_sha256') != initial_hash or dm.get('use_ema') is not False:
        raise ValueError('Trajectory source must match the documented raw initialization')
    if dm.get('guidance_scale') != 6. or dm.get('latents_shape') != [6505,6,21,16,60,104]:
        raise ValueError('Source trajectory data does not establish the historical 6505-row/CFG6 protocol')
    summaries, shapes = {}, []
    for arm in ARMS:
        record = spec[arm]
        path = Path(record['checkpoint']).resolve()
        digest = sha256_file(path)
        if digest != record['sha256']:
            raise ValueError(f'{arm}: source checkpoint hash mismatch')
        payload, state = raw_state(path)
        m = payload.get('experiment_metadata', {})
        expected_arm = 'curved' if arm == 'original' else 'rectified'
        required = {'experiment': SOURCE_EXPERIMENT, 'intervention': expected_arm,
                    'seed': 2026, 'training_num_frames': 9,
                    'dataset_manifest_sha256': data_hash, 'target_ema_decay': .95,
                    'world_size': 8, 'framewise': True, 'evaluation_use_ema': False}
        if payload.get('step') != 300 or any(m.get(k) != v for k, v in required.items()):
            raise ValueError(f'{arm}: require the documented 300-update/9-frame adjacent-CD source, not 1000-update endpoint weights')
        if m.get('training_objective') != 'adjacent_state_consistency_l2' or m.get('endpoint_loss_weight', 0) != 0:
            raise ValueError('Source is not the historical pure adjacent-CD objective')
        if m.get('rectification_treatment') == 'high_noise_local_progressive':
            raise ValueError('Local rectification is a different experiment, not the historical pair')
        times = m.get('training_timesteps', [])
        if len(times) != 5 or any(abs(a-b) > 1e-3 for a,b in zip(times, [1000,937.5,1000*5/6,625,0])):
            raise ValueError('Unexpected source timestep grid')
        evidence = Path(record['training_record']).resolve()
        intent = read(evidence)
        if m.get('initial_generator_sha256', intent.get('initial_generator_sha256')) != initial_hash:
            raise ValueError('Initialization hash provenance incomplete; a matching old filename is insufficient')
        if intent.get('seed') != 2026 or intent.get('intervention') != expected_arm or intent.get('dataset_manifest_sha256') != data_hash:
            raise ValueError('Training record disagrees with checkpoint provenance')
        cfg = intent.get('config', {})
        controls = {'seed': 2026, 'max_steps': 300, 'batch_size': 1, 'training_num_frames': 9,
                    'lr': 2e-6, 'beta1': .9, 'beta2': .999, 'weight_decay': .01,
                    'max_grad_norm': 10., 'target_ema_decay': .95, 'num_frame_per_block': 1}
        if any(cfg.get(k) != v for k,v in controls.items()):
            raise ValueError('Source training record does not establish the historical recipe')
        shapes.append({k: list(v.shape) for k, v in state.items()})
        summaries[arm] = {'checkpoint': str(path), 'sha256': digest,
                          'metadata': m, 'training_config': cfg, 'training_record': str(evidence),
                          'training_record_sha256': sha256_file(evidence)}
        del payload, state
    if shapes[0] != shapes[1]:
        raise ValueError('Source generator architectures differ')
    if summaries['original']['training_config'] != summaries['rectified']['training_config']:
        raise ValueError('Source training configs differ')
    for key in ('training_timesteps', 'target_ema_decay', 'training_objective', 'world_size'):
        if summaries['original']['metadata'].get(key) != summaries['rectified']['metadata'].get(key):
            raise ValueError(f'Unmatched source control: {key}')
    return {'origin': spec['origin'], 'initial_sha256': initial_hash,
            'trajectory_manifest_sha256': data_hash, 'arms': summaries,
            'quality_status': 'unverified_until_new_heldout_exports_are_inspected'}


def partition_prompts(prompts, count=64):
    if len(prompts) <= count:
        raise ValueError('Need training rows after the holdout')
    heldout_prompts = set(prompts[:count])
    train = [i for i, p in enumerate(prompts) if i >= count and p not in heldout_prompts]
    if len(train) < 8:
        raise ValueError('Insufficient training rows after excluding duplicate holdout prompts')
    cases = [{'id': f'holdout_{i:04d}', 'row': i, 'prompt': prompts[i], 'seed': 2026 + i}
             for i in range(0, count, 4)]
    if len(cases) != 16:
        raise ValueError('Exactly 16 fixed holdout cases required')
    return train, cases


def prepare(spec_path, data, output):
    from utils.dataset import CleanLatentLMDBDataset, retrieve_row_from_lmdb
    output, data = Path(output).resolve(), Path(data).resolve()
    if output.exists():
        raise FileExistsError('Use an independent empty experiment directory')
    sources = audit_source_pair(read(spec_path))
    ds = CleanLatentLMDBDataset(str(data))
    try:
        prompts = [retrieve_row_from_lmdb(ds.env, 'prompts', str, i) for i in range(len(ds))]
        train, cases = partition_prompts(prompts)
        for i in (0, train[0]):
            if tuple(ds[i]['clean_latent'].shape) != (21, 16, 60, 104):
                raise ValueError('Require 21-frame Wan clean latents')
    finally:
        ds.env.close()
    from experiments.rebuttal.paper_vbench_protocol import ASSETS, PINNED_HASHES
    from experiments.rebuttal.prepare_vbench_prompts import unique_vbench_prompts
    info = ASSETS / 'VBench_full_info.json'
    if sha256_file(info) != PINNED_HASHES[info.name]:
        raise ValueError('VBench prompt source changed')
    vb = unique_vbench_prompts(info)
    if len(vb) != 944:
        raise ValueError('Require 944 original prompts')
    formal = [{'id': f'vbench_{i:04d}', 'prompt': p, 'seed': i,
               'filename': f'{p}-0.mp4'} for i, p in enumerate(vb)]
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    manifest = {'profile': PROFILE, 'config': cfg, 'sources': sources,
        'source_spec': str(Path(spec_path).resolve()), 'source_spec_sha256': sha256_file(spec_path),
        'data': str(data), 'data_sha256': sha256_file(data / 'data.mdb'),
        'data_size_mtime': [data.joinpath('data.mdb').stat().st_size, data.joinpath('data.mdb').stat().st_mtime_ns],
        'training_rows': train, 'holdout': cases, 'vbench': formal,
        'full_info': str(info), 'full_info_sha256': sha256_file(info), 'code': code_hashes(),
        'holdout_scope': 'Excluded from progressive stages, including duplicate prompt text. Source teacher exposure may include these rows.'}
    output.mkdir(parents=True)
    atomic_json(output / 'manifest.json', manifest)
    (output / 'manifest.sha256').write_text(sha256_file(output / 'manifest.json') + '\n')
    return manifest


def check(output):
    output = Path(output)
    if (output / 'manifest.sha256').read_text().strip() != sha256_file(output / 'manifest.json'):
        raise ValueError('Prepared manifest changed')
    m = read(output / 'manifest.json')
    if m['profile'] != PROFILE or m['config'] != OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True) or m['code'] != code_hashes():
        raise ValueError('Experiment config or implementation changed; use the prepared revision')
    if sha256_file(m['source_spec']) != m['source_spec_sha256']:
        raise ValueError('Source selection changed')
    for arm in ARMS:
        evidence = m['sources']['arms'][arm]
        if sha256_file(evidence['training_record']) != evidence['training_record_sha256']:
            raise ValueError('Source training record changed')
    if sha256_file(m['full_info']) != m['full_info_sha256']:
        raise ValueError('VBench prompt source changed')
    data = Path(m['data']) / 'data.mdb'
    if [data.stat().st_size, data.stat().st_mtime_ns] != m['data_size_mtime']:
        raise ValueError('Training data changed')
    return m


def completed_pair(output, stage):
    m = check(output)
    records = {}
    for arm in ARMS:
        path = checkpoint_for(output, arm, stage)
        previous = checkpoint_for(output, arm, 'source' if stage == '4to2' else '4to2')
        done = read(path.parent.parent / 'training.done')
        expected = {'profile': PROFILE, 'arm': arm, 'stage': stage, 'seed': 2026,
                    'updates': 300, 'parent_sha256': sha256_file(previous),
                    'checkpoint_sha256': sha256_file(path), 'data_sha256': m['data_sha256'],
                    'config': m['config'], 'use_ema': False,
                    'interval_counts': [150,150] if stage == '4to2' else [300]}
        if any(done.get(k) != v for k,v in expected.items()):
            raise ValueError('Unmatched stage budget or lineage')
        rows = [json.loads(x) for x in (path.parent.parent/'metrics.jsonl').read_text().splitlines()]
        if [x['step'] for x in rows] != list(range(1,301)) or any(
            not math.isfinite(x[k]) for x in rows for k in ('loss','grad_norm')):
            raise ValueError('Need 300 finite recorded optimizer updates')
        if rows[-1]['input_stream_sha256'] != done['input_stream_sha256']:
            raise ValueError('Input stream receipt mismatch')
        if read(path.parent.parent/'step10_health.json').get('teacher_repeat_check') != 'pass':
            raise ValueError('Missing teacher repeatability health check')
        records[arm] = done
    if records['original']['input_stream_sha256'] != records['rectified']['input_stream_sha256']:
        raise ValueError('Arms did not consume identical data and initial noise')
    return records


def checkpoint_for(output, arm, stage):
    m = check(output)
    if stage == 'source':
        return Path(m['sources']['arms'][arm]['checkpoint'])
    return Path(output) / 'training' / stage / arm / 'checkpoint_model_000300/model.pt'


def validate_stage_checkpoint(path, m, arm, stage, parent_hash):
    payload, state = raw_state(path)
    meta = payload.get('experiment_metadata', {})
    expected = {'profile': PROFILE, 'arm': arm, 'stage': stage,
                'seed': 2026, 'updates': 300, 'parent_sha256': parent_hash,
                'data_sha256': m['data_sha256'], 'config': m['config'], 'use_ema': False}
    if payload.get('step') != 300 or any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint is not from the correct completed progressive stage/arm')
    return state


def require_review(output, phase):
    """Review is entered by the inspecting agent/person, never auto-filled by decoding."""
    output = Path(output)
    receipt = output / 'previews' / phase / 'export.done.json'
    review = read(output / 'reviews' / f'{phase}.json')
    if review.get('export_sha256') != sha256_file(receipt):
        raise ValueError('Review does not identify this exact preview export')
    export = read(receipt)
    expected = {x['path']: x for x in export['videos']}
    inspected = review.get('inspected', [])
    if not expected or len(inspected) != len(expected) or {x['path'] for x in inspected} != set(expected):
        raise ValueError('Inspect every fixed paired preview, including both source samplers')
    for row in inspected:
        if row.get('usable') is not True or row.get('method') not in ('video_playback', 'temporal_frames') or not row.get('notes'):
            raise ValueError('Preview is unreviewed or unusable; retain outputs and diagnose before the next stage')
        if row.get('method') == 'temporal_frames' and review.get('dynamic_playback_pending') is not True:
            raise ValueError('Frame inspection must disclose pending dynamic playback')
        if sha256_file(row['path']) != expected[row['path']]['sha256']:
            raise ValueError('Reviewed video changed')
    return review
