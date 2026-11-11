"""One arm/stage of progressive distillation. GPU execution is launcher-only."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.iclr.progressive_protocol import (PROFILE, check, checkpoint_for,
    raw_state, require_review, validate_stage_checkpoint)
from experiments.iclr.progressive_core import STAGES, distillation_loss, two_step_target
from experiments.iclr.protocol import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', required=True, choices=('original', 'rectified'))
    parser.add_argument('--stage', required=True, choices=tuple(STAGES))
    args = parser.parse_args()
    if not os.environ.get('TMUX') or os.environ.get('WORLD_SIZE') != '8':
        raise RuntimeError('Use the independent tmux launcher on eight idle GPUs')
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, DistributedSampler, Dataset
    from utils.dataset import CleanLatentLMDBDataset
    from utils.distributed import launch_distributed_job, fsdp_wrap, get_fsdp_wrap_kwargs, fsdp_state_dict
    from utils.misc import set_seed
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

    root = Path(args.output).resolve()
    m = check(root)
    previous = 'source' if args.stage == '4to2' else '4to2'
    require_review(root, previous)
    teacher_path = checkpoint_for(root, args.arm, previous)
    parent_hash = sha256_file(teacher_path)
    if previous == 'source':
        if parent_hash != m['sources']['arms'][args.arm]['sha256']:
            raise ValueError('Source teacher changed')
        _, initial_state = raw_state(teacher_path)
    else:
        initial_state = validate_stage_checkpoint(teacher_path, m, args.arm, previous,
                                                 m['sources']['arms'][args.arm]['sha256'])
    config = OmegaConf.create(m['config'])
    run = root / 'training' / args.stage / args.arm
    if run.exists():
        raise FileExistsError('No weights-only resume; use a fresh experiment after interruption')
    launch_distributed_job()
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', torch.cuda.current_device())
    set_seed(2026 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16
    # All ranks finish the empty-output check before rank0 writes anything.
    dist.barrier()
    metadata = {'profile': PROFILE, 'arm': args.arm, 'stage': args.stage,
        'seed': 2026, 'updates': 300, 'parent_sha256': parent_hash,
        'parent_checkpoint': str(teacher_path), 'data_sha256': m['data_sha256'],
        'config': m['config'], 'use_ema': False, 'weight_source': 'generator',
        'objective': 'deterministic_two_teacher_steps_to_one_student_step_x0_mse',
        'history': 'clean_previous_frames_teacher_forcing; inference uses generated history'}
    if rank == 0:
        run.mkdir(parents=True)
        atomic_json(run / 'run.intent.json', {**metadata, 'manifest_sha256': sha256_file(root/'manifest.json'),
            'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(),
                        'attention_fallback': os.environ.get('CAUSAL_DISABLE_FLEX_ATTENTION'),
                        'wan_wrapper': __import__('utils.wan_wrapper', fromlist=['x']).__file__,
                        'distributed': __import__('utils.distributed', fromlist=['x']).__file__}})
    dist.barrier()

    def build(trainable):
        model = WanDiffusionWrapper(**m['config']['model_kwargs'], is_causal=True)
        model.model.num_frame_per_block = 1
        model.load_state_dict(initial_state, strict=True)
        model.requires_grad_(trainable).eval()
        if trainable:
            model.enable_gradient_checkpointing()
        return fsdp_wrap(model, **get_fsdp_wrap_kwargs(config, 'generator', ['causal_wan_block']))

    student, teacher = build(True), build(False)
    del initial_state
    gc.collect()
    text = fsdp_wrap(WanTextEncoder().eval().requires_grad_(False),
                     **get_fsdp_wrap_kwargs(config, 'text_encoder')).eval()
    base = CleanLatentLMDBDataset(m['data'])

    class Rows(Dataset):
        def __len__(self): return len(m['training_rows'])
        def __getitem__(self, i):
            index = m['training_rows'][i]
            return {**base[index], 'row_index': index}

    ds = Rows()
    sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=2026, drop_last=True)
    loader = DataLoader(ds, sampler=sampler, batch_size=1, num_workers=0, drop_last=True,
                        generator=torch.Generator().manual_seed(2026))
    iterator, epoch = iter(loader), 0
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-6, betas=(.9,.999), weight_decay=.01)
    stream_digest = hashlib.sha256()
    interval_counts = [0] * STAGES[args.stage][1]
    started = time.monotonic()
    for update in range(1,301):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        stored = batch['clean_latent']
        if tuple(stored.shape) != (1,21,16,60,104):
            raise ValueError('Unexpected stored training shape')
        # Retain the historical teacher's nine-frame training window.
        clean = stored[:, :config.training_num_frames].to(device=device, dtype=dtype)
        # Independent per-update generator keeps noise identical across arms despite model RNG use.
        noise_seed = 2026 + 1000000 * (args.stage == '2to1') + update * world + rank
        rng = torch.Generator(device=device).manual_seed(noise_seed)
        noise = torch.randn(clean.shape, device=device, dtype=dtype, generator=rng)
        local = {'row': batch['row_index'].tolist(), 'prompt': batch['prompts'], 'noise_seed': noise_seed,
                 'noise_sha256': hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest()}
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        stream_digest.update(json.dumps(gathered, sort_keys=True).encode())
        with torch.no_grad():
            conditioning = text(text_prompts=list(batch['prompts']))

        def predict(model, x, sigma):
            t = torch.full(clean.shape[:2], sigma * 1000, device=device, dtype=torch.float32)
            return model(noisy_image_or_video=x.to(dtype), conditional_dict=conditioning,
                         timestep=t, clean_x=clean)[1]

        optimizer.zero_grad(set_to_none=True)
        loss, details = distillation_loss(lambda x,t: predict(student,x,t),
            lambda x,t: predict(teacher,x,t), clean, noise, args.stage, update)
        if update == 1:
            before_rng = torch.cuda.get_rng_state(device)
            repeat, _ = two_step_target(lambda x,t: predict(teacher,x,t), details['input'], *details['times'])
            valid = torch.tensor(int(torch.equal(repeat, details['target']) and
                torch.equal(before_rng, torch.cuda.get_rng_state(device))), device=device)
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
            if not valid.item():
                raise RuntimeError('Frozen teacher is not repeatable or consumed sampling RNG')
        finite = torch.isfinite(loss).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item(): raise FloatingPointError('Nonfinite progressive loss')
        loss.backward()
        grad = student.clip_grad_norm_(10.)
        finite = torch.isfinite(grad).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item(): raise FloatingPointError('Nonfinite student gradient')
        optimizer.step()
        values = torch.stack([loss.detach().float(), grad.detach().float()])
        dist.all_reduce(values)
        values /= world
        interval_counts[details['interval']] += 1
        if rank == 0:
            row = {'step': update, 'loss': values[0].item(), 'grad_norm': values[1].item(),
                'interval': details['interval'], 'times': details['times'],
                'input_stream_sha256': stream_digest.hexdigest(), 'elapsed_seconds': time.monotonic()-started}
            with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
            if update == 10:
                atomic_json(run/'step10_health.json', {**row, 'finite': True, 'teacher_repeat_check': 'pass',
                    'estimated_stage_seconds': row['elapsed_seconds'] / 10 * 300})
        del loss, details
        if update % 50 == 0: gc.collect()
    state = fsdp_state_dict(student)
    if rank == 0:
        directory = run/'checkpoint_model_000300'
        directory.mkdir()
        temp = directory/'.model.pt.tmp'
        torch.save({'step': 300, 'generator': state, 'experiment_metadata': metadata}, temp)
        path = directory/'model.pt'
        temp.replace(path)
        atomic_json(run/'training.done', {**metadata, 'checkpoint_sha256': sha256_file(path),
            'input_stream_sha256': stream_digest.hexdigest(), 'interval_counts': interval_counts,
            'elapsed_seconds': time.monotonic()-started})
    dist.barrier()
    base.env.close()
    dist.destroy_process_group()


if __name__ == '__main__': main()
