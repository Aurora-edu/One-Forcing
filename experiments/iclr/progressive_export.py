"""Deterministic progressive previews and original-prompt all1 export."""
import argparse
import gc
import os
from pathlib import Path

from experiments.iclr.progressive_protocol import (ARMS, check, checkpoint_for,
    raw_state, validate_stage_checkpoint)
from experiments.iclr.progressive_core import sample_latents
from experiments.iclr.protocol import atomic_json, sha256_file


def cases_for(m, phase):
    return m['vbench'] if phase == 'final' else m['holdout']


def modes_for(phase):
    if phase == 'source':
        return [('deterministic4', 4, False), ('legacy4', 4, True)]
    return [('all2', 2, False)] if phase == '4to2' else [('all1', 1, False)]


def video_path(root, phase, arm, mode, case):
    if phase == 'final':
        return Path(root)/'final'/arm/'videos'/case['filename']
    return Path(root)/'previews'/phase/arm/mode/(case['id']+'.mp4')


def validate_video(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        shapes = [(frame.width, frame.height) for frame in container.decode(stream)]
    if len(shapes) != 81 or set(shapes) != {(832,480)} or abs(fps-16) > 1e-3:
        raise ValueError(f'Invalid encoded video: {path}, frames={len(shapes)}, fps={fps}')
    return {'frames': 81, 'fps': fps, 'width': 832, 'height': 480}


def write_video(path, pixels, preview):
    import av
    from PIL import Image, ImageDraw
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    arrays = pixels.clamp(0,255).to(__import__('torch').uint8).cpu().numpy()
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=16)
        stream.width, stream.height, stream.pix_fmt = 832,480,'yuv420p'
        stream.options = {'crf': '18'}
        for array in arrays:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    if preview:
        canvas = Image.new('RGB', (4*416,3*260))
        draw = ImageDraw.Draw(canvas)
        for slot in range(12):
            index = round(slot*80/11)
            x,y = (slot%4)*416, (slot//4)*260
            canvas.paste(Image.fromarray(arrays[index]).resize((416,240)), (x,y))
            draw.text((x+4,y+242), f'frame {index}, {index/16:.2f}s', fill='white')
        canvas.save(path.with_suffix('.jpg'))
    return validate_video(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', required=True, choices=('source','4to2','2to1','final'))
    args = parser.parse_args()
    if not os.environ.get('TMUX') or os.environ.get('WORLD_SIZE') != '8':
        raise RuntimeError('Use the independent tmux launcher')
    import torch
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    root = Path(args.output).resolve()
    m = check(root)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    stage = '2to1' if args.phase == 'final' else args.phase
    records, checkpoints = [], {}
    for arm in ARMS:
        path = checkpoint_for(root, arm, stage)
        digest = sha256_file(path)
        if stage == 'source':
            if digest != m['sources']['arms'][arm]['sha256']: raise ValueError('Source changed')
            payload, state = raw_state(path)
            del payload
        else:
            previous = 'source' if stage == '4to2' else '4to2'
            state = validate_stage_checkpoint(path, m, arm, stage, sha256_file(checkpoint_for(root,arm,previous)))
        config = OmegaConf.create({'denoising_step_list': [1000], 'warp_denoising_step': False,
            'num_frame_per_block': 1, 'independent_first_frame': False, 'context_noise': 0,
            'model_kwargs': m['config']['model_kwargs']})
        pipeline = CausalInferencePipeline(config, device=device)
        pipeline.generator.load_state_dict(state, strict=True)
        del state
        pipeline.eval().requires_grad_(False).to(device=device,dtype=torch.bfloat16)
        checkpoints[arm] = {'path': str(path), 'sha256': digest, 'load': 'strict_raw_generator'}
        for mode, steps, legacy in modes_for(args.phase):
            for case in cases_for(m,args.phase)[rank::world]:
                destination = video_path(root,args.phase,arm,mode,case)
                if destination.exists(): raise FileExistsError(destination)
                torch.manual_seed(case['seed'])
                rng = torch.Generator(device=device).manual_seed(case['seed'])
                noise = torch.randn((1,21,16,60,104),device=device,dtype=torch.bfloat16,generator=rng)
                with torch.no_grad():
                    latents = sample_latents(pipeline,noise,[case['prompt']],steps,legacy_renoise=legacy)
                    if not torch.isfinite(latents).all(): raise FloatingPointError('Invalid generated latents')
                    pixels = pipeline.vae.decode_to_pixel(latents,use_cache=False)
                    if not torch.isfinite(pixels).all(): raise FloatingPointError('Invalid decoded pixels')
                    pixels = ((pixels[0].float()*.5+.5).clamp(0,1)*255).permute(0,2,3,1).cpu()
                metadata = write_video(destination,pixels,args.phase != 'final')
                records.append({'path': str(destination), 'sha256': sha256_file(destination),
                    'arm': arm, 'mode': mode, 'case': case, 'steps_every_frame': steps,
                    'legacy_renoise': legacy, 'ffe': False, 'use_ema': False, **metadata})
                print(f'EXPORTED {destination}',flush=True)
                pipeline.vae.model.clear_cache()
                del pixels, latents, noise
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    folder = root/'final' if args.phase == 'final' else root/'previews'/args.phase
    atomic_json(folder/f'rank{rank}.json', {'rank': rank, 'world': world, 'phase': args.phase,
        'manifest_sha256': sha256_file(root/'manifest.json'), 'checkpoints': checkpoints,
        'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(device)}, 'videos': records})


if __name__ == '__main__': main()
