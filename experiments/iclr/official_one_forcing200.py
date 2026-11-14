"""One fresh official framewise DMD+GAN run: seed2027, 200 iterations.

Inference/scoring use run_official_baselines.py with --model one_forcing so the
Qwen manifest and per-video RNG protocol match the two public baseline runs.
"""
from experiments.iclr.official_frame_chunk import ONE_PROFILE, main


if __name__ == "__main__":
    main(profile=ONE_PROFILE)
