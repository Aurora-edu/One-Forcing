"""Official-code DMD+GAN vs DMD-only: seed42, 300 iterations, raw/native FFE.

The external launcher shares the source/asset/export audits with the existing
official frame/chunk experiment. The official checkout itself is never edited.
"""
from experiments.iclr.official_frame_chunk import GAN_PROFILE, main


if __name__ == "__main__":
    main(profile=GAN_PROFILE)
