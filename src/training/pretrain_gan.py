"""Deprecated alias for the proposal-compliant Stage B warm-up.

GAN/discriminator training is no longer part of the main method because the
proposal reserves it for an optional ablation.
"""
from training.pretrain_enhancer import main


if __name__ == "__main__":
    print("[DEPRECATED] Use: python -m training.pretrain_enhancer")
    main()
