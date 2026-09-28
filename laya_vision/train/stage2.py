"""Stage 2 (decisions): train projector + Laya encoder + head on image tasks mixed with text rows.

    python -m laya_vision.train.stage2 --config configs/stage2_a.yaml [--override init_from=runs/stage1_a/final ...]

Options (config ``train.*``): ``unfreeze_vision_blocks: N`` with ``lr.vision`` (1e-6) trains the
last N tower blocks; ``lora.enabled: true`` trains LoRA adapters on the encoder instead of all its
weights (adapters are merged before every save, so checkpoints stay plain Laya state dicts);
``early_stop_text_drop: 0.02`` stops when held-out text accuracy drops more than 2 points below
its value at step 0.
"""
from .loop import cli

STAGE2_DEFAULTS = {
    "output_dir": "runs/stage2",
    "train": {"lr": {"projector": 1e-4, "encoder": 2.5e-5, "head": 1e-4, "vision": 0.0},
              "weight_decay": 0.01, "epochs": 3, "early_stop_text_drop": 0.02},
}


def main(argv=None):
    return cli(2, STAGE2_DEFAULTS, argv)


if __name__ == "__main__":
    main()
