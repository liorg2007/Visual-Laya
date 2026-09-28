"""Stage 1 (alignment): train pooler + projector + modality embedding; Laya and the tower stay frozen.

    python -m laya_vision.train.stage1 --config configs/stage1_a.yaml [--override train.max_steps=100 ...]

Gradients still flow *through* the frozen Laya encoder to the projector (``detach_encoder: false``).
"""
from .loop import cli

STAGE1_DEFAULTS = {
    "output_dir": "runs/stage1",
    "train": {"lr": {"projector": 1e-3, "encoder": 0.0, "head": 0.0, "vision": 0.0},
              "weight_decay": 0.0, "epochs": 1, "best_metric": "image.accuracy"},
}


def main(argv=None):
    return cli(1, STAGE1_DEFAULTS, argv)


if __name__ == "__main__":
    main()
