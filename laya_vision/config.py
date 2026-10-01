"""Vision-side configuration, stored under ``cfg["vision"]`` in ``rl_agent_config.json``."""
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

POOL_MODES = ("avg", "shuffle")


def pooled_grid(grid: int, k: int, mode: str) -> int:
    """Side length of the token grid after pooling a ``grid x grid`` patch grid with factor ``k``."""
    if mode not in POOL_MODES:
        raise ValueError("pool_mode must be one of %s, got %r" % (POOL_MODES, mode))
    if k < 1:
        raise ValueError("pool_k must be >= 1")
    # avg uses ceil_mode; shuffle pads the grid up to a multiple of k. Both give ceil(grid / k).
    return math.ceil(grid / k)


@dataclass
class VisionConfig:
    tower: str = "google/siglip-base-patch16-224"
    tower_revision: Optional[str] = None
    # -1 = last_hidden_state (post-LN in HF SigLIP), -2 = second-to-last layer output (LLaVA convention)
    feature_layer: int = -2
    pool_mode: str = "avg"
    pool_k: int = 2
    # Filled from the tower config when the tower is built; kept here so a checkpoint is self-describing.
    image_size: int = 224
    patch_size: int = 16
    vision_width: int = 768
    tower_trained: bool = False
    # Projector guards against modality collapse (see projector.RunningStandardize). Off by default
    # so checkpoints written before they existed load unchanged; configs/stage1_a.yaml turns them on.
    proj_in_norm: bool = False
    proj_standardize: bool = False
    # Image-modality temperatures, same layout as Laya's `temperature` / `temperature_by_options`.
    temperature: Optional[List[float]] = None
    temperature_by_options: Dict[str, float] = field(default_factory=dict)
    laya_base: Optional[str] = None
    laya_version: Optional[str] = None

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def pooled_grid(self) -> int:
        return pooled_grid(self.grid, self.pool_k, self.pool_mode)

    @property
    def n_tokens(self) -> int:
        return self.pooled_grid ** 2

    @property
    def pooled_width(self) -> int:
        return self.vision_width * (self.pool_k ** 2 if self.pool_mode == "shuffle" else 1)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["n_tokens"] = self.n_tokens  # informational; recomputed on load
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisionConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
