import json
import os

import pytest


@pytest.fixture(scope="session")
def tiny_laya_dir(tmp_path_factory):
    """A random tiny Laya checkpoint (real tokenizer, 128-wide 3-layer ModernBERT)."""
    from laya_vision.testing import make_tiny_laya_dir

    return make_tiny_laya_dir(str(tmp_path_factory.mktemp("tiny_laya")))


@pytest.fixture(scope="session")
def laya_cfg(tiny_laya_dir):
    with open(os.path.join(tiny_laya_dir, "rl_agent_config.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def tok(tiny_laya_dir, laya_cfg):
    from laya.agent import _load_tokenizer

    return _load_tokenizer(os.path.join(tiny_laya_dir, "tokenizer"), laya_cfg)


@pytest.fixture(scope="session")
def tiny_siglip_config():
    from laya_vision.testing import tiny_siglip_config as make

    return make()
