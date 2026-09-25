"""``--sglang-config`` accepts an inline ``base64:`` payload, like the other file arguments."""

import base64
from argparse import Namespace

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX


def _make_args(sglang_config: str) -> Namespace:
    return Namespace(
        sglang_config=sglang_config,
        prefill_num_servers=None,
        rollout_num_gpus=6,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        hf_checkpoint="/ckpt/student",
        offload_rollout=False,
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        critic_train_only=False,
        starts_inference_engines=True,
    )


def test_inline_config_declares_a_frozen_sampler_and_a_frozen_teacher():
    yaml_text = "sglang:\n  - name: default\n    model_path: /ckpt/teacher\n    update_weights: false\n    server_groups:\n      - worker_type: regular\n        num_gpus: 4\n  - name: teacher\n    model_path: /ckpt/teacher\n    update_weights: false\n    server_groups:\n      - worker_type: regular\n        num_gpus: 2\n"
    inline = PSEUDO_FILE_PREFIX + base64.b64encode(yaml_text.encode()).decode()

    config = resolve_sglang_config(_make_args(inline))

    assert [m.name for m in config.models] == ["default", "teacher"]
    assert all(m.update_weights is False for m in config.models)
    assert {g.model_path for m in config.models for g in m.server_groups} == {"/ckpt/teacher"}
