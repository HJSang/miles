"""Score-centered on-policy distillation (SC-OPD) of Qwen3-1.7B from a Qwen3-8B teacher.

Trains the student on DAPO-17K prompts with the pure distillation loss derived in
``docs/advanced/score-centered-opd.md``. The rollout engines serve a FROZEN sampler
declared through ``--sglang-config`` (``update_weights: false``): either the teacher,
whose rollout log-probs double as the teacher log-probs, or the initial student, with
a second frozen teacher model served in the same job for scoring. The trainer
optimizes the score-centered reverse-KL to the teacher; the task reward is never
trained on and is only logged. A dedicated eval fleet scores HF snapshots of the
student on AIME 2024, AIME 2025 and MATH-500 with the boxed grader.

``--sampler student --no-freeze-sampler`` recovers the stale on-policy variant, where
the sampler is re-synced to the trainer every ``--refresh-interval`` rollouts (K).

Requires Megatron-LM at ``--megatron-path`` (the eval fleet needs Megatron HF export),
one 8-GPU node by default, and the models and datasets that ``prepare`` downloads
under ``--model-dir`` / ``--data-dir``.

Args:
    sampler: Which frozen model generates rollouts. ``teacher`` reads the teacher
        log-probs straight from the rollout engine; ``student`` serves a second frozen
        teacher model for scoring.
    freeze_sampler: Keep the sampler frozen for the whole run. Only ``--sampler
        student`` may be unfrozen; then ``refresh_interval`` applies.
    refresh_interval: K, the sampler refresh interval in rollout steps (unfrozen only).
    actor_num_gpus / sampler_num_gpus / teacher_num_gpus / eval_num_gpus: GPU layout.
        ``teacher_num_gpus`` is only used with ``--sampler student``.
    score_centering_top_k: Sampler top-k log-probs kept for score centering (paper: 128).
    opd_kl_coef: Beta, the distillation coefficient.
    enable_mis: Compose score centering with masked importance sampling (paper's MIS+SC).
    enable_thinking: Qwen3 thinking mode for rollouts and eval (off by default).
    monitor_rm_type: Built-in reward logged on training rollouts; never trained on.

Examples:
    python scripts/run_qwen3_1_7b_sc_opd.py --sampler teacher
    python scripts/run_qwen3_1_7b_sc_opd.py --sampler student --no-freeze-sampler --refresh-interval 64
"""

import json
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_EVAL_SOURCES = {
    "aime24": "aime-2024/aime-2024.jsonl",
    "aime25": "aime-2025/aime-2025.jsonl",
    "math500": "MATH-500/test.jsonl",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-1.7B"
    teacher_model_name: str = "Qwen3-8B"
    megatron_model_type: str = "qwen3-1.7B"
    hardware: Literal["auto", "H100", "H200", "B200", "GB200", "GB300"] = "auto"
    num_gpus_per_node: int | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    sampler: Literal["teacher", "student"] = "teacher"
    freeze_sampler: bool = True
    refresh_interval: int = 1
    actor_num_gpus: int = 2
    sampler_num_gpus: int | None = None
    teacher_num_gpus: int = 2
    eval_num_gpus: int = 2

    score_centering_top_k: int = 128
    opd_kl_coef: float = 1.0
    enable_mis: bool = False
    enable_thinking: bool = False
    monitor_rm_type: str = "math"

    num_rollout: int = 300
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 8192
    global_batch_size: int = 256
    eval_interval: int = 20
    eval_max_response_len: int = 16384
    aime_n_samples: int = 16
    math500_n_samples: int = 2

    def __post_init__(self):
        self.hardware = U.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.sampler_num_gpus is None:
            self.sampler_num_gpus = 4 if self.sampler == "teacher" else 2
        if self.sampler == "teacher" and not self.freeze_sampler:
            raise ValueError("The teacher sampler is always frozen; --no-freeze-sampler needs --sampler student.")
        if self.refresh_interval < 1:
            raise ValueError("--refresh-interval must be at least 1.")

        total = self.actor_num_gpus + self.rollout_num_gpus + self.eval_num_gpus
        available = self.num_nodes * self.num_gpus_per_node
        if total > available:
            raise ValueError(
                f"GPU layout needs {total} GPUs (actor {self.actor_num_gpus} + rollout {self.rollout_num_gpus} + eval {self.eval_num_gpus}) but {available} are available."
            )

    @property
    def scoring_teacher_num_gpus(self) -> int:
        return self.teacher_num_gpus if self.sampler == "student" else 0

    @property
    def rollout_num_gpus(self) -> int:
        return self.sampler_num_gpus + self.scoring_teacher_num_gpus

    @property
    def student_hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def teacher_hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.teacher_model_name}"

    @property
    def sampler_hf_checkpoint(self) -> str:
        return self.teacher_hf_checkpoint if self.sampler == "teacher" else self.student_hf_checkpoint

    @property
    def eval_data_dir(self) -> str:
        return f"{self.data_dir}/sc_opd_eval"


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.student_hf_checkpoint}")
    U.exec_command_cpu(f"hf download Qwen/{args.teacher_model_name} --local-dir {args.teacher_hf_checkpoint}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2025", data_dir=args.data_dir)
    U.hf_download_dataset("HuggingFaceH4/MATH-500", data_dir=args.data_dir)

    builder = f"{U.repo_base_dir}/scripts/tools/build_boxed_math_eval.py"
    for name, relative_src in _EVAL_SOURCES.items():
        U.exec_command_cpu(
            f"python3 {builder} --src {args.data_dir}/{relative_src} --dst {args.eval_data_dir}/{name}.jsonl"
        )

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.student_hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def _sglang_config_text(args: ScriptArgs) -> str:
    """One router per model: the frozen sampler, the scoring teacher (student sampler only), the eval fleet."""
    models = [
        {
            "name": "default",
            "model_path": args.sampler_hf_checkpoint,
            "update_weights": not args.freeze_sampler,
            "server_groups": [{"worker_type": "regular", "num_gpus": args.sampler_num_gpus, "num_gpus_per_engine": 1}],
        }
    ]
    if args.sampler == "student":
        models.append(
            {
                "name": "teacher",
                "model_path": args.teacher_hf_checkpoint,
                "update_weights": False,
                "server_groups": [
                    {"worker_type": "regular", "num_gpus": args.teacher_num_gpus, "num_gpus_per_engine": 1}
                ],
            }
        )
    if args.eval_num_gpus > 0:
        models.append(
            {
                "name": "eval",
                "server_groups": [
                    {"worker_type": "regular", "num_gpus": args.eval_num_gpus, "num_gpus_per_engine": 1}
                ],
            }
        )
    return json.dumps({"sglang": models}, indent=2)


def _eval_config_text(args: ScriptArgs) -> str:
    n_samples = {"aime24": args.aime_n_samples, "aime25": args.aime_n_samples, "math500": args.math500_n_samples}
    datasets = [
        {
            "name": name,
            "path": f"{args.eval_data_dir}/{name}.jsonl",
            "rm_type": "math",
            "n_samples_per_eval_prompt": n_samples[name],
        }
        for name in _EVAL_SOURCES
    ]
    config = {
        "eval": {
            "defaults": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_response_len": args.eval_max_response_len,
            },
            "datasets": datasets,
        }
    }
    return json.dumps(config, indent=2)


def _mis_config_text() -> str:
    return "\n".join(
        [
            "use_tis: true",
            "use_rs: false",
            "tis_level: token",
            "tis_mode: mask",
            "tis_lower_bound: 0.5",
            "tis_upper_bound: 5.0",
            "tis_batch_normalize: false",
            "rs_lower_bound: null",
            "rs_upper_bound: null",
            "rs_veto_threshold: null",
        ]
    )


def execute(args: ScriptArgs):
    chat_template_kwargs = json.dumps({"enable_thinking": args.enable_thinking})

    ckpt_args = f"--hf-checkpoint {args.student_hf_checkpoint} --load {args.model_dir}/{args.model_name}_torch_dist --save {args.output_dir}/{args.run_id}/checkpoints --save-interval 50 "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        f"--apply-chat-template-kwargs '{chat_template_kwargs}' "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    # Pure distillation: the teacher log-probs are the only training signal, the boxed
    # grader is logged for monitoring, and eval datasets grade with the boxed grader.
    opd_args = f"--custom-rm-path miles.rollout.on_policy_distillation.reward_func --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards --use-opd --opd-type sglang --opd-kl-coef {args.opd_kl_coef} --opd-log-prob-top-k 0 --opd-monitor-rm-type {args.monitor_rm_type} "
    if args.sampler == "teacher":
        opd_args += "--opd-teacher-from-rollout-logprobs "
    else:
        opd_args += "--opd-teacher-model teacher "

    score_centering_args = f"--use-score-centering --score-centering-top-k {args.score_centering_top_k} "
    if args.enable_mis:
        score_centering_args += f"--use-tis --custom-config-path {U.encode_pseudo_file(_mis_config_text())} --custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "

    sampler_args = f"--sglang-config {U.encode_pseudo_file(_sglang_config_text(args))} "
    if not args.freeze_sampler:
        sampler_args += f"--update-weights-interval {args.refresh_interval} "

    algo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    eval_args = f"--eval-interval {args.eval_interval} --eval-config {U.encode_pseudo_file(_eval_config_text(args))} --eval-num-gpus {args.eval_num_gpus} --eval-num-gpus-per-engine 1 --eval-hf-dir /dev/shm/{args.run_id}/eval_hf "

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    perf_args = "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu 16384 "

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-chunked-prefill-size 4096 "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    train_args = f"{ckpt_args} {rollout_args} {opd_args} {score_centering_args} {sampler_args} {algo_args} {eval_args} {optimizer_args} {perf_args} {sglang_args} {misc_args} {U.get_default_wandb_args(__file__, run_id=args.run_id)} {args.extra_args} "

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
