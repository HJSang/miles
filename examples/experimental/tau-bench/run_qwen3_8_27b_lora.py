"""tau-bench RL on the TITO session path: Qwen3.8-27B, native LoRA on Megatron, fully async.

Trains tau-bench's tool-calling agent on retail/train and evaluates it on airline/test (a domain
it never trains on) and retail/dev. One node: the actor holds the base model plus the adapter on
``--actor-num-gpus`` GPUs (TP = that count); the other GPUs run one SGLang engine each, serving
both the policy (with the live adapter, through the session server) and the user simulator (the
frozen base, through the router miles hands to the agent). Score centering corrects the
off-policy drift of the asynchronous rollouts.

Needs tau-bench with its injectable model call (https://github.com/JD-ETH/tau-bench/pull/1)
installed in the training image. ``prepare`` downloads the model, converts it to torch_dist and
writes the task rows; ``execute`` submits the job.

Args: see ``ScriptArgs``; every field is a ``--flag``.

    python examples/experimental/tau-bench/run_qwen3_8_27b_lora.py
    python examples/experimental/tau-bench/run_qwen3_8_27b_lora.py --reasoning-effort medium --max-turns 20
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: str = "Qwen3.8-27B"
    megatron_model_type: str = "qwen3.8-27B"
    num_gpus_per_node: int = 8
    # TP = actor GPUs: one data-parallel rank
    actor_num_gpus: int = 2
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    lora_rank: int = 32
    lora_alpha: int = 32
    lr: float = 1e-5

    # thinking under a tight per-turn budget: the reward favors short reasoning
    reasoning_effort: Literal["low", "medium", "xhigh"] = "low"
    rollout_max_response_len: int = 1024
    max_seq_len: int = 32768
    max_turns: int = 30
    # reasoning included
    user_simulator_max_tokens: int = 32768

    num_rollout: int = 200
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    eval_interval: int = 10
    extra_args: str = ""

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def tau_data_dir(self) -> str:
        return f"{self.data_dir}/tau-bench"

    @property
    def example_dir(self) -> str:
        return f"{U.repo_base_dir}/examples/experimental/tau-bench"


def prepare(args: ScriptArgs):
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.model_dir} {args.tau_data_dir}")
    backend.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.hf_checkpoint}")
    backend.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )
    backend.exec_command_cpu(f"python {args.example_dir}/tau1_mock.py --local_dir {args.tau_data_dir}")


def execute(args: ScriptArgs):
    backend = args.create_backend()
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--load {args.model_dir}/{args.model_name}_torch_dist "
        "--megatron-to-hf-mode raw --no-load-optim --no-load-rng --finetune "
        f"--save {args.output_dir}/{args.run_id}/checkpoints "
        "--save-interval 50 "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        "--lora-dropout 0.0 "
        "--no-gradient-accumulation-fusion "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path tau_agent.run "
        "--custom-rm-path tau_agent.reward_func "
        "--use-session-server "
        "--tito-model qwen38small "
        f"--max-seq-len {args.max_seq_len} "
        f'--apply-chat-template-kwargs \'{{"reasoning_effort":"{args.reasoning_effort}"}}\' '
    )

    rollout_args = (
        f"--prompt-data {args.tau_data_dir}/retail_train_tasks.jsonl "
        "--input-key prompt "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 1.0 "
        "--rollout-top-k -1 "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--balance-data "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
    )

    # generation runs up to five batches ahead; groups more than four weight versions old are dropped
    async_args = "--fully-async --max-weight-staleness 4 --async-data-buffer-capacity-factor 5 "

    eval_args = (
        f"--eval-interval {args.eval_interval} "
        f"--eval-prompt-data airline-test {args.tau_data_dir}/airline_test_tasks.jsonl "
        f"retail-dev {args.tau_data_dir}/retail_dev_tasks.jsonl "
        "--n-samples-per-eval-prompt 1 "
        f"--eval-max-response-len {args.rollout_max_response_len} "
        "--eval-temperature 0 "
    )

    # Score centering (arXiv:2609.20807): group-centered rewards with the sampler's top-k
    # candidates recorded on every policy turn; it replaces PPO clipping.
    algo_args = (
        "--advantage-estimator grpo "
        "--loss-type score_centering "
        "--rollout-top-logprobs-num 128 "
        "--score-centering-is none "
        "--use-rollout-logprobs "
        "--disable-grpo-std-normalization "
        "--calculate-per-token-loss "
        "--entropy-coef 0.00 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.actor_num_gpus} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.8 "
        f"--sglang-max-lora-rank {args.lora_rank} "
        "--sglang-lora-backend triton "
        "--sglang-tool-call-parser qwen3_coder "
        "--sglang-reasoning-parser qwen3 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_gpus_per_node - args.actor_num_gpus} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{agent_args} "
        f"{rollout_args} "
        f"{async_args} "
        f"{eval_args} "
        f"{algo_args} "
        f"{optimizer_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    backend.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars={
            # tau_agent is imported by module name in the rollout's agent processes
            "PYTHONPATH": f"{args.megatron_path}:{args.example_dir}",
            "TAU_USER_SIMULATOR_MAX_TOKENS": str(args.user_simulator_max_tokens),
            "TAU_MAX_TURNS": str(args.max_turns),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
