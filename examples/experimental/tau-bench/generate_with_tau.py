"""
Tau-Bench Integration for miles Training

This module provides the main interface for training agents in tau-bench environments
using the miles framework. It handles agent-environment interactions and converts
results to the format expected by miles's training pipeline.
"""

import logging
import os
from dataclasses import replace

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)
# litellm logs two INFO lines per user-simulator call; keep the rollout log readable.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# User simulator configuration.
# tau-bench drives the user simulator through litellm, so any litellm-supported
# provider works. Pick the provider/model with the env vars below and supply the
# matching key through that provider's own *_API_KEY env var:
#   Gemini  : TAU_USER_MODEL_PROVIDER=gemini   TAU_USER_MODEL=gemini-2.5-flash-lite  GEMINI_API_KEY=...
#   DeepSeek: TAU_USER_MODEL_PROVIDER=deepseek TAU_USER_MODEL=deepseek-chat          DEEPSEEK_API_KEY=...
# Defaults preserve the original Gemini behavior.
USER_MODEL_PROVIDER = os.environ.get("TAU_USER_MODEL_PROVIDER", "gemini")
USER_MODEL = os.environ.get("TAU_USER_MODEL", "gemini-2.5-flash-lite")

# litellm reads the provider's own *_API_KEY env var; just confirm it is set so
# the run fails fast with a clear message instead of deep inside a rollout.
_PROVIDER_KEY_ENV = {"gemini": "GEMINI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
_key_env = _PROVIDER_KEY_ENV.get(USER_MODEL_PROVIDER, f"{USER_MODEL_PROVIDER.upper()}_API_KEY")
if not os.environ.get(_key_env):
    raise RuntimeError(
        f"tau-bench user simulator needs {_key_env} set for provider "
        f"'{USER_MODEL_PROVIDER}'. Export it before launching the run."
    )

# Tau-bench configuration
TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"]
    "user_model": USER_MODEL,  # Cheap model for the user simulator
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router",  # Unused, required
    "model": "qwen3-4b",  # Unused, required
    "user_model_provider": USER_MODEL_PROVIDER,
}
tau_config = RunConfig(**TAU_CONFIGS)


def res_to_sample(res: InteractionResult, sample: Sample) -> Sample:
    """The prompt sample miles handed us, filled with the interaction.

    Its identity fields (index, group_index, rollout_id, ...) are what miles uses to tell the
    siblings of a group apart, so they are kept; the input sample itself is left untouched.
    """
    status = {
        Status.COMPLETED: Sample.Status.COMPLETED,
        Status.TRUNCATED: Sample.Status.TRUNCATED,
        Status.ABORTED: Sample.Status.ABORTED,
    }[res.status]
    return replace(
        sample,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        response_length=res.response_length,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata={**sample.metadata, **res.info},
        rollout_log_probs=res.rollout_log_probs,
        rollout_topk_token_ids=res.rollout_topk_token_ids,
        rollout_topk_log_probs=res.rollout_topk_log_probs,
        weight_versions=res.weight_versions,
    )


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """
    Generate a complete agent-environment interaction trajectory for tau-bench.

    This is the main entry point for miles training. It creates a tau-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to miles's Sample format for training.

    Args:
        input: Rollout state, sample containing the task index, and sampling parameters

    Returns:
        GenerateFnOutput containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    args, sample, sampling_params = input.args, input.sample, input.sampling_params

    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)

    # The dataset row names the tau-bench domain and split the task comes from (see
    # tau1_mock.py), so a run can train on one domain and evaluate on another. Rows
    # without them fall back to the configured defaults.
    env_name = sample.metadata.get("env", tau_config.env)
    task_split = sample.metadata.get("task_split", tau_config.task_split)
    logger.info(f"Starting agent-environment interaction for {env_name}/{task_split} task {task_index}")

    # Initialize tau-bench environment
    env = get_env(
        env_name=env_name,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=task_split,
        task_index=task_index,
    )

    # Create trainable agent
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute agent-environment interaction
    # Note: The sample.prompt field contains the task index for repeatability
    interaction_result = await agent.asolve(
        env, agent.rollout_args, agent.sampling_params, task_index, state=input.state, evaluation=input.evaluation
    )

    # Convert to miles Sample format
    result_sample = res_to_sample(interaction_result, sample)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return GenerateFnOutput(samples=result_sample)
