"""tau-bench episodes on miles' TITO session path.

miles calls ``run`` once per episode (``--custom-agent-function-path tau_agent.run``). tau-bench's
own tool-calling agent runs the episode and speaks OpenAI chat to the session endpoint, so the
session server owns the token ids, log-probs and adapter selection of every policy turn. The user
simulator is the frozen base model on the same rollout engines, reached through the router, where
no adapter is applied.

The simulator reaches the engines through ``metadata["inference_router_url"]`` with
``metadata["inference_router_headers"]``, which miles hands to every agent call.

Env vars, read in the rollout's agent processes (the launcher forwards them):
  TAU_USER_SIMULATOR_MAX_TOKENS  new-token cap for one simulated user reply, reasoning included (default 32768)
  TAU_MAX_TURNS                  policy turns per episode (default 30, tau-bench's own limit)
"""

import asyncio
import inspect
import logging
import os
from typing import Any

from openai import OpenAI
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.chat import ChatFn, ChatResult
from tau_bench.envs import get_env
from tau_bench.envs.user import LLMUserSimulationEnv
from tau_bench.types import SolveResult

from miles.rollout.agentic.session import openai_session_url
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def openai_chat(client: OpenAI, **request_kwargs: Any) -> ChatFn:
    """tau-bench's model call over an OpenAI-compatible endpoint, returning the turn as generated.

    ``request_kwargs`` are chat-completion request fields. The SDK takes the OpenAI ones as keyword
    arguments; SGLang extensions such as ``top_k`` go in the request body through ``extra_body``.
    """
    create = client.chat.completions.create
    sdk_params = inspect.signature(create).parameters

    def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResult:
        kwargs = {**request_kwargs, "tools": tools} if tools else dict(request_kwargs)
        extra_body = {
            **kwargs.pop("extra_body", {}),
            **{k: kwargs.pop(k) for k in list(kwargs) if k not in sdk_params},
        }
        choice = create(model="default", messages=messages, extra_body=extra_body, **kwargs).choices[0]
        return ChatResult(message=choice.message.model_dump(exclude_none=True), finish_reason=choice.finish_reason)

    return chat


def require_reply(chat: ChatFn, attempts: int = 3) -> ChatFn:
    """``chat`` re-run while the model returns no reply content (thinking only, or cut off): a user
    turn must say something. Fails after ``attempts``."""

    def chat_requiring_reply(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResult:
        for attempt in range(1, attempts + 1):
            result = chat(messages, tools)
            if result.message.get("content"):
                return result
            logger.warning(f"empty reply (finish_reason={result.finish_reason}), attempt {attempt}/{attempts}")
        raise RuntimeError(f"no reply content after {attempts} attempts (last finish_reason={result.finish_reason})")

    return chat_requiring_reply


def _solve(base_url: str, request_kwargs: dict[str, Any], metadata: dict[str, Any]) -> SolveResult:
    # The simulator thinks at low effort; the engines' reasoning parser keeps that out of the reply.
    simulator = LLMUserSimulationEnv(
        chat=require_reply(
            openai_chat(
                OpenAI(
                    base_url=f"{metadata['inference_router_url']}/v1",
                    api_key="miles",
                    default_headers=metadata["inference_router_headers"],
                ),
                max_tokens=int(os.environ.get("TAU_USER_SIMULATOR_MAX_TOKENS", "32768")),
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}},
            )
        )
    )
    env = get_env(
        env_name=metadata["env"],
        user_strategy="llm",
        user_model=None,
        task_split=metadata["task_split"],
        task_index=metadata["task_index"],
        user=simulator,
    )
    agent = ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        chat=openai_chat(OpenAI(base_url=openai_session_url(base_url), api_key="miles"), **request_kwargs),
    )
    return agent.solve(
        env, task_index=metadata["task_index"], max_num_steps=int(os.environ.get("TAU_MAX_TURNS", "30"))
    )


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one tau-bench episode against the session; the result's reward lands in the sample metadata."""
    # tau-bench's loop and simulator calls are synchronous; a thread keeps the agent process's loop free.
    result = await asyncio.to_thread(_solve, base_url, request_kwargs, metadata)
    turns = sum(message["role"] == "assistant" for message in result.messages)
    logger.info(
        f"tau-bench {metadata['env']}/{metadata['task_split']} task {metadata['task_index']}: "
        f"reward={result.reward} turns={turns}"
    )
    return {"reward": result.reward, "tau_turns": turns}


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """The episode reward recorded by ``run``; one value per sample.

    An episode whose agent call failed (miles logs the traceback) has no reward: it scores 0 so the
    run goes on, and the warning here makes the failure visible per session.
    """
    if isinstance(samples, list):
        return [await reward_func(args, sample) for sample in samples]
    if "reward" not in samples.metadata:
        logger.warning(f"session {samples.metadata.get('session_id')}: no reward, the episode did not finish")
        return 0.0
    return samples.metadata["reward"]
