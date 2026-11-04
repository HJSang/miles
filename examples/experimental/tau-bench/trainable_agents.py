import asyncio
import functools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any

import numpy as np
from openai_tool_adapter import create_openai_adapter
from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import Action, RunConfig
from transformers import AutoTokenizer

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
    pad_rollout_topk_logprobs,
)
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.http_utils import post
from miles.utils.types import WeightVersionsPerCall

# Set up logger for this module
logger = logging.getLogger(__name__)


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class InteractionResult:
    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    loss_mask: list[int] | None = None
    tokens: list[int] | None = None
    response_length: int = 0
    status: Status = Status.COMPLETED
    # Per response token, from the engine call that produced it: the sampled token's log-prob,
    # the top-k candidates (--rollout-top-logprobs-num) and the weight version. Observation
    # rows hold 0.0 / -1 / -inf.
    rollout_log_probs: list[float] | None = None
    rollout_topk_token_ids: np.ndarray | None = None
    rollout_topk_log_probs: np.ndarray | None = None
    weight_versions: list[WeightVersionsPerCall] = field(default_factory=list)


@dataclass
class _Trajectory:
    """The token sequence of one interaction, in exactly the order the engine consumed it."""

    prompt_token_ids: list[int]
    response_token_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)
    rollout_topk_token_ids: np.ndarray | None = None
    rollout_topk_log_probs: np.ndarray | None = None
    weight_versions: list[WeightVersionsPerCall] = field(default_factory=list)

    @property
    def tokens(self) -> list[int]:
        return self.prompt_token_ids + self.response_token_ids

    def add_generated(self, meta_info: dict[str, Any], *, topk: int, sampling_logprobs_mode: str) -> list[int]:
        """Append the engine's own output tokens with their sampled and candidate log-probs."""
        generated = meta_info.get("output_token_logprobs") or []
        token_ids = [token_id for _, token_id, *_ in generated]
        self.response_token_ids += token_ids
        self.rollout_log_probs += [log_prob for log_prob, *_ in generated]
        self.loss_mask += [1] * len(token_ids)
        if topk:
            append_rollout_topk_logprobs(self, meta_info, topk, sampling_logprobs_mode=sampling_logprobs_mode)
        self.weight_versions.append(WeightVersionsPerCall.from_meta_info(meta_info, output_end=len(self.tokens)))
        return token_ids

    def add_observation(self, token_ids: list[int]) -> None:
        """Append tokens the model only reads: a tool result or user reply and the next generation prompt."""
        self.response_token_ids += token_ids
        self.rollout_log_probs += [0.0] * len(token_ids)
        self.loss_mask += [0] * len(token_ids)
        pad_rollout_topk_logprobs(self, len(token_ids))


def call_to_action_sglang(calls: list[Any], text_response: str) -> Action:
    """
    Convert sglang response message to Action, similar to original message_to_action
    but adapted for sglang response format.
    """
    # Default action if no action was found.
    action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
    if calls:
        if len(calls) > 1:
            logger.debug("Multiple tool calls identified, only taking first.")
        tool_call = calls[0]
        params = json.loads(tool_call["parameters"])
        if not isinstance(params, dict):
            logger.warning(f"{params} does not follow dict structure for action")
        else:
            action = Action(name=tool_call["name"], kwargs=params)
    return action


TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)

# Qwen3-family chat templates render <think> blocks only for assistant turns after the latest
# user message and drop them from earlier turns. Re-rendering the history then shrinks a prefix
# the model has already consumed, which breaks the prefix-based accounting in _observation_tokens.
# Render every assistant turn the same way instead, so the rendered prefix only ever grows.
_REASONING_ONLY_AFTER_LAST_QUERY = "loop.index0 > ns.last_query_index"

# Thinking off keeps tau-bench turns short; set TAU_ENABLE_THINKING=1 to train with reasoning.
ENABLE_THINKING = os.environ.get("TAU_ENABLE_THINKING", "0") == "1"

# sglang tool-call parser matching the chat template's tool-call format: "qwen25" for the JSON
# <tool_call> format (Qwen2.5 / Qwen3), "qwen3_coder" for the XML <function=...> format (Qwen3.5).
TOOL_CALL_PARSER = os.environ.get("TAU_TOOL_CALL_PARSER", "qwen25")


def _stable_chat_template(tokenizer: AutoTokenizer) -> str:
    return tokenizer.chat_template.replace(_REASONING_ONLY_AFTER_LAST_QUERY, "true")


@functools.cache
def _env_executor(max_workers: int) -> ThreadPoolExecutor:
    """tau-bench steps its environment with blocking litellm calls to the user simulator. They run
    off the event loop, one worker per in-flight trajectory, so the other trajectories keep
    generating meanwhile."""
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tau-env")


class TrainableAgentMixin:
    """
    Mixin class that provides trainable agent functionality for tau-bench environments.

    This mixin extends the original tau-bench agent with async LLM interaction
    capabilities for reinforcement learning training using sglang servers.
    """

    def _reformulate_tool_call(self, text: str) -> str:
        """
        Reformulate tool call instruction for tau-bench environment.

        The default tool template assumes one or more function calls, but for
        tau-bench, at most one tool call or skip tool calls are the valid options.

        Args:
            text: Original tool instruction text

        Returns:
            Reformulated tool instruction text
        """
        return text.replace("You may call one or more functions to assist with the user query.", TOOL_INSTRUCTION)

    def _render(
        self,
        tokenizer: AutoTokenizer,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render messages with the chat template, consistently across turns (see _stable_chat_template)."""
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            chat_template=_stable_chat_template(tokenizer),
            enable_thinking=ENABLE_THINKING,
        )

    def _request_payload(
        self, input_ids: list[int], sampling_params: dict[str, Any], *, evaluation: bool
    ) -> dict[str, Any]:
        """A /generate request over the exact tokens trained on, returning the sampler's log-probs.

        Training requests also record the top-k candidates when --rollout-top-logprobs-num is set
        (needed by --loss-type score_centering); evaluation never collects them.
        """
        payload = {"input_ids": input_ids, "sampling_params": deepcopy(sampling_params), "return_logprob": True}
        if not evaluation:
            configure_rollout_topk_logprobs_request(self.rollout_args, payload)
        return payload

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make an LLM call tracking.

        Args:
            url: SGLang server URL
            payload: Request payload containing input_ids and sampling parameters

        Returns:
            LLM response from sglang server
        """
        return await post(url, payload)

    def _observation_tokens(
        self, tokenizer: AutoTokenizer, messages: list[dict[str, Any]], *, generated_eos: bool
    ) -> list[int]:
        """Tokens between the assistant turn the engine just produced and its next generation prompt.

        The chat template closes the assistant turn with ``eos_token + "\\n"`` after the trimmed
        content. The engine produced the content itself and, unless it hit the length limit, the
        eos token too, so only the rest is tokenized: the trained sequence stays exactly what the
        engine consumed instead of a re-tokenization of the rendered text.
        """
        prev = self._render(tokenizer, messages[:-1], add_generation_prompt=False)
        curr = self._render(tokenizer, messages, add_generation_prompt=True)
        turn_end = tokenizer.eos_token + "\n"
        assert prev.endswith(turn_end) and curr.startswith(prev), "unexpected assistant turn rendering"
        text = ("" if generated_eos else tokenizer.eos_token) + "\n" + curr[len(prev) :]
        return tokenizer.encode(text, add_special_tokens=False)

    def _parse_tool(self, response: str) -> dict[str, Any]:
        """
        Parse tool calls from LLM response string.

        Args:
            response: Raw response text from sglang

        Returns:
            Parsed tool call result in OpenAI format
        """
        return self.openai_adapter.parse_response_to_openai_format(response)

    async def _run_in_env(self, fn, *fn_args, **fn_kwargs):
        """Run a blocking environment call (tool execution or user-simulator turn) off the event loop."""
        args = self.rollout_args
        in_flight = args.async_max_concurrent_samples or args.rollout_batch_size * args.n_samples_per_prompt
        executor = _env_executor(in_flight)
        return await asyncio.get_running_loop().run_in_executor(executor, partial(fn, *fn_args, **fn_kwargs))

    async def _execute_tool(self, env, action: Action):
        """
        Execute a tool/action in the environment.

        Args:
            env: Tau-bench environment instance
            action: Action to execute

        Returns:
            Environment step result
        """
        return await self._run_in_env(env.step, action)

    async def _initialize_environment(self, env, task_index: int | None) -> tuple[str, dict[str, Any]]:
        """
        Initialize the environment and get initial observation.

        Args:
            env: Tau-bench environment instance
            task_index: Task index to reset to

        Returns:
            Tuple of (observation, info)
        """
        if task_index is not None:
            env_reset_res = await self._run_in_env(env.reset, task_index=task_index)
        else:
            env_reset_res = await self._run_in_env(env.reset)
        return env_reset_res.observation, env_reset_res.info.model_dump()

    def _build_initial_messages(self, obs: str) -> list[dict[str, Any]]:
        """
        Build initial conversation messages.

        Args:
            obs: Initial observation from environment

        Returns:
            List of initial messages
        """
        return [{"role": "system", "content": self.wiki}, {"role": "user", "content": obs}]

    def _prepare_prompt_tokens(self, state: GenerateState, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        """
        Prepare prompt text and tokenize it.

        Args:
            state: GenerateState instance with tokenizer
            messages: Conversation messages

        Returns:
            Tuple of (prompt_text, prompt_token_ids)
        """
        prompt_text = self._render(state.tokenizer, messages, add_generation_prompt=True, tools=self.tools_info)
        # Reformulate tool call instruction for tau-bench
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return prompt_text, prompt_token_ids

    async def asolve(
        self,
        env,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        task_index: int | None = None,
        max_num_steps: int = 30,
        *,
        state: GenerateState,
        evaluation: bool = False,
    ) -> InteractionResult:
        """
        Execute async agent-environment interaction for training.

        This method extends the original Agent to support async interaction with LLM
        server for reinforcement learning training. It maintains conversation history,
        tracks tokens, and records metadata for training purposes.

        Args:
            env: Tau-bench environment instance
            rollout_args: Rollout configuration arguments
            sampling_params: LLM sampling parameters
            task_index: Specific task index to solve (optional)
            max_num_steps: Maximum number of interaction steps
            evaluation: Whether this is an evaluation rollout (no candidate log-prob collection)

        Returns:
            InteractionResult containing the complete interaction trajectory
        """
        # Initialize environment and state
        url = f"http://{rollout_args.sglang_router_ip}:" f"{rollout_args.sglang_router_port}/generate"
        tokenizer = state.tokenizer
        topk = 0 if evaluation else rollout_args.rollout_top_logprobs_num

        # Get initial environment state
        obs, info = await self._initialize_environment(env, task_index)

        # Build initial conversation
        messages = self._build_initial_messages(obs)
        prompt_text, prompt_token_ids = self._prepare_prompt_tokens(state, messages)

        # Initialize tracking variables
        traj = _Trajectory(prompt_token_ids=prompt_token_ids)
        total_reward = 0.0

        # Initialize result
        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})

        # Multi-turn interaction loop
        for _ in range(max_num_steps):
            # Send the tokens accumulated so far to sglang
            payload = self._request_payload(traj.tokens, sampling_params, evaluation=evaluation)
            output = await self._call_llm(url, payload)

            # Check for abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                res.status = Status.ABORTED
                return self._build_final_result(res, total_reward, info, messages, traj)

            generated = traj.add_generated(
                output["meta_info"],
                topk=topk,
                sampling_logprobs_mode=payload.get("sampling_logprobs_mode", "selected"),
            )
            response = output["text"]
            # Remove end of conversation token if present
            if response.endswith("<|im_end|>"):
                response = response[:-10]

            # Parse tool calls using OpenAI adapter
            logger.debug(f"Using OpenAI adapter to parse response: {response[:100]}...")
            try:
                openai_result = self._parse_tool(response)
                logger.debug(f"OpenAI adapter result: success={openai_result['success']}")

                if not openai_result["success"]:
                    logger.warning(f"OpenAI adapter failed: {openai_result['error']}")
                    logger.warning(
                        f"rollout response: {response} can not be parsed into " f"tool calls {openai_result['error']}"
                    )
                    res.status = Status.ABORTED
                    return self._build_final_result(res, total_reward, info, messages, traj)

                # Extract parsed results
                parsed = openai_result["parsed_result"]
                logger.debug(
                    f"Successfully parsed - normal_text: '{parsed['normal_text']}', " f"calls: {parsed['calls']}"
                )

            except Exception as e:
                logger.warning(f"Exception in OpenAI adapter: {e}")
                logger.warning(f"rollout response: {response} can not be parsed into " f"tool calls {e}")
                res.status = Status.ABORTED
                return self._build_final_result(res, total_reward, info, messages, traj)

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response})

            # Execute action in environment
            agent_content, calls = parsed["normal_text"], parsed["calls"]
            logger.debug(f"Creating action from - content: '{agent_content}', " f"calls: {calls}")
            action = call_to_action_sglang(calls, agent_content)
            logger.debug(f"Created action: {action}")

            try:
                env_response = await self._execute_tool(env, action)
            except Exception as e:
                logger.warning("Environment step failed, this is usually related to " "the User simulation call.")
                logger.warning(f"Error: {e}")
                res.status = Status.ABORTED
                return self._build_final_result(res, total_reward, info, messages, traj)

            logger.debug(f"Environment response: reward={env_response.reward}, " f"done={env_response.done}")

            # Update message history based on action type
            if action.name != RESPOND_ACTION_NAME:
                messages.append(
                    {
                        "role": "tool",
                        "name": action.name,
                        "content": env_response.observation,
                    }
                )
            else:
                # Direct response from user
                messages.append({"role": "user", "content": env_response.observation})

            # Update token tracking: the observation and the next generation prompt
            generated_eos = generated[-1:] == [tokenizer.eos_token_id]
            traj.add_observation(self._observation_tokens(tokenizer, messages, generated_eos=generated_eos))

            # Update reward and info
            total_reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # Check if done
            if env_response.done:
                res.status = Status.COMPLETED
                break

        # Handle truncation
        if not env_response.done:
            res.status = Status.TRUNCATED

        return self._build_final_result(res, total_reward, info, messages, traj)

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        traj: _Trajectory,
    ) -> InteractionResult:
        """
        Build the final interaction result with all collected data.

        Args:
            res: InteractionResult instance to populate
            total_reward: Total reward accumulated during interaction
            info: Environment info dictionary
            messages: Complete conversation messages
            traj: Token-level record of the interaction

        Returns:
            Populated InteractionResult
        """
        res.reward = total_reward
        res.info = info
        res.messages = messages
        res.loss_mask = traj.loss_mask
        res.tokens = traj.tokens
        res.response = "".join([msg.get("content", "") for msg in messages if msg["role"] == "assistant"])
        res.response_length = len(traj.response_token_ids)
        res.rollout_log_probs = traj.rollout_log_probs
        res.rollout_topk_token_ids = traj.rollout_topk_token_ids
        res.rollout_topk_log_probs = traj.rollout_topk_log_probs
        res.weight_versions = traj.weight_versions

        logger.debug(
            f"_build_final_result: response_length={res.response_length}, "
            f"prompt_token_len={len(traj.prompt_token_ids)}, "
            f"response='{res.response[:100]}...'"
        )
        return res


class TrainableToolCallingAgent(ToolCallingAgent, TrainableAgentMixin):
    """
    A trainable version of ToolCallingAgent that uses sglang rollout for training.

    This agent combines the original ToolCallingAgent functionality with the
    TrainableAgentMixin to support async interaction with sglang servers for
    reinforcement learning training.
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        rollout_args: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        # Initialize the parent ToolCallingAgent
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
        )

        # Store rollout and sampling parameters as instance variables
        self.rollout_args = rollout_args or {
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 30000,
            "use_http2": False,
        }
        self.sampling_params = sampling_params or {
            "temperature": self.temperature,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
        }
        # Initialize OpenAI adapter
        self.openai_adapter = create_openai_adapter(tools_info=self.tools_info, parser_type=TOOL_CALL_PARSER)


def agent_factory(
    tools_info: list[dict[str, Any]],
    wiki,
    config: RunConfig,
    rollout_args: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> Agent:
    if config.agent_strategy == "tool-calling":
        return TrainableToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            rollout_args=rollout_args,
            sampling_params=sampling_params,
        )
    else:
        raise NotImplementedError(f"Unsupported agent strategy: {config.agent_strategy}")
