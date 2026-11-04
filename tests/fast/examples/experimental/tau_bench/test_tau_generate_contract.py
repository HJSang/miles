import importlib.util
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tests.fast.import_isolation_utils import REPO_ROOT

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample


async def test_tau_agent_receives_caller_state_and_returns_converted_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tau wrapper passes runtime state into asolve and wraps its actual converted result."""
    status = Enum("Status", ["COMPLETED", "TRUNCATED", "ABORTED"])
    result = SimpleNamespace(
        status=status.COMPLETED,
        prompt="question",
        tokens=[10, 20],
        response="answer",
        reward=0.75,
        loss_mask=[1],
        info={"task": 7},
        response_length=1,
        rollout_log_probs=[-0.5],
        rollout_topk_token_ids=None,
        rollout_topk_log_probs=None,
        weight_versions=[],
    )
    state = SimpleNamespace(args=SimpleNamespace(partial_rollout=False))
    sampling = {"temperature": 0.5}
    agent = SimpleNamespace(rollout_args=state.args, sampling_params=sampling, asolve=AsyncMock(return_value=result))
    env = SimpleNamespace(tools_info=["tool"], wiki="wiki")
    get_env = Mock(return_value=env)
    factory = Mock(return_value=agent)
    for name, attributes in {
        "tau_bench": {},
        "tau_bench.envs": {"get_env": get_env},
        "tau_bench.types": {"RunConfig": SimpleNamespace},
        "trainable_agents": {"InteractionResult": SimpleNamespace, "Status": status, "agent_factory": factory},
    }.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("TAU_USER_MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder")
    spec = importlib.util.spec_from_file_location(
        "tau_contract", REPO_ROOT / "examples/experimental/tau-bench/generate_with_tau.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The dataset row names the domain and split; the sample keeps the identity miles gave it.
    prompt_sample = Sample(group_index=2, index=11, prompt="7", metadata={"env": "airline", "task_split": "test"})
    output = await module.generate(
        GenerateFnInput(state=state, sample=prompt_sample, sampling_params=sampling, evaluation=False)
    )

    assert get_env.call_args.kwargs["env_name"] == "airline"
    assert get_env.call_args.kwargs["task_split"] == "test"
    assert get_env.call_args.kwargs["task_index"] == 7
    agent.asolve.assert_awaited_once_with(env, state.args, sampling, 7, state=state, evaluation=False)
    assert factory.call_args.kwargs["rollout_args"] is state.args
    assert factory.call_args.kwargs["sampling_params"] is sampling
    assert isinstance(output, GenerateFnOutput)
    assert output.samples is not prompt_sample and prompt_sample.tokens == []
    assert (output.samples.group_index, output.samples.index) == (2, 11)
    assert output.samples.tokens == [10, 20]
    assert output.samples.response == "answer"
    assert output.samples.response_length == 1
    assert output.samples.reward == 0.75
    assert output.samples.loss_mask == [1]
    assert output.samples.status is Sample.Status.COMPLETED
    assert output.samples.rollout_log_probs == [-0.5]
    assert output.samples.metadata == {"env": "airline", "task_split": "test", "task": 7}
