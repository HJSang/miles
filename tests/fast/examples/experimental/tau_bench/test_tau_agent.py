"""Wiring of the tau-bench agent function, with tau-bench and the OpenAI client replaced by fakes."""

import asyncio
import importlib.util
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from tests.fast.import_isolation_utils import REPO_ROOT

from miles.utils.types import Sample

SESSION_URL = "http://router:30000/sessions/abc"
ROUTER_URL = "http://engines:30200"
SIMULATOR_URL = f"{ROUTER_URL}/v1"


class _FakeMessage:
    def __init__(self, **fields: Any):
        self._fields = fields

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in self._fields.items() if not (exclude_none and v is None)}


@dataclass
class _FakeChatClient:
    """An OpenAI client that answers every request with one scripted message and records the request."""

    base_url: str
    replies: list[_FakeMessage]  # in order; the last one repeats
    headers: dict[str, str] | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, tools=None, max_tokens=None, temperature=None, top_p=None, extra_body=None):
        """The OpenAI SDK's explicit signature: an SGLang field such as top_k must arrive in extra_body."""
        request = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        self.requests.append(
            {**{k: v for k, v in request.items() if v is not None}, "tools": tools, "extra_body": extra_body}
        )
        reply = self.replies[min(len(self.requests), len(self.replies)) - 1]
        return SimpleNamespace(choices=[SimpleNamespace(message=reply, finish_reason="stop")])


@dataclass
class _ChatResult:
    message: dict[str, Any]
    finish_reason: str | None = None
    cost: float = 0.0


class _FakeSimulator:
    """tau-bench's LLMUserSimulationEnv surface: keeps the injected chat."""

    def __init__(self, chat=None, model=None, provider=None):
        self.chat = chat


class _FakeAgent:
    """tau-bench's ToolCallingAgent surface: calls the policy chat once and relays the simulator's reply."""

    built: list["_FakeAgent"] = []

    def __init__(self, tools_info, wiki, chat=None, **kwargs):
        self.tools_info, self.wiki, self.chat = tools_info, wiki, chat
        _FakeAgent.built.append(self)

    def solve(self, env, task_index=None, max_num_steps=30):
        self.solve_args = (env, task_index, max_num_steps)
        policy_turn = self.chat([{"role": "user", "content": "hi"}], self.tools_info)
        simulator_turn = env.user.chat([{"role": "user", "content": policy_turn.message["content"]}], None)
        return SimpleNamespace(
            reward=1.0,
            messages=[
                {"role": "user", "content": "hi"},
                policy_turn.message,
                {"role": "user", "content": simulator_turn.message["content"]},
            ],
        )


@pytest.fixture
def tau_agent(monkeypatch: pytest.MonkeyPatch):
    envs: list[SimpleNamespace] = []

    def get_env(**kwargs):
        env = SimpleNamespace(
            tools_info=[{"type": "function"}], wiki="Be helpful.", user=kwargs["user"], kwargs=kwargs
        )
        envs.append(env)
        return env

    fake_modules = {
        "tau_bench": {},
        "tau_bench.agents": {},
        "tau_bench.agents.tool_calling_agent": {"ToolCallingAgent": _FakeAgent},
        "tau_bench.chat": {"ChatFn": object, "ChatResult": _ChatResult},
        "tau_bench.envs": {"get_env": get_env},
        "tau_bench.envs.user": {"LLMUserSimulationEnv": _FakeSimulator},
        "tau_bench.types": {"SolveResult": SimpleNamespace},
    }
    for name, attributes in fake_modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    clients: dict[str, _FakeChatClient] = {}
    replies = {
        f"{SESSION_URL}/v1": [
            _FakeMessage(role="assistant", content="Which order?", reasoning_content="Ask first.", tool_calls=None)
        ],
        # The simulator's first attempt is a thinking-only reply, which the agent retries.
        SIMULATOR_URL: [
            _FakeMessage(role="assistant", content=None, reasoning_content="hmm"),
            _FakeMessage(role="assistant", content="Order 7.", reasoning_content="They asked."),
        ],
    }

    def fake_openai(base_url: str, api_key: str, default_headers: dict[str, str] | None = None):
        clients[base_url] = _FakeChatClient(base_url=base_url, replies=replies[base_url], headers=default_headers)
        return clients[base_url]

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    monkeypatch.setenv("TAU_MAX_TURNS", "5")
    monkeypatch.delenv("TAU_USER_SIMULATOR_MAX_TOKENS", raising=False)
    _FakeAgent.built.clear()

    spec = importlib.util.spec_from_file_location(
        "tau_agent", REPO_ROOT / "examples/experimental/tau-bench/tau_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, envs=envs, clients=clients)


def test_episode_wires_policy_to_the_session_and_simulator_to_the_router(tau_agent):
    metadata = {
        "env": "retail",
        "task_split": "train",
        "task_index": 7,
        "session_id": "abc",
        "inference_router_url": ROUTER_URL,
        "inference_router_headers": {"X-SMG-Routing-Key": "abc"},
    }
    result = asyncio.run(
        tau_agent.module.run(
            base_url=SESSION_URL,
            prompt=[{"role": "user", "content": "placeholder"}],
            request_kwargs={"max_tokens": 1024, "temperature": 1.0, "top_k": -1, "top_p": 1.0},
            metadata=metadata,
        )
    )
    assert result == {"reward": 1.0, "tau_turns": 1}

    [env] = tau_agent.envs
    assert {k: env.kwargs[k] for k in ("env_name", "task_split", "task_index")} == {
        "env_name": "retail",
        "task_split": "train",
        "task_index": 7,
    }
    assert isinstance(env.user, _FakeSimulator)
    [agent] = _FakeAgent.built
    assert agent.solve_args == (env, 7, 5)
    assert agent.tools_info is env.tools_info and agent.wiki == env.wiki

    [policy_request] = tau_agent.clients[f"{SESSION_URL}/v1"].requests
    assert policy_request["model"] == "default" and policy_request["tools"] is env.tools_info
    assert (policy_request["max_tokens"], policy_request["temperature"]) == (1024, 1.0)
    assert policy_request["top_p"] == 1.0 and policy_request["extra_body"] == {"top_k": -1}

    simulator = tau_agent.clients[SIMULATOR_URL]
    assert simulator.headers == {"X-SMG-Routing-Key": "abc"}
    assert tau_agent.clients[f"{SESSION_URL}/v1"].headers is None
    retried, simulator_request = simulator.requests
    assert retried["messages"] == simulator_request["messages"]
    assert simulator_request["tools"] is None
    assert simulator_request["max_tokens"] == 32768
    assert simulator_request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}
    }
    assert simulator_request["messages"] == [{"role": "user", "content": "Which order?"}]


def test_openai_chat_returns_the_turn_as_generated(tau_agent):
    reply = _FakeMessage(role="assistant", content=None, tool_calls=[{"id": "c1"}], reasoning_content="r")
    client = _FakeChatClient(base_url="x", replies=[reply])
    result = tau_agent.module.openai_chat(client, temperature=0.5)([{"role": "user", "content": "hi"}], None)
    assert result.message == {"role": "assistant", "tool_calls": [{"id": "c1"}], "reasoning_content": "r"}
    assert result.finish_reason == "stop"
    assert client.requests[0]["temperature"] == 0.5 and client.requests[0]["tools"] is None


def test_reward_func_reads_the_episode_reward(tau_agent):
    samples = [
        Sample(metadata={"reward": 1.0}),
        Sample(metadata={"reward": 0.0}),
        Sample(metadata={"session_id": "s"}),
    ]
    # The last sample's episode never finished: it scores 0 instead of failing the run.
    assert asyncio.run(tau_agent.module.reward_func(None, samples)) == [1.0, 0.0, 0.0]
    assert asyncio.run(tau_agent.module.reward_func(None, samples[0])) == 1.0


def test_require_reply_gives_up_after_the_last_attempt(tau_agent):
    client = _FakeChatClient(base_url="x", replies=[_FakeMessage(role="assistant", content="")])
    chat = tau_agent.module.require_reply(tau_agent.module.openai_chat(client), attempts=2)
    with pytest.raises(RuntimeError, match="no reply content after 2 attempts"):
        chat([{"role": "user", "content": "hi"}], None)
    assert len(client.requests) == 2
