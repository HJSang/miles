# Tau bench

miles training in an agentic multi-turn tool-use environment, on the
[TITO session path](../../../docs/user-guide/agentic-rollout.md): tau-bench's own tool-calling
agent talks OpenAI chat to the session server, which owns every policy turn's token ids,
log-probs and adapter selection. The recipe trains Qwen3.8-27B with native LoRA on Megatron,
fully async, with score centering.

- The policy trains on **retail/train** (500 tasks) and is evaluated on **airline/test** (50
  tasks, a domain it never trains on) and **retail/dev** (20 tasks).
- The user simulator is the **frozen base model** on the same rollout engines, thinking at low
  effort: policy turns
  carry the live LoRA adapter through the session, simulator turns go to the engines' router (miles hands its address to the agent) and get
  no adapter. No external API is needed.
- Policy turns think (`reasoning_effort` is a launcher flag) under a 1024-token budget per
  turn, so the reward favors short reasoning.

## Environment setup

Use the `radixark/miles:dev` image and install tau-bench with its injectable model call (the
`tito` branch, [JD-ETH/tau-bench#1](https://github.com/JD-ETH/tau-bench/pull/1): `ToolCallingAgent(chat=...)`,
`get_env(user=...)`, every tool call of a turn executed, episodes ending on
`finish_reason == "length"`):

```bash
cd /root/
git clone -b tito https://github.com/HJSang/tau-bench.git
cd tau-bench
pip install -e . --no-deps
```

## Running

`prepare` downloads `Qwen/Qwen3.8-27B` to `--model-dir`, converts it to torch_dist next to it,
and writes one prompt row per task under `--data-dir/tau-bench` (`tau1_mock.py`). The job
then starts; W&B logging engages when `WANDB_API_KEY` is set.

```bash
cd /root/miles
python examples/experimental/tau-bench/run_qwen3_8_27b_lora.py
```

Every field of `ScriptArgs` is a flag, for example:

```bash
python examples/experimental/tau-bench/run_qwen3_8_27b_lora.py --reasoning-effort medium --max-turns 20
```

## Files

| File | Role |
|---|---|
| `tau_agent.py` | `run`: one tau-bench episode per sample with tau-bench's own agent and simulator, the policy chat bound to the session, the simulator chat to the router. `reward_func`: the episode reward. |
| `tau1_mock.py` | Writes the task rows: a placeholder prompt plus `env`, `task_split` and `task_index` metadata, which `run` turns into the environment. |
| `run_qwen3_8_27b_lora.py` | The launcher. |

To train on another domain or split, point `--prompt-data` and `--eval-prompt-data` at the
corresponding `tau1_mock.py` output; the rows carry the domain, so nothing else changes.
