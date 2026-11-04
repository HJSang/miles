# Tau bench 
This example shows miles training in an agentic multi-turn tool use environment. 


## Environment Setup 
Use the `zhuzilin/miles:latest` image and initialize the environment required for Search-R1:

```bash
cd /root/
git clone https://github.com/radixark/miles.git
cd miles
pip install -e . --no-deps
# for tau bench 
cd /root/
git clone https://github.com/JD-ETH/tau-bench.git
cd tau-bench
git checkout feature/litellm-retry
pip install -e . --no-deps 
```

Use the following script to generate mock data for miles training. 

```bash
cd /root/miles/examples/experimental/tau-bench
python tau1_mock.py --local_dir /root/tau-bench/
```

Initialize the Qwen2.5-3B-Instruct model needed for tool use:

```bash
# hf checkpoint
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# mcore checkpoint
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3-4B-Instruct-2507)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
    --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

## Running the Script

The user simulator runs through litellm, so any litellm-supported provider works.
Select the provider/model and supply the matching key via environment variables —
no need to edit `generate_with_tau.py`. The launch script forwards these to the
ray workers.

**DeepSeek:**

```bash
export TAU_USER_MODEL_PROVIDER=deepseek
export TAU_USER_MODEL=deepseek-chat
export DEEPSEEK_API_KEY=sk-...

cd /root/miles
bash examples/experimental/tau-bench/run_qwen3_4B.sh
```

**Gemini (default):**

```bash
export TAU_USER_MODEL_PROVIDER=gemini          # optional, this is the default
export TAU_USER_MODEL=gemini-2.5-flash-lite    # optional, this is the default
export GEMINI_API_KEY=...

cd /root/miles
bash examples/experimental/tau-bench/run_qwen3_4B.sh
```

If the matching `*_API_KEY` is not set, the run fails fast at startup with a clear error.
## Training on one domain, evaluating on another

`tau1_mock.py` records the tau-bench domain and split of every task in the row's
`metadata`, and `generate_with_tau.py` builds the environment from them. Point
`--prompt-data` at one domain and `--eval-prompt-data` at another, e.g. train on
retail/train and evaluate on airline/test (held out) plus retail/dev (in domain):

```bash
--prompt-data /root/tau-bench/retail_train_tasks.jsonl
--eval-prompt-data airline-test /root/tau-bench/airline_test_tasks.jsonl retail-dev /root/tau-bench/retail_dev_tasks.jsonl
```

## Qwen3.5 policies

Qwen3.5 chat templates emit tool calls in the XML `<function=...>` format and render
`<think>` blocks. Select the matching sglang parser, and keep thinking off (the default)
so turns stay short:

```bash
export TAU_TOOL_CALL_PARSER=qwen3_coder   # default qwen25: JSON <tool_call> (Qwen2.5 / Qwen3)
export TAU_ENABLE_THINKING=0              # 1 keeps the reasoning of every turn in context
```

Both are read inside the rollout actors; the launch script forwards them to ray.

## A local user simulator

Any OpenAI-compatible server works as the user simulator through litellm's `openai`
provider, e.g. an sglang server on spare GPUs:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --served-model-name usersim --port 30100
export TAU_USER_MODEL_PROVIDER=openai TAU_USER_MODEL=usersim
export OPENAI_API_KEY=local OPENAI_API_BASE=http://127.0.0.1:30100/v1
```
