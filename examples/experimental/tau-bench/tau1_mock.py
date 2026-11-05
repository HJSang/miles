"""Write one prompt row per tau-bench task.

Each row names the task by domain, split and index; the agent builds the real conversation from
the environment. The prompt is a placeholder message list, which the TITO session path requires.
"""

import argparse
import json
import os

from tau_bench.envs import get_env

ALL_DATA_MAPPINGS = {"retail": ["train", "test", "dev"], "airline": ["test"]}


def main():
    parser = argparse.ArgumentParser(description="Tau1 Mock Script")
    parser.add_argument("--local_dir", required=True, help="Path to the local directory")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    for env_name, splits in ALL_DATA_MAPPINGS.items():
        for split in splits:
            env = get_env(env_name=env_name, user_strategy="human", user_model=None, task_split=split)
            output_path = os.path.join(args.local_dir, f"{env_name}_{split}_tasks.jsonl")
            with open(output_path, "w") as f:
                for i in range(len(env.tasks)):
                    row = {
                        "prompt": [{"role": "user", "content": f"tau-bench {env_name}/{split} task {i}"}],
                        "metadata": {"env": env_name, "task_split": split, "task_index": i},
                    }
                    f.write(json.dumps(row) + "\n")
            print(f"Saved {len(env.tasks)} {env_name} ({split}) tasks to {output_path}")


if __name__ == "__main__":
    main()
