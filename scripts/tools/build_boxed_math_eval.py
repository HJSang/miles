"""Build boxed-answer math eval sets in the DAPO-17K prompt format.

Wraps every problem in the instruction that ``zhuzilin/dapo-math-17k`` uses, so the
``math`` (boxed) grader sees the answer format the student was trained on. Accepts
the ``zhuzilin/aime-2024`` and ``zhuzilin/aime-2025`` jsonl rows (``prompt`` as a
message list, ``label``) and ``HuggingFaceH4/MATH-500`` ``test.jsonl`` rows
(``problem``, ``answer``), and writes ``{"prompt": [...], "label": "..."}`` rows.

Usage:
    python scripts/tools/build_boxed_math_eval.py \
        --src /root/datasets/MATH-500/test.jsonl \
        --dst /root/datasets/sc_opd_eval/math500.jsonl
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

DAPO_INSTRUCTION_PREFIX = "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
DAPO_INSTRUCTION_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def _problem_text(row: dict) -> str:
    if "problem" in row:
        return str(row["problem"]).strip()
    prompt = row["prompt"]
    if isinstance(prompt, str):
        return prompt.strip()
    user_turns = [m["content"] for m in prompt if m.get("role") == "user"]
    if len(user_turns) != 1:
        raise ValueError(f"Expected exactly one user turn in the prompt, got {len(user_turns)}: {prompt!r}")
    return str(user_turns[0]).strip()


def _label_text(row: dict) -> str:
    label = row["answer"] if "answer" in row else row["label"]
    return str(label).strip()


def build_row(row: dict) -> dict:
    problem = _problem_text(row)
    if problem.startswith(DAPO_INSTRUCTION_PREFIX):
        content = problem
    else:
        content = f"{DAPO_INSTRUCTION_PREFIX}{problem}{DAPO_INSTRUCTION_SUFFIX}"
    return {"prompt": [{"role": "user", "content": content}], "label": _label_text(row)}


def convert_rows(rows: Iterable[dict]) -> list[dict]:
    return [build_row(row) for row in rows]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source jsonl (zhuzilin AIME format or MATH-500 test.jsonl).")
    parser.add_argument("--dst", required=True, help="Destination jsonl with DAPO-style boxed prompts.")
    args = parser.parse_args()

    rows = convert_rows(_read_jsonl(Path(args.src)))
    _write_jsonl(Path(args.dst), rows)
    print(f"wrote {len(rows)} rows to {args.dst}")


if __name__ == "__main__":
    main()
