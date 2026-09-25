from pathlib import Path

import pytest

from miles.utils.external_utils.model_args_utils import import_module_from_path

_TOOL = Path(__file__).resolve().parents[3] / "scripts" / "tools" / "build_boxed_math_eval.py"


@pytest.fixture(scope="module")
def tool():
    return import_module_from_path(_TOOL, "build_boxed_math_eval_under_test")


def test_aime_rows_are_wrapped_in_the_dapo_instruction(tool):
    row = {"prompt": [{"role": "user", "content": "Find the number of residents."}], "label": "73"}

    out = tool.build_row(row)

    assert out["label"] == "73"
    assert out["prompt"] == [
        {
            "role": "user",
            "content": tool.DAPO_INSTRUCTION_PREFIX + "Find the number of residents." + tool.DAPO_INSTRUCTION_SUFFIX,
        }
    ]


def test_math500_rows_use_problem_and_answer(tool):
    row = {"problem": "Convert $(0,3)$ to polar.", "answer": "\\left( 3, \\frac{\\pi}{2} \\right)", "level": 2}

    out = tool.build_row(row)

    assert out["label"] == "\\left( 3, \\frac{\\pi}{2} \\right)"
    assert out["prompt"][0]["content"].startswith(tool.DAPO_INSTRUCTION_PREFIX)
    assert "Convert $(0,3)$ to polar." in out["prompt"][0]["content"]


def test_dapo_style_prompts_are_not_wrapped_twice(tool):
    content = tool.DAPO_INSTRUCTION_PREFIX + "What is 1+1?" + tool.DAPO_INSTRUCTION_SUFFIX
    row = {"prompt": [{"role": "user", "content": content}], "label": 2}

    assert tool.build_row(row)["prompt"][0]["content"] == content
    assert tool.build_row(row)["label"] == "2"


def test_multi_turn_prompts_are_rejected(tool):
    row = {"prompt": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], "label": "1"}

    with pytest.raises(ValueError, match="exactly one user turn"):
        tool.build_row(row)
