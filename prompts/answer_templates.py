"""Prompt snippets used by runtime inference."""

FINAL_ANSWER_SYNTHESIS_PROMPTS = {
    "parse_failure": (
        "Your previous response did not contain a complete <answer>...</answer> tag. "
        "Do not call any tools. Based only on the current conversation, image observations, "
        "and tool outputs, provide the best supported final answer now. Use exactly one "
        "complete <answer>...</answer> tag. If uncertain, still give the best supported answer."
    ),
    "tool_budget": (
        "The planned tool budget has been reached. Do not call any more tools. "
        "Consolidate the existing image observations and tool outputs, resolve any conflicts "
        "by preferring the most directly validated evidence, and provide the best supported "
        "final answer now. Use exactly one complete <answer>...</answer> tag."
    ),
    "max_turns": (
        "You have reached the maximum tool/reasoning turns. Do not call any tools. "
        "Based only on the current conversation, image observations, and tool outputs, "
        "provide the best supported final answer now. Use exactly one complete "
        "<answer>...</answer> tag. If uncertain, still give the best supported answer."
    ),
}


def final_answer_synthesis_prompt(reason: str) -> str:
    return FINAL_ANSWER_SYNTHESIS_PROMPTS.get(reason, FINAL_ANSWER_SYNTHESIS_PROMPTS["max_turns"])
