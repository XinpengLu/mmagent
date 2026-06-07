SKILL_INJECTION_HEADER = """Here are practical experiences and skills for tool-based visual reasoning:

<skill>
{skill_content}
</skill>

You can use it as reference if it is relevant to help you solve the problem. You can also have your own ideas or other approaches.
When the skill or linked experiences specify an action policy for the current task, treat it as the default execution plan unless the input clearly makes it unnecessary.
When linked failure experiences warn about a planned step, treat them as constraints on execution rather than optional background notes.
When a skill says to validate a visual intermediate result, generate a visible tool image and inspect it before finalizing; a printed object handle is not enough.
If a <mandatory_tool_plan> block is provided above, complete all applicable listed tool steps before the final answer; one combined tool call is acceptable only when it produces the expected intermediate evidence for each step.
When the plan asks for an evidence ledger, the final answer must follow that ledger rather than an earlier unverified estimate.

Your instruction is following:
"""


MANDATORY_TOOL_PLAN_TEMPLATE = """MANDATORY TOOL PLAN:
Allowed/expected tools: {tool_text}

Tool steps:
{steps}

Rules:
- Complete the listed tool steps in order when applicable; one tool call may combine multiple steps if it produces each expected intermediate result.
- Do not stop after the first broad view when later listed steps are needed to localize, verify, compare, or reconcile the answer.
- Derive all crops, coordinates, thresholds, labels, candidates, and constants from the current image/problem.
- Do not treat a manually listed candidate set, object count, measured value, or selected option as verified until the tool output includes visible or tabular evidence for omissions and conflicts.
- For candidate-based tasks, keep accepted, uncertain, and rejected items separate; do not silently drop uncertain visible evidence unless the tool output clearly rules it out.
- For failure guardrails, use the tool step to check and correct the planned answer; do not blindly repeat a failed method.
- When candidate answers are available, compare the tool-backed result against every candidate.

Validation checks:
{validations}

Fallback:
{fallbacks}{code_templates}
"""


DEFAULT_TOOL_PLAN_TEMPLATE = """MANDATORY TOOL PLAN:
Allowed/expected tools: {tool_text}

Tool steps:
{steps}

Rules:
- Do not provide the final answer until at least one enabled tool call has been executed.
- Choose the tool by evidence need, not by benchmark name.
- Derive tool inputs from the current image, problem text, or verified search result.
- Keep tool use compact; one well-targeted call is enough when it produces decisive evidence.

Validation checks:
- Base the final answer on the tool output and reconcile it with the visible image/problem.

Fallback:
- If the first tool fails or returns empty evidence, run one simpler check with an available tool.
"""
