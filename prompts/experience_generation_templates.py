SINGLE_ROLLOUT_SUMMARY = """You are a world-class reasoning analysis expert. A multimodal agent system uses tool-based visual reasoning to solve the given problem. Please summarize the following trajectory step-by-step:

1. For each step: describe the tool used, the reasoning behind it, and the outcome.
2. Identify errors, detours, or backtracking; explain why they occurred and their impact.
3. Note any image preprocessing or intermediate visual checks and their contribution.
4. Preserve notable code snippets that produced correct results or caused errors.

<trajectory>
{trajectory}
</trajectory>

Prefer final solving/validation code over display-only crop or visualization probes. When code is reusable, keep the core 8-80 lines that parse the current problem, derive image-specific inputs, compute the answer, and validate it.

Provide a clear, structured summary."""


INTRA_SAMPLE_CRITIQUE = """Review these rollouts against the ground truth and extract reusable lessons.

Rules:
1. Output at most {max_ops} operations.
2. Use `kind: "success"` only for methods that led to a correct answer. If all rollouts failed, output only failure lessons.
3. Ground every lesson in the rollouts. Do not invent methods or recipes not evidenced in the trajectories.
4. Start each `content` with a trigger condition ("When...", "If...", "For...") so the agent knows when to apply it.
5. Keep `content` under 64 words.
6. For successful rollouts, extract the actual tool sequence, validation checks, reusable code pattern, and fallback that made the success reliable. The `content` must describe the concrete operation that worked; do not write generic lessons like "reuse successful code paths".
7. For failed rollouts, separate unrecoverable mistakes from recoverable partial competence. If a failed trajectory contains a locally useful crop, tool call, code fragment, measurement, or validation attempt, preserve that partial step in `tool_sequence`, explain the broken step in `anti_pattern`, and write a concrete `fallback_plan` for repairing it. Do not relabel the failure as success here.
8. Disallowed vague success wording: "similar visual reasoning problem", "verified tool-backed solution", "reuse successful/verified code path", or "tool-derived evidence" unless the same sentence names the concrete operation that was supported by the rollout.
9. Reusable code must not hardcode instance facts from one rollout: fixed layouts, coordinates, crop ratios, labels, candidate strings, recognized outputs, numeric results, or final answers. Derive these from the current image/problem text inside the code; omit `code_template` if the only working code is instance-specific.
10. If a method uses a crop or region of interest, include a validation check that the crop covers the full target scope named by the current question before narrowing further.

Fields:
- `option`: "add" or "modify"
- `kind`: "success" or "failure"
- `task_tags`: short lowercase task category tags
- `tool_tags`: tools involved (e.g., "code_interpreter", "web_search", "zoom")
- `content`: the reusable lesson (must start with a trigger condition)
- `tool_sequence`: ordered list of tool steps, each with `tool`, `purpose`, and optional `expected_output`
- `validation_checks`: list of checks the agent must perform before answering
- `fallback_plan`: list of fallback steps when the first tool result is empty, noisy, or inconsistent
- `code_template`: (optional) a reusable code snippet if the lesson involves a coding pattern
- For `code_template`, prefer 8-80 lines of minimal executable solving code that derives current inputs, computes an intermediate result, and validates the answer. Do not store display-only setup code.
- For structured-input tasks, code templates may include parsing and checking logic, but must not include rollout-specific labels, dimensions, candidate strings, or answer values.
- For region-focused tasks, code templates must derive the region from current image evidence or question text; do not store fixed proportional coordinates copied from a rollout.
- `anti_pattern`: (optional, for failures) the specific mistake to avoid
- `modified_from`: only for modify (the experience ID being modified)

Return brief analysis, then a JSON array:
```json
[
  {{
    "option": "add",
    "kind": "failure",
    "task_tags": ["tool_verified_reasoning"],
    "tool_tags": ["code_interpreter"],
    "content": "When a task depends on tool-verifiable intermediate evidence, derive the current inputs from the question, compute the evidence, and validate it before answering.",
    "tool_sequence": [
      {{"tool": "code_interpreter", "purpose": "derive task-specific inputs from the current question and image", "expected_output": "inspectable intermediate evidence tied to the current query"}},
      {{"tool": "code_interpreter", "purpose": "compute and validate the candidate answer from that evidence", "expected_output": "a result plus a check that can support or reject the answer"}}
    ],
    "validation_checks": [
      "Inspect the intermediate evidence before finalizing.",
      "Reject a result if the evidence does not cover the full target scope named by the question."
    ],
    "fallback_plan": [
      "If the first evidence extraction is noisy or incomplete, revise the derived inputs and rerun the check.",
      "If checks disagree, report only the answer supported by the clearest current evidence."
    ],
    "anti_pattern": "Answering directly without validating that the intermediate evidence matches the current target scope.",
    "code_template": ""
  }}
]
```
Quality over quantity.

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<experiences_used>
{experiences}
</experiences_used>

<ground_truth>
{groundtruth}
</ground_truth>
"""


SUPERVISED_CORRECTION_PROMPT = """You are converting failed training rollouts into one unverified recovery candidate.

The training answer is known, but the candidate must still be a reusable tool policy, not a memorized answer.
Bias toward proposing a candidate whenever the failed rollouts contain any local
tool evidence, crop, transform, measurement, visual probe, or near-miss reasoning
that can be repaired into a new executable check. Return `recoverable=false`
only for system/API failures, empty trajectories, or failures with no useful
local action to preserve.

Return JSON only. If the failures contain no recoverable local evidence or tool pattern, return:
```json
{{"recoverable": false, "reason": "brief reason"}}
```

If recoverable, return:
```json
{{
  "recoverable": true,
  "content": "When ..., repair the failed pattern by ...",
  "task_tags": ["short", "lowercase", "tags"],
  "tool_tags": ["code_interpreter"],
  "tool_sequence": [
    {{
      "tool": "code_interpreter",
      "purpose": "specific revised tool action using current image/problem inputs",
      "expected_output": "intermediate evidence that can support or reject the candidate answer"
    }}
  ],
  "validation_checks": [
    "must-pass check before final answer"
  ],
  "fallback_plan": [
    "what to revise and rerun if the repaired step is empty/noisy/inconsistent"
  ],
  "anti_pattern": "the failed step that should not be repeated",
  "code_template": ""
}}
```

Rules:
1. Use the ground truth only to diagnose what evidence the failed rollouts missed or misread.
2. Do not include the ground-truth answer, fixed coordinates, labels, candidate strings, numeric results, or instance-specific constants in `content` or `code_template`.
3. Preserve and repair locally useful failed steps into executable checks on new problems; do not reject a candidate merely because no rollout solved the sample.
4. Abstract single-sample nouns into reusable roles such as "target entity", "target region", "reference signal", or "candidate answer" unless the category itself is the reusable task family.
5. Do not make a candidate whose wording only applies to one object instance from the question, such as one named object, one scene, or one answer option.
6. Keep the policy compact: one candidate, at most three tool steps, at most six validation/fallback bullets.
7. If the repair can be implemented with reusable code, include a compact runnable `code_template` that starts from the current image/problem and performs the core current-input check. Leave it empty only when the useful code would require fixed coordinates, fixed labels, or other instance-specific constants.
8. It is allowed and encouraged to describe directional correction signals such as missing evidence, wrong scope, wrong unit, wrong orientation, or over-excluded uncertain candidates. Do not mention the exact ground-truth value.
9. When a failure involved rejecting or accepting candidates, the repaired policy should produce an evidence ledger with accepted, uncertain, and rejected items; uncertain visible candidates should not be silently discarded.

<question>
{question}
</question>

<failed_rollout_summaries>
{summaries}
</failed_rollout_summaries>

<ground_truth>
{groundtruth}
</ground_truth>

<experiences_used>
{experiences}
</experiences_used>
"""


MERGE_PROMPT = """Merge the following experiences into a single, comprehensive experience.

[Experiences to merge]:
{experiences_text}

Requirements:
1. Contain all important information from all experiences
2. Be clear, generalizable, and no more than 64 words
3. Avoid redundancy while preserving unique insights
4. Preserve the concrete task operation and tool evidence; do not replace it with generic wording like "reuse successful code paths".
5. If the experiences describe different task operations, keep only the shared lesson that is truly common.

Output ONLY the merged experience text."""


EXPERIENCE_REFINE_PROMPT = """You are an experience library curator. Perform a global refinement pass.

### Current Library Size: {exp_count} experiences
### Target: Reduce to 80-100 high-quality, diverse experiences

### Goals:
1. Merge Redundant: Combine experiences expressing the same core insight.
2. Generalize Over-Specific: Abstract scenario-specific experiences into general patterns.
3. Delete Low-Value: Remove experiences that are too obvious or too narrow.

### Operations:
- merge: Combine 2+ experiences (provide merged_from list)
- delete: Remove an experience (provide deleted_id)

### Quality Standard:
- Under 64 words, clear, actionable, generalizable
- Starts with trigger condition ("When...", "For...", "If...")
- No specific object names unless essential

### Important:
- Only act on clear redundancy or low quality. Preserve diversity.
- If the library is already well-curated, minimal changes are fine.

Output reasoning, then:
```json
[
  {{"option": "merge", "experience": "merged text", "merged_from": ["E12", "E23"]}},
  {{"option": "delete", "deleted_id": "E45"}}
]
```

<experiences>
{experiences}
</experiences>
"""


INJECTION_HEADER = """Here are practical tips for tool-based visual reasoning, gathered from similar problems:

{bullets}

Consider applying these when you encounter matching situations. Reference them by ID (e.g., [E12]) in your reasoning.

Your instruction is following:
"""
