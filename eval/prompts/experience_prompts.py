SINGLE_ROLLOUT_SUMMARY = """You are a World-Class reasoning analysis expert. A multimodal agent system uses tool-based visual reasoning to solve the given problem. Please summarize the following trajectory step-by-step:

1. For each step: describe the tool used, the reasoning behind it, and the outcome.
2. Identify errors, detours, or backtracking — explain why they occurred and their impact.
3. Note any image preprocessing or intermediate visual checks and their contribution.
4. Preserve notable code snippets (≤5 lines) that produced correct results or caused errors.

<trajectory>
{trajectory}
</trajectory>

Provide a clear, structured summary."""


INTRA_SAMPLE_CRITIQUE = """Review these rollouts against the ground truth and extract reusable lessons.

Rules:
1. Output at most {max_ops} operations.
2. Use `kind: "success"` only for methods that led to a correct answer. If all rollouts failed, output only failure lessons.
3. Ground every lesson in the rollouts. Do not invent methods or recipes not evidenced in the trajectories.
4. Start each `content` with a trigger condition ("When…", "If…", "For…") so the agent knows when to apply it.
5. Keep `content` under 64 words.

Fields:
- `option`: "add" or "modify"
- `kind`: "success" or "failure"
- `task_tags`: short lowercase task category tags
- `tool_tags`: tools involved (e.g., "code_interpreter", "web_search", "zoom")
- `content`: the reusable lesson (must start with a trigger condition)
- `code_template`: (optional) a reusable code snippet (≤5 lines) if the lesson involves a coding pattern
- `anti_pattern`: (optional, for failures) the specific mistake to avoid
- `modified_from`: only for modify (the experience ID being modified)

Return brief analysis, then a JSON array:
```json
[
  {{
    "option": "add",
    "kind": "failure",
    "task_tags": ["counting"],
    "tool_tags": ["code_interpreter"],
    "content": "When counting objects in a cluttered scene, use code_interpreter to segment and count via connected-component labeling rather than visual estimation.",
    "anti_pattern": "Directly estimating count from the raw image without segmentation",
    "code_template": "labels = measure.label(binary_mask)\\ncount = labels.max()"
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


MERGE_PROMPT = """Merge the following experiences into a single, comprehensive experience.

[Experiences to merge]:
{experiences_text}

Requirements:
1. Contain all important information from all experiences
2. Be clear, generalizable, and no more than 64 words
3. Avoid redundancy while preserving unique insights

Output ONLY the merged experience text."""


EXPERIENCE_REFINE_PROMPT = """You are an experience library curator. Perform a global refinement pass.

### Current Library Size: {exp_count} experiences
### Target: Reduce to 80-100 high-quality, diverse experiences

### Goals:
1. **Merge Redundant**: Combine experiences expressing the same core insight.
2. **Generalize Over-Specific**: Abstract scenario-specific experiences into general patterns.
3. **Delete Low-Value**: Remove experiences that are too obvious or too narrow.

### Operations:
- **merge**: Combine 2+ experiences (provide merged_from list)
- **delete**: Remove an experience (provide deleted_id)

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
