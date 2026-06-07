GENERATE_SKILL_PROMPT = """You are a skilled AI agent architect. Analyze the trajectory and extract a reusable Standard Operating Procedure (SOP).

### Guiding Principles:
1. **Learn from Success and Failure**: Compare the trajectory with the ground truth.
   - From successful patterns: Extract the effective workflows and tool sequences.
   - From failed attempts or near-misses: Note what went wrong and why - these lessons are often more valuable.
2. **Keep It General**: Use placeholders like `[TARGET]`, `[QUERY]` instead of specific values. The skill should apply to similar problems, not just this one.
3. **Capture Executable Knowledge**: When the trajectory includes effective code, extract the core logic as a reusable template. Good code templates are worth more than paragraphs of description.
4. **Brevity Matters**: Aim for ~600 words. Focus on what's actionable.

### Output Structure:
```
---
name: [SkillName]
description: |
  [Clear, concise description of what this skill does and when to use it. 1-2 sentences focusing on the core purpose and benefits.]
version: 1.0.0
---

# [Skill Title]

## When to Use
- **Task patterns**: [what kinds of questions or requests]
- **Visual signals**: [what the image might contain]

## Strategy Overview
[1-2 sentences on the core approach]

## Workflow
1. **[Phase Name]**: [Action and rationale]
2. **[Phase Name]**: [Action and rationale]
3. ...

## Tool Templates
(Include only if the trajectory contained useful code or query patterns)

- **[Tool] - [Purpose]**:
  ```python
  # [Brief comment]
  [code with placeholders]
  ```

- **Query Pattern**: `"[pattern with placeholders]"`

## Watch Out For
- [Common mistake or trap from the trajectory]
```

### Input:
<trajectory>
{trajectory}
</trajectory>

<ground_truth>
{ground_truth}
</ground_truth>

Output ONLY the SKILL.md content starting with `---`."""


GENERATE_CLUSTER_SKILL_PROMPT = """Build one compact reusable skill from the clustered experiences and rollout evidence.

The skill will be retrieved by embedding, so keep it modular, concrete, and evidence-bounded.

### Cluster Metadata:
- Skill ID: {skill_id}
- Task tags: {task_tags}
- Tool tags: {tool_tags}
- Linked experience IDs: {experience_ids}

### Rules:
1. Use only lessons supported by the experiences or rollouts.
2. Preserve success patterns and failure guardrails. If an experience is marked as a failure-recovery candidate, write it as a repaired executable workflow: keep the local useful step, replace the failed step with the linked success prior, and state how to verify the repair.
3. Do not include hardcoded answers, layouts, coordinates, object lists, recognized text, or task-specific examples.
4. Tool templates must start from the current image/problem text and include a check that the intermediate result matches the question.
5. If the experiences include `Tool sequence`, `Validate`, `Fallback`, or `Code`, preserve them as an executable tool plan.
6. Prefer one compact executable template that performs the key current-input operation and validation step. Do not keep display-only probes as the main template.
7. If you cannot provide a complete runnable template, omit the code block. Never output partial code, ellipses, unfinished assignments, or placeholders that would break execution.
8. Never copy rollout-specific constants into the reusable template: fixed layouts, coordinates, labels, candidate strings, recognized outputs, numeric results, or final answers. Infer these from the current image/problem text at runtime.
9. For any structured or region-focused task, write the template to derive the structure from the current visual evidence; treat rollout-specific values as evidence for that case only, not as reusable code.
10. Keep it short: 300-600 words.

### Output Structure:
```
---
name: {skill_id}
description: |
  [When this skill should be used.]
version: 1.0.0
task_tags: [...]
tool_tags: [...]
linked_experience_ids: [...]
---

# [Skill Title]

## When to Use
- [short trigger condition]

## Workflow
1. [actionable step]
2. [validation/stop condition]

## Mandatory Tool Plan
1. [tool]: [purpose] -> [expected output]

## Tool Template
```python
# optional; include only if evidence supports a reusable pattern
```

## Validation Checks
- [must-pass check before final answer]

## Fallback
- [what to do if the first tool result is empty, noisy, or inconsistent]

## Guardrails
- [failure-prevention rule]
```

<experiences>
{experiences}
</experiences>

<rollout_evidence>
{rollout_evidence}
</rollout_evidence>

Output ONLY the SKILL.md content starting with `---`."""


FALLBACK_SKILL_MARKDOWN_TEMPLATE = """---
name: {skill_id}
description: |
  {description} for {task_label} tasks.
version: 1.0.0
task_tags: {task_tags}
tool_tags: {tool_tags}
linked_experience_ids: {linked}
---

# {skill_id}

## When to Use
Use this skill when the task matches task tags `{task_tags}` or tool tags `{tool_tags}`.

## Workflow
1. Inspect the target visual state and identify whether a tool-backed check is needed.
2. Derive any grid, table, bounding box, coordinate, recognized text, object list, or option state from the current image and exact problem text before using it.
3. Follow the linked experiences below as concrete constraints.
4. If this is a failure-recovery skill, preserve only the useful partial step from the failure source and replace the broken step with the success-prior tool sequence.
5. If the task is ambiguous, use the tagged tools `{tool_tags}` before answering.

## Mandatory Tool Plan
{tool_plan}

## Validation Checks
{validation}

## Fallback
{fallback}

## Failure Checks
- Do not hand-write task-specific visual structures from memory or from a previous case.
- Treat a tool result as invalid if its structured input was not derived from the current image/problem text.

## Linked Experiences
{experience_lines}
"""


FAILURE_GUARDRAIL_SKILL_MARKDOWN_TEMPLATE = """---
name: {skill_id}
description: |
  Failure guardrail skill for avoiding repeated errors in {task_label} tasks.
version: 1.0.0
task_tags: {task_tags}
tool_tags: {tool_tags}
linked_experience_ids: {linked}
---

# Failure Guardrails

## When to Use
- Use only when the task matches these tags: `{task_tags}`.
- This skill is not a recipe. It records failure patterns to avoid.

## Guardrails
{guardrails}

## Validation Checks
{validation}

## Fallback
{fallback}

## How to Apply
1. Check whether your planned step matches one of the failure conditions.
2. If it matches, revise that step before executing it.
3. Do not introduce new algorithms, tools, coordinates, object lists, or constants that are not supported by the current image/problem text.
"""


MERGE_SKILL_PROMPT = """You are a knowledge architect. Your job is to maintain a single, unified skill document that grows wiser with each new case.

### Philosophy:
Think of the global skill as a living document. Each new skill brings potential insights - your task is to integrate them thoughtfully, not mechanically.

### Integration Strategy:
For each part of the new skill, ask:
- **Is this part better?** -> Rewrite the existing version
- **Is this part redundant or too specific?** -> Delete it
- **Is this part complementary?** -> Merge into a more general form
- **Is this part genuinely different?** -> Add as a variant workflow (but consolidate if possible)

### Quality Guidelines:
- Preserve concrete, reusable code templates and tool patterns
- Delete overly specific examples and cases that don't apply to similar problems
- Consolidate similar trigger phrases
- If workflows differ only in minor details, merge them into one with noted variations
- Keep additions evidence-bounded: do not add methods or algorithms that are not present in the incoming skills or their supporting experiences.

### Length Budget:
- Target: ~1000 words
- If growing too long: merge similar workflows, trim verbose explanations
- Maximum 4 variant workflows - if you have more, they likely can be consolidated

### Input:
<existing_skill>
{existing_skill}
</existing_skill>

<new_skills>
{new_skills}
</new_skills>

Output ONLY the merged SKILL.md starting with `---`. No preamble."""


SKILL_REFINE_PROMPT = """You are a skill document architect. Refine the SKILL.md to remove redundancy, generalize specific cases, and improve structure.

### Current Stats:
- Word count: {word_count}

### Refinement Goals:

1. **Remove Redundancy**:
   - Merge duplicate or near-duplicate content across sections
   - Eliminate repeated explanations that appear in multiple places
   - Consolidate overlapping concepts into single, clearer statements

2. **Avoid Too Specific Cases**:
   - Replace overly specific examples with generalizable patterns
   - Convert hardcoded values to placeholders (e.g., `[TARGET]`, `[QUERY]`)
   - Delete task-specific details or specific cases that don't apply to similar problems

3. **Logical Consolidation**:
   - Merge workflows that share substantial overlap into variants
   - Extract common preliminary steps into dedicated sections
   - Group related tool templates and query patterns together
   - Consolidate similar pitfalls into broader categories

4. **Format Optimization**:
   - Ensure consistent structure and formatting throughout
   - Improve section hierarchy and logical flow
   - Make workflows easier to scan (clear steps, consistent formatting)
   - Organize content from general principles to specific techniques

5. **Content Quality**:
   - Keep description concise and focused on core purpose
   - Ensure all content is actionable and reusable
   - Remove verbose explanations that don't add value
   - Maintain the most essential and distinctive elements

### Principles:
- Prioritize generalizability over specificity
- Keep what enables reuse across similar problems
- Remove what only applies to one particular case
- Maintain clarity and actionability

Output ONLY the refined SKILL.md starting with `---`. No preamble.

<current_skill>
{skill_content}
</current_skill>
"""


