# Prompt 1: Summarizes a single trajectory, focusing on tool usage and reasoning.
SINGLE_ROLLOUT_SUMMARY = """You are a World-Class reasoning analysis expert. A multimodal agent system uses tool-based visual reasoning to solve the given problem. The agent may have been provided with some experiences and skill guidance. Please summarize the following trajectory (also called rollout) step-by-step:

1. For each turn or step:
   1.1 Describe which tool was used and with what parameters, explain the reasoning for this specific action, and note which experience (if any) and which skill guidance (if any) were applied and how they influenced the action.
   1.2 If this turn was part of meta-reasoning skills: identify the meta-reasoning type (e.g., question decomposition, sequential reflection, self-correction, self-verification, etc.) and explain how its outcome influenced subsequent steps or the final result.

2. Given the trajectory and the correct answer, identify and explain any steps that represent detours, errors, backtracking, or any other failure patterns, highlighting why they might have occurred and what their impact was on the trajectory's progress. Discuss how the agent's tool-using knowledge and meta-reasoning skills handled or mitigated these issues.

3. Maintain all the core outcomes of each turn or step, even if it was part of a flawed process.

4. Thinking with images actions (if intermediate images are provided):
   - Document any image preprocessing operations (cropping, filtering, enhancement, etc.) or image searching (searching the web for relevant images, etc.) operations, these operations generated intermediate images. You need to analyze and note their impact.
   - For intermediate images that were generated and used: identify which visual features were extracted and how they can help the agent to reason better.
   - Suggest specific visual operations that could improve reasoning. If no intermediate images were generated and used in some rollout steps, but could have been helpful, note this as a potential point of improvement.

<trajectory>
{trajectory}
</trajectory>

Provide a clear, structured summary of the trajectory."""


# Prompt 2: Critiques multiple trajectories to generate new, generalizable experiences about tool use.
INTRA_SAMPLE_CRITIQUE = """You are a reasoning analysis expert. Review these problem-solving attempts and extract practical lessons that could help future attempts.

### Analysis Framework:

1. **Trajectory Review**:
   - What worked? What key decisions or insights led to correct answers?
   - What didn't work? Where did reasoning go wrong, and why?
   - Were any provided experiences missed or not helpful enough?
   - How were different tools combined? What sequences were effective?
   - For visual operations: What preprocessing helped? What was missing?

2. **Experience Extraction**:
   Create structured experiences in two outcome categories:
   - **success**: What worked in correct or partially useful trajectories.
   - **failure**: What failed in wrong trajectories and how future agents should avoid it.

   The two categories must use the same JSON format. Include failure experiences when a trajectory answered incorrectly, skipped a needed tool, used the wrong tool, misread visual evidence, or applied an experience/skill poorly.

   You have two options: [add, modify]
   - **add**: A genuinely new lesson not covered by existing experiences.
   - **modify**: Improve an existing experience (reference its ID) if you find it could be more accurate, clearer, or more actionable based on this trajectory.

   The `<experiences_used>` section below shows the experiences that were used to guide this sample. Review them against the trajectory outcomes to identify potential improvements.

   You can apply at most {max_ops} updates for this case.

3. **Experience Quality**:
   - Keep each `content` under 64 words.
   - Start with the situation or condition when the advice applies.
   - Make it general enough to apply to similar problems.
   - Focus on actionable guidance, not abstract principles.
   - Avoid specific examples - the experience should generalize.
   - Use short lowercase tags for `task_tags` and `tool_tags`.
   - Use `visual_tags` for reusable visual states such as "small_pointer", "maze_grid", "low_contrast", "tiny_text".
   - Use `action_policy` to state the concrete next action/tool policy, e.g. {"tool": "code_interpreter", "when": "path options are long", "do": "simulate each option"}.
   - Use `confidence` between 0 and 1 to indicate how reliable this lesson is.
   - Set `failure_mode` to null for success experiences. For failure experiences, use a concise label such as "shortcut_without_tool", "wrong_tool", "bad_crop", "wrong_scale_mapping", "misread_visual_evidence", or "misleading_memory".

Provide detailed reasoning following the above framework, then conclude with:
```json
[
  {{
    "option": "add",
    "kind": "success",
    "task_tags": ["task_type", "visual_operation"],
    "tool_tags": ["code_interpreter"],
    "visual_tags": ["visual_signal"],
    "failure_mode": null,
    "confidence": 0.8,
    "action_policy": {"tool": "code_interpreter", "when": "condition", "do": "specific reusable action"},
    "content": "the new generalizable success experience"
  }},
  {{
    "option": "modify",
    "kind": "failure",
    "task_tags": ["task_type"],
    "tool_tags": [],
    "visual_tags": ["visual_signal"],
    "failure_mode": "shortcut_without_tool",
    "confidence": 0.7,
    "action_policy": {"tool": "code_interpreter", "when": "direct visual judgment is risky", "do": "verify before final answer"},
    "content": "the modified failure-prevention experience",
    "modified_from": "E17"
  }},
  ...
]
```
Note: You may use only one type of update. Quality over quantity.

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


MERGE_PROMPT = """You are an experience library management expert. Merge the following experiences into a single, comprehensive experience.

[Experiences to merge]:
{experiences_text}

Requirements:
1. Contain all important information points from all experiences
2. Be clear, generalizable, and no more than 64 words
3. Maintain core lessons and decision points
4. Avoid redundancy while preserving unique insights

Output ONLY the merged experience text, no other text or comments.
"""


# Prompt 4: Refines the entire experience library after batch updates
EXPERIENCE_REFINE_PROMPT = """You are an experience library curator. The library has grown through multiple batches and may contain redundancy. Perform a global refinement pass.

### Current Library Size: {exp_count} experiences
### Target: Reduce to 80-100 high-quality, diverse experiences

### Refinement Goals:
1. **Merge Truly Redundant**: Combine experiences expressing the same core insight.
2. **Generalize Over-Specific**: Abstract experiences tied to specific scenarios into general patterns.
3. **Delete Low-Value or Too-Specific**: Remove experiences that are too obvious or rarely applicable, or too specific and only applicable to a single case or scenario.

### Operations:
- **merge**: Combine 2+ experiences into one (provide merged_from list)
- **delete**: Remove an experience entirely (provide deleted_id)

### Quality Standard:
- Under 64 words, clear and actionable, and generalizable to a class of problems.
- Starts with the trigger condition ("When...", "For...", "If...")
- Generalizable to a class of problems
- No specific examples or object names unless essential

### Important:
- Only act on clear redundancy or low quality. Preserve diversity.
- If the library is already well-curated, minimal changes are fine.

Output your reasoning, then:
```json
[
  {{"option": "merge", "experience": "merged text", "merged_from": ["E12", "E23"]}},
  {{"option": "delete", "deleted_id": "E45"}},
  ...
]
```

<experiences>
{experiences}
</experiences>
"""


# Prompt 5: A simple header for injecting experiences, framed for an agentic context.
INJECTION_HEADER = """Here are practical tips for tool-based visual reasoning, gathered from similar problems:

{bullets}

These experiences highlight common patterns and pitfalls. When you encounter matching situations, consider applying the suggested approaches. You can reference them by ID (e.g., [E12]) in your reasoning.

Your instruction is following:
"""
