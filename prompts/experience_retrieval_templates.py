TASK_DECOMPOSITION_PROMPT = """You are an Expert Visual Reasoning Strategist. Deconstruct this visual task into 2-3 distinct subtasks to retrieve relevant methodological guidance.

Task: {task_description}

For each subtask, generate a JSON object with:
- Type: A concise category for the methodological phase.
- Query: A methodology-focused search query targeting one of:
    1. Tool Utilization: Best practices for using specific tools effectively.
    2. Reasoning Strategy: Frameworks or solution paths for the task.
    3. Challenge Mitigation: Techniques to handle anticipated visual or reasoning challenges.

CRITICAL: Abstract away from specific image details and focus on the underlying technical challenge.

Output ONLY a valid JSON array:

[
  {{"type": "visual_extraction", "query": "Techniques for analyzing dark images using Code Interpreter"}},
  {{"type": "logic_synthesis", "query": "Frameworks for mapping visual features to knowledge using Web Search"}}
]"""


IMAGE_DESCRIPTION_PROMPT = """Provide a compact visual description of this image for later trajectory summarization. Focus only on task-relevant objects, spatial relationships, visible measurements/text, masks/crops/overlays, and whether the image supports or rejects an intermediate answer. Use one English paragraph, no headings or bullets, no problem-solving steps, no more than 120 words."""


RETRIEVAL_IMAGE_DESCRIPTION_PROMPT = """Describe the visual characteristics of this image in 1-2 sentences for retrieval purposes (lighting, objects, layout, challenges)."""


EXPERIENCE_REWRITE_SYSTEM_PROMPT = """You are an expert at adapting generalizable experiences to specific task contexts."""


EXPERIENCE_REWRITE_PROMPT = """Rewrite retrieved experiences to provide actionable guidance for this specific task.

Task: {task_description}

Retrieved experiences:
{experiences_text}

Rewriting guidelines:
1. Adapt the advice to be relevant to the current task type, but keep it reusable; do not embed task-specific details (object names, answer options, exact values).
2. Preserve the [kind; ...] metadata prefix exactly as-is.
3. If the experience warns about a failure mode, frame the rewrite as a guardrail.
4. Use suggestive language ("Consider checking...", "It is effective to...").
5. Do not invent new methods or algorithms not present in the retrieved experience.

If an experience is irrelevant, omit it from the output.

Output ONLY a valid JSON object mapping experience IDs to rewritten text:

{{
  "id1": "[failure; task=...] rewritten experience 1",
  "id2": "[success; task=...] rewritten experience 2"
}}"""


INJECTION_HEADER = """Here are practical tips for tool-based visual reasoning, gathered from similar problems:

{bullets}

Consider applying these when you encounter matching situations. Reference them by ID (e.g., [E12]) in your reasoning.
Treat failure experiences as guardrails: if your planned step matches the failure condition, revise before executing.

Your instruction is following:
"""
