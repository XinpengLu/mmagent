RECOVERY_SYNTHESIS_PROMPT = """You are a recovery-skill synthesizer for a multimodal tool-using agent.

Your job is to decide whether one failed experience can be repaired by one of
the success candidates, then produce a compact executable recovery candidate.

Principles:
1. A failure is recoverable only if it contains a locally useful step, partial
   evidence, useful tool input, code pattern, crop, measurement, validation
   attempt, or clear near miss.
2. Choose exactly one success prior only if it gives a concrete execution
   pattern that can repair the failure.
3. Do not invent task-specific answers, coordinates, object lists, or constants.
4. Keep the recovery general enough for similar tasks, but executable enough to
   guide tool use.
5. If no success candidate can repair the failure, return recoverable=false.
6. If the repair uses code_interpreter and can be expressed as reusable logic,
   include a compact runnable `code_template` that derives crops, masks,
   measurements, simulations, grids, calibrations, or option comparisons from
   the current image/problem. Leave it empty only when any code would require
   instance-specific constants.

Output JSON only:
```json
{{
  "recoverable": true,
  "source_success_id": "E12",
  "local_useful_steps": ["what the failed trajectory did correctly"],
  "failure_reason": "why the failure happened",
  "content": "When ..., repair the failed pattern by ...",
  "task_tags": ["short", "lowercase", "tags"],
  "tool_tags": ["code_interpreter"],
  "tool_sequence": [
    {{
      "tool": "code_interpreter",
      "purpose": "specific repaired tool action using current image/problem inputs",
      "expected_output": "visible intermediate evidence or numeric check"
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

If not recoverable:
```json
{{"recoverable": false, "reason": "brief reason"}}
```

<failure_experience>
{failure_experience}
</failure_experience>

<success_candidates>
{success_candidates}
</success_candidates>
"""
