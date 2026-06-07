"""Configuration loading for one benchmark phase."""

import json
import os

import yaml

from engine.api_processors import set_global_prompts as set_processors_prompts


class ExperimentConfig:
    """Loads static experiment inputs and normalizes runtime arguments."""

    def __init__(self, args):
        self.args = args
        self.base_system_prompt = ""
        self.data = []
        self.sampling_params = {}

    def load(self):
        self._load_model_config()
        self._configure_experience_batching()
        self._load_prompts()
        self._load_tool_configs()
        self._load_data()
        self._build_sampling_params()
        return self

    def _load_model_config(self):
        self.args.model_name = os.environ.get("REASONING_MODEL_NAME")
        if not self.args.model_name:
            raise ValueError("REASONING_MODEL_NAME environment variable must be set")
        print("Starting benchmark inference...")
        print(f"Reasoning model: {self.args.model_name}")

    def _configure_experience_batching(self):
        if getattr(self.args, "experience_online_generate", False):
            small_batch = self.args.rollouts_per_sample
            large_batch = getattr(self.args, "experience_large_batch", None) or small_batch
            self.args.experience_large_batch = large_batch
            if large_batch % small_batch != 0:
                raise ValueError(
                    f"experience_large_batch ({large_batch}) must be a multiple of "
                    f"rollouts_per_sample ({small_batch})"
                )
            self.args.experience_samples_per_large_batch = large_batch // small_batch
            print(f"\n{'=' * 80}")
            print("Experience Generation Batching:")
            print(f"  Small batch (rollouts per sample): {small_batch}")
            print(f"  Large batch (rollouts per batch): {large_batch}")
            print(f"  Samples per large batch: {self.args.experience_samples_per_large_batch}")
            print(f"{'=' * 80}\n")

        if self.args.rollouts_per_sample > 1:
            print(f"\n{'=' * 80}")
            print(f"Multi-rollout mode enabled: {self.args.rollouts_per_sample} rollouts per sample")
            print(f"Base seed: {self.args.seed_base}")
            print("This will enable pass@k and average@k evaluation")
            print(f"{'=' * 80}\n")

    def _load_prompts(self):
        with open(self.args.system_prompts_path, "r", encoding="utf-8") as f:
            prompts_yaml = yaml.safe_load(f)
        self.base_system_prompt = prompts_yaml["system_prompts"][self.args.system_prompt_key]
        set_processors_prompts(self.base_system_prompt)
        print("Inference prompts loaded successfully.")

    def _load_tool_configs(self):
        tool_config_path = getattr(self.args, "tool_config_path", "configs/tool_configs.yaml")
        if os.path.exists(tool_config_path):
            with open(tool_config_path, "r", encoding="utf-8") as f:
                self.args.tool_configs = yaml.safe_load(f) or {}
            print(f"Loaded tool configs from {tool_config_path}")
        else:
            self.args.tool_configs = {}
            print(f"Tool config file not found at {tool_config_path}, using defaults")

    def _load_data(self):
        if self.args.input_file.endswith(".jsonl"):
            with open(self.args.input_file, "r", encoding="utf-8") as f:
                self.data = [json.loads(line) for line in f]
        else:
            with open(self.args.input_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)

        original_count = len(self.data)
        if self.args.max_samples is not None and self.args.max_samples > 0:
            self.data = self.data[: self.args.max_samples]
            print(
                f"Loaded {original_count} samples, limiting to {len(self.data)} "
                f"samples (--max-samples={self.args.max_samples})."
            )
        else:
            print(f"Loaded {len(self.data)} samples.")

    def _build_sampling_params(self):
        self.sampling_params = {
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_tokens": self.args.max_completion_tokens,
        }
        if self.args.max_total_tokens <= self.args.max_completion_tokens:
            adjusted_total = self.args.max_completion_tokens + 4096
            print(
                f"Warning: max_total_tokens ({self.args.max_total_tokens}) must be larger than "
                f"max_completion_tokens ({self.args.max_completion_tokens}); using {adjusted_total}."
            )
            self.args.max_total_tokens = adjusted_total
