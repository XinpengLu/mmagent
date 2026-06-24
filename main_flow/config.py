import json
import os
import yaml

from engine.api_processors import set_global_prompts as set_processors_prompts


DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 1.0


class ExperimentConfig:
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

    def _configure_experience_batching(self):
        if self.args.phase != "train":
            return
        
        small_batch = self.args.rollouts_per_sample
        large_batch = self.args.experience_large_batch

        if large_batch % small_batch != 0:
            raise ValueError(
                f"experience_large_batch ({large_batch}) must be a multiple of "
                f"rollouts_per_sample ({small_batch})"
            )
        self.args.experience_samples_per_large_batch = large_batch // small_batch

    def _load_prompts(self):
        with open(self.args.system_prompts_path, "r", encoding="utf-8") as f:
            prompts_yaml = yaml.safe_load(f)
        self.base_system_prompt = prompts_yaml["system_prompts"][self.args.system_prompt_key]
        set_processors_prompts(self.base_system_prompt)


    def _load_tool_configs(self) -> None:
        with open(self.args.tool_config_path, "r", encoding="utf-8") as f:
            self.args.tool_configs = yaml.safe_load(f) or {}

    def _load_data(self) -> None:
        with open(self.args.input_file, "r", encoding="utf-8") as f:
            if self.args.input_file.endswith(".jsonl"):
                self.data = [json.loads(line) for line in f if line.strip()]
            else:
                self.data = json.load(f)

        original_count = len(self.data)
        max_samples = self.args.max_samples

        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
            print(
                f"Loaded {original_count} samples, limiting to {len(self.data)} "
                f"samples (--max-samples={max_samples})."
            )
        else:
            print(f"Loaded {original_count} samples.")

    def _build_sampling_params(self):
        self.sampling_params = {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "max_tokens": int(self.args.max_tokens),
        }
