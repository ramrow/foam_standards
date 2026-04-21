# config.py
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    max_loop: int = 25
    batchsize: int = 10
    searchdocs: int = 10
    run_times: int = 1
    database_path: str = Path(__file__).resolve().parent.parent / "database"
    run_directory: str = Path(__file__).resolve().parent.parent / "runs"
    case_dir: str = ""
    max_time_limit: int = 3600
    recursion_limit: int = 100
    input_writer_generation_mode: str = "sequential_dependency"
    reuse_generated_dir: str = ""

    # foambench-style multi-service config
    selected_service: str = "general"
    models: dict = field(default_factory=lambda: {
        "general": {
            "model_provider": "vllm",
            "model_version": "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
            "temperature": 0.5,
        },
    })

    # Backward-compatible single-model fields used by runtime
    model_provider: str = "vllm"
    model_version: str = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
    temperature: float = 0.5

    # Embedding config kept for retrieval path
    embedding_provider: str = "huggingface"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    def __post_init__(self) -> None:
        def _env_nonempty(key: str):
            v = os.getenv(key)
            return v.strip() if isinstance(v, str) and v.strip() else None

        # choose service profile first
        service_env = _env_nonempty("FOAMAGENT_MODEL_SERVICE")
        if service_env and service_env in self.models:
            self.selected_service = service_env

        if self.selected_service in self.models:
            prof = self.models[self.selected_service]
            self.model_provider = prof.get("model_provider", self.model_provider)
            self.model_version = prof.get("model_version", self.model_version)
            self.temperature = float(prof.get("temperature", self.temperature))

        # explicit overrides win
        provider_env = _env_nonempty("FOAMAGENT_MODEL_PROVIDER")
        version_env = _env_nonempty("FOAMAGENT_MODEL_VERSION")
        temp_env = _env_nonempty("FOAMAGENT_TEMPERATURE")

        if provider_env:
            self.model_provider = provider_env
        if version_env:
            self.model_version = version_env
        if temp_env:
            try:
                self.temperature = float(temp_env)
            except ValueError:
                pass

        print(f"[Config] selected_service={self.selected_service}")
        print(f"[Config] model_provider={self.model_provider}")
        print(f"[Config] model_version={self.model_version}")
        print(f"[Config] temperature={self.temperature}")
