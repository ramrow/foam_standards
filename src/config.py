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

    selected_service: str = "general"
    models: dict = field(default_factory=lambda: {
        "general": {
            "model_provider": "bedrock",
            "model_version": "arn:aws:bedrock:us-west-2:567316078106:inference-profile/us.anthropic.claude-sonnet-4-6",
            "temperature": 0.6,
            "max_tokens": 90000,
        },
        "input-writer": {
            "model_provider": "vllm",
            "model_version": "foamqwen",
            "temperature": 0.6,
            "base_url": "http://127.0.0.1:8000/v1",
            "max_tokens": 90000,
        },
    })

    model_provider: str = "bedrock"
    model_version: str = "arn:aws:bedrock:us-west-2:567316078106:inference-profile/us.anthropic.claude-sonnet-4-6"
    temperature: float = 0.6

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        def _env_nonempty(key: str):
            v = os.getenv(key)
            return v.strip() if isinstance(v, str) and v.strip() else None

        service_env = _env_nonempty("FOAMAGENT_MODEL_SERVICE")
        if service_env and service_env in self.models:
            self.selected_service = service_env

        if self.selected_service in self.models:
            prof = self.models[self.selected_service]
            self.model_provider = prof.get("model_provider", self.model_provider)
            self.model_version = prof.get("model_version", self.model_version)
            self.temperature = float(prof.get("temperature", self.temperature))

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


