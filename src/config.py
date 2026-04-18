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

    # Single-model setup only
    selected_service: str = "general"
    models: dict = field(default_factory=lambda: {
        "general": {
            "model_provider": "huggingface",
            "model_version": "Qwen/Qwen3.5-35B-A3B",
            "temperature": 0.6,
            "base_url": "http://127.0.0.1:8000/v1",
            "max_tokens": 32768,
        }
    })

    model_provider: str = "huggingface"
    model_version: str = "Qwen/Qwen3.5-35B-A3B"
    temperature: float = 0.6

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        def _env_nonempty(key: str):
            v = os.getenv(key)
            return v.strip() if isinstance(v, str) and v.strip() else None

        # Always use general (single-model setup)
        self.selected_service = "general"

        prof = self.models["general"]
        self.model_provider = prof.get("model_provider", self.model_provider)
        self.model_version = prof.get("model_version", self.model_version)
        self.temperature = float(prof.get("temperature", self.temperature))

        provider_env = _env_nonempty("FOAMAGENT_MODEL_PROVIDER")
        version_env = _env_nonempty("FOAMAGENT_MODEL_VERSION")
        temp_env = _env_nonempty("FOAMAGENT_TEMPERATURE")
        base_url_env = _env_nonempty("FOAMAGENT_BASE_URL")

        if provider_env:
            self.model_provider = provider_env
        if version_env:
            self.model_version = version_env
            self.models["general"]["model_version"] = version_env
        if temp_env:
            try:
                self.temperature = float(temp_env)
                self.models["general"]["temperature"] = self.temperature
            except ValueError:
                pass
        if base_url_env:
            self.models["general"]["base_url"] = base_url_env

        print(f"[Config] selected_service={self.selected_service}")
        print(f"[Config] model_provider={self.model_provider}")
        print(f"[Config] model_version={self.model_version}")
        print(f"[Config] base_url={self.models['general']['base_url']}")
        print(f"[Config] temperature={self.temperature}")



