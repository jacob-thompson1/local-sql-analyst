"""Load and validate config.yaml."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import yaml


@dataclass
class Config:
    # Model
    model: str
    enable_thinking: bool
    embedding_model: str
    ollama_host: str
    temperature_sql: float
    temperature_plan: float
    temperature_chart: float
    temperature_explain: float
    temperature_classify: float
    num_ctx: int

    # Paths
    db_connection_string: str
    shared_drive_root: str
    ddl_file: str
    notes_file: str
    file_index_db: str
    output_folder: str
    data_folder: str

    # Retrieval
    schema_top_k: int
    examples_top_k: int

    # Execution
    query_timeout_sec: int
    max_rows: int
    default_top: int
    sql_retry_limit: int

    # UX
    require_plan_approval: bool
    require_sql_approval: bool
    auto_open_excel: bool

    # Identity
    user_id: str

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Config file not found: {p.resolve()}. "
                f"Copy config.yaml.example to config.yaml and edit it."
            )
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls(**data)
        cfg._ensure_dirs()
        return cfg

    def _ensure_dirs(self):
        for d in (self.output_folder, self.data_folder):
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def embeddings_cache_path(self) -> str:
        return str(Path(self.data_folder) / "embeddings_cache.db")

    @property
    def examples_db_path(self) -> str:
        return str(Path(self.data_folder) / "examples.db")

    @property
    def schema_cache_path(self) -> str:
        return str(Path(self.data_folder) / "schema_chunks.json")

    @property
    def query_log_path(self) -> str:
        return str(Path(self.data_folder) / "query_log.jsonl")
