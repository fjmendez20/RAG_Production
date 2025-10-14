import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbeddingConfig:
    # Modelo más pequeño y rápido
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True

@dataclass
class SplitterConfig:
    chunk_size: int = 400  # Aumentado para mejor contexto
    chunk_overlap: int = 50  # Aumentado para mejor continuidad
    separators: tuple = ("\n\n", "\n", ". ", "? ", "! ", " ", "")

@dataclass
class VectorStoreConfig:
    # Usar path dentro del contenedor
    persist_directory: str = os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db")
    collection_name: str = "production_documents"
    similarity_metric: str = "cosine"

@dataclass
class SearchConfig:
    default_n_results: int = 4
    similarity_threshold: float = 0.3  # Más bajo para mejor recall
    enable_fallback: bool = True
    max_fallback_threshold: float = 0.1

@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    # Configuración general de la app
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")