import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True

@dataclass
class SplitterConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: tuple = ("\n\n", "\n", " ", "")

@dataclass
class VectorStoreConfig:
    # En producción, usar directorio temporal
    persist_directory: str = "/tmp/chroma_db"
    collection_name: str = "production_documents"
    similarity_metric: str = "cosine"

@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)