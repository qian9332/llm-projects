"""
RAG项目配置文件
包含模型配置、向量数据库配置、检索配置等
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model_name: str = "text-embedding-ada-002"
    dimension: int = 1536
    batch_size: int = 32
    max_tokens: int = 8191
    # 本地模型配置
    local_model_path: Optional[str] = None
    use_local: bool = False


@dataclass
class VectorStoreConfig:
    """向量数据库配置"""
    store_type: str = "chroma"  # chroma, faiss, milvus, pinecone
    persist_directory: str = "./data/vector_store"
    collection_name: str = "rag_collection"
    # 检索配置
    top_k: int = 5
    similarity_threshold: float = 0.7
    # FAISS配置
    faiss_index_type: str = "IVFFlat"
    nlist: int = 100


@dataclass
class LLMConfig:
    """大语言模型配置"""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    # API配置
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    # 本地模型配置
    local_model_path: Optional[str] = None
    use_local: bool = False


@dataclass
class TextProcessorConfig:
    """文本处理配置"""
    # 文本分割配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", "。", "！", "？", "；", " ", ""])
    # 语义分割配置
    use_semantic_split: bool = False
    semantic_threshold: float = 0.8


@dataclass
class RAGConfig:
    """RAG系统总配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    text_processor: TextProcessorConfig = field(default_factory=TextProcessorConfig)
    
    # RAG引擎配置
    engine_type: str = "context"  # context, condense, hybrid
    max_history_turns: int = 10
    context_window: int = 4096
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "./logs/rag.log"


def load_config_from_env() -> RAGConfig:
    """从环境变量加载配置"""
    config = RAGConfig()
    
    # API配置
    config.llm.api_key = os.getenv("OPENAI_API_KEY", config.llm.api_key)
    config.llm.api_base = os.getenv("OPENAI_API_BASE", config.llm.api_base)
    
    # 模型配置
    config.llm.model_name = os.getenv("LLM_MODEL_NAME", config.llm.model_name)
    config.embedding.model_name = os.getenv("EMBEDDING_MODEL_NAME", config.embedding.model_name)
    
    return config


# 默认配置实例
default_config = load_config_from_env()
