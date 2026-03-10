"""
RAG项目 - 检索增强生成系统
"""

from core.rag_system import RAGSystem, RAGSystemBuilder, create_rag_system
from config.settings import RAGConfig, default_config

__version__ = "1.0.0"
__author__ = "qian9332"

__all__ = [
    "RAGSystem",
    "RAGSystemBuilder", 
    "create_rag_system",
    "RAGConfig",
    "default_config"
]
