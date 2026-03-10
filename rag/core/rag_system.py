"""
RAG系统主模块
整合所有组件，提供统一的API接口
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
import os

from config.settings import RAGConfig, default_config
from core.embeddings import BaseEmbedding, create_embedding
from core.vector_store import BaseVectorStore, Document, create_vector_store
from core.retriever import BaseRetriever, create_retriever
from core.generator import BaseGenerator, create_generator, RAGGenerator
from engines.rag_engines import (
    BaseRAGEngine, ContextChatEngine, CondensePlusContextEngine,
    HybridRAGEngine, ChatSession, create_rag_engine
)
from utils.text_processor import TextProcessor


class RAGSystem:
    """
    RAG系统主类
    
    提供完整的RAG功能：
    - 文档索引
    - 问答查询
    - 多轮对话
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统
        
        Args:
            config: RAG配置，如果为None则使用默认配置
        """
        self.config = config or default_config
        self._init_components()
        
        # 会话管理
        self.sessions: Dict[str, ChatSession] = {}
    
    def _init_components(self):
        """初始化各组件"""
        # 嵌入模型
        self.embedding_model = create_embedding(self.config.embedding)
        
        # 向量数据库
        self.vector_store = create_vector_store(self.config.vector_store)
        
        # 检索器
        self.retriever = create_retriever(
            self.vector_store, self.embedding_model, self.config
        )
        
        # 生成器
        self.generator = create_generator(self.config.llm)
        
        # RAG引擎
        self.engine = create_rag_engine(
            self.retriever, self.generator, self.config
        )
        
        # 文本处理器
        self.text_processor = TextProcessor(self.config.text_processor)
    
    def index_documents(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        extract_qa: bool = False
    ) -> int:
        """
        索引文档
        
        Args:
            documents: 文档列表，每个元素为 (content, metadata) 元组
            extract_qa: 是否提取问答对
        
        Returns:
            索引的文档数量
        """
        # 处理文档
        chunks = self.text_processor.process_documents(documents, extract_qa)
        
        if not chunks:
            return 0
        
        # 生成嵌入向量
        texts = [chunk.content for chunk in chunks]
        embedding_result = self.embedding_model.embed_documents(texts)
        
        # 创建Document对象
        docs = []
        for chunk, embedding in zip(chunks, embedding_result.embeddings):
            doc = Document(
                id=chunk.id,
                content=chunk.content,
                embedding=embedding,
                metadata=chunk.metadata
            )
            docs.append(doc)
        
        # 添加到向量数据库
        self.vector_store.add_documents(docs)
        
        return len(docs)
    
    def index_from_files(
        self,
        file_paths: List[str],
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        从文件索引文档
        
        Args:
            file_paths: 文件路径列表
            metadata: 元数据
        
        Returns:
            索引的文档数量
        """
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_metadata = metadata.copy() if metadata else {}
            file_metadata['source'] = file_path
            file_metadata['filename'] = os.path.basename(file_path)
            
            documents.append((content, file_metadata))
        
        return self.index_documents(documents)
    
    def query(
        self,
        question: str,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        查询
        
        Args:
            question: 问题
            session_id: 会话ID，如果提供则支持多轮对话
        
        Returns:
            查询结果
        """
        # 获取或创建会话
        session = self._get_session(session_id)
        
        # 执行查询
        result = self.engine.query(question, session)
        
        # 更新会话
        if session:
            session.add_message("user", question)
            session.add_message("assistant", result.answer)
        
        return {
            "query": result.query,
            "answer": result.answer,
            "sources": result.sources,
            "metadata": result.metadata,
            "session_id": session.session_id if session else None
        }
    
    async def query_async(
        self,
        question: str,
        session_id: str = None
    ) -> Dict[str, Any]:
        """异步查询"""
        session = self._get_session(session_id)
        result = await self.engine.query_async(question, session)
        
        if session:
            session.add_message("user", question)
            session.add_message("assistant", result.answer)
        
        return {
            "query": result.query,
            "answer": result.answer,
            "sources": result.sources,
            "metadata": result.metadata,
            "session_id": session.session_id if session else None
        }
    
    def _get_session(self, session_id: str = None) -> Optional[ChatSession]:
        """获取或创建会话"""
        if session_id is None:
            return None
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id=session_id)
        
        return self.sessions[session_id]
    
    def create_session(self) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ChatSession(session_id=session_id)
        return session_id
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史"""
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        return [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
            for msg in session.messages
        ]
    
    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "document_count": self.vector_store.count(),
            "session_count": len(self.sessions),
            "config": {
                "embedding_model": self.config.embedding.model_name,
                "llm_model": self.config.llm.model_name,
                "vector_store_type": self.config.vector_store.store_type,
                "engine_type": self.config.engine_type
            }
        }


class RAGSystemBuilder:
    """
    RAG系统构建器
    提供流畅的API来构建RAG系统
    """
    
    def __init__(self):
        self.config = RAGConfig()
    
    def with_embedding_model(self, model_name: str, use_local: bool = False) -> 'RAGSystemBuilder':
        """设置嵌入模型"""
        self.config.embedding.model_name = model_name
        self.config.embedding.use_local = use_local
        return self
    
    def with_llm(self, model_name: str, use_local: bool = False) -> 'RAGSystemBuilder':
        """设置LLM"""
        self.config.llm.model_name = model_name
        self.config.llm.use_local = use_local
        return self
    
    def with_vector_store(self, store_type: str, persist_dir: str = None) -> 'RAGSystemBuilder':
        """设置向量数据库"""
        self.config.vector_store.store_type = store_type
        if persist_dir:
            self.config.vector_store.persist_directory = persist_dir
        return self
    
    def with_engine(self, engine_type: str) -> 'RAGSystemBuilder':
        """设置RAG引擎类型"""
        self.config.engine_type = engine_type
        return self
    
    def with_chunk_size(self, chunk_size: int, overlap: int = 50) -> 'RAGSystemBuilder':
        """设置文本块大小"""
        self.config.text_processor.chunk_size = chunk_size
        self.config.text_processor.chunk_overlap = overlap
        return self
    
    def with_top_k(self, top_k: int) -> 'RAGSystemBuilder':
        """设置检索数量"""
        self.config.vector_store.top_k = top_k
        return self
    
    def build(self) -> RAGSystem:
        """构建RAG系统"""
        return RAGSystem(self.config)


# 便捷函数
def create_rag_system(
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = "gpt-3.5-turbo",
    vector_store: str = "chroma",
    engine_type: str = "context"
) -> RAGSystem:
    """
    快速创建RAG系统
    
    Args:
        embedding_model: 嵌入模型名称
        llm_model: LLM模型名称
        vector_store: 向量数据库类型
        engine_type: RAG引擎类型
    
    Returns:
        RAG系统实例
    """
    return (RAGSystemBuilder()
            .with_embedding_model(embedding_model)
            .with_llm(llm_model)
            .with_vector_store(vector_store)
            .with_engine(engine_type)
            .build())
