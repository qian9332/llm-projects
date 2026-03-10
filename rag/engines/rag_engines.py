"""
RAG引擎模块
实现多种RAG引擎：ContextChatEngine、CondensePlusContextMode等
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

from core.vector_store import BaseVectorStore, Document
from core.embeddings import BaseEmbedding
from core.retriever import BaseRetriever, VectorRetriever, RetrievalResult
from core.generator import BaseGenerator, RAGGenerator, GenerationResult
from config.settings import RAGConfig


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    """聊天会话"""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append(ChatMessage(role=role, content=content))
    
    def get_history(self, max_turns: int = None) -> List[ChatMessage]:
        """获取历史消息"""
        if max_turns is None:
            return self.messages
        return self.messages[-max_turns * 2:]  # 包含用户和助手的消息


class BaseRAGEngine(ABC):
    """RAG引擎基类"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        config: RAGConfig
    ):
        self.retriever = retriever
        self.generator = generator
        self.config = config
        self.rag_generator = RAGGenerator(generator)
    
    @abstractmethod
    def query(self, question: str, session: ChatSession = None) -> GenerationResult:
        """处理查询"""
        pass
    
    @abstractmethod
    async def query_async(self, question: str, session: ChatSession = None) -> GenerationResult:
        """异步处理查询"""
        pass


class ContextChatEngine(BaseRAGEngine):
    """
    ContextChatEngine - 上下文聊天引擎
    
    一种流行且简单的方法，通过检索与用户查询相关的上下文，
    并将其与聊天历史记录一起发送给语言模型。
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        config: RAGConfig
    ):
        super().__init__(retriever, generator, config)
        self.max_history_turns = config.max_history_turns
    
    def query(self, question: str, session: ChatSession = None) -> GenerationResult:
        """处理查询"""
        # 1. 检索相关上下文
        retrieval_result = self.retriever.retrieve(question)
        
        # 2. 构建包含历史的上下文
        context = self._build_context_with_history(retrieval_result, session)
        
        # 3. 生成回答
        answer = self.generator.generate(question, context)
        
        # 4. 构建结果
        sources = self._extract_sources(retrieval_result)
        
        return GenerationResult(
            query=question,
            answer=answer,
            sources=sources,
            metadata={
                "engine": "ContextChatEngine",
                "has_history": session is not None and len(session.messages) > 0,
                "context_length": len(context)
            }
        )
    
    async def query_async(self, question: str, session: ChatSession = None) -> GenerationResult:
        """异步处理查询"""
        retrieval_result = self.retriever.retrieve(question)
        context = self._build_context_with_history(retrieval_result, session)
        answer = await self.generator.generate_async(question, context)
        sources = self._extract_sources(retrieval_result)
        
        return GenerationResult(
            query=question,
            answer=answer,
            sources=sources,
            metadata={
                "engine": "ContextChatEngine",
                "has_history": session is not None and len(session.messages) > 0,
                "context_length": len(context)
            }
        )
    
    def _build_context_with_history(
        self,
        retrieval_result: RetrievalResult,
        session: ChatSession = None
    ) -> str:
        """构建包含历史的上下文"""
        parts = []
        
        # 添加聊天历史
        if session and session.messages:
            history = session.get_history(self.max_history_turns)
            if history:
                history_text = "\n".join([
                    f"{'用户' if msg.role == 'user' else '助手'}：{msg.content}"
                    for msg in history
                ])
                parts.append(f"=== 对话历史 ===\n{history_text}")
        
        # 添加检索到的上下文
        if retrieval_result.results:
            context_text = "\n\n".join([
                f"[参考文档{i+1}]\n{result.document.content}"
                for i, result in enumerate(retrieval_result.results)
            ])
            parts.append(f"=== 相关上下文 ===\n{context_text}")
        
        return "\n\n".join(parts)
    
    def _extract_sources(self, retrieval_result: RetrievalResult) -> List[Dict[str, Any]]:
        """提取来源信息"""
        return [
            {
                "content": result.document.content,
                "score": result.score,
                "rank": result.rank,
                "metadata": result.document.metadata
            }
            for result in retrieval_result.results
        ]


class CondensePlusContextEngine(BaseRAGEngine):
    """
    CondensePlusContextMode - 压缩+上下文引擎
    
    一种更复杂的方法，将聊天历史和最后一条消息压缩成新查询，
    以提高检索效率和生成答案的相关性。
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        config: RAGConfig
    ):
        super().__init__(retriever, generator, config)
        self.max_history_turns = config.max_history_turns
    
    def query(self, question: str, session: ChatSession = None) -> GenerationResult:
        """处理查询"""
        # 1. 压缩查询（结合历史）
        condensed_query = self._condense_query(question, session)
        
        # 2. 使用压缩后的查询进行检索
        retrieval_result = self.retriever.retrieve(condensed_query)
        
        # 3. 构建上下文
        context = self._build_context(retrieval_result)
        
        # 4. 生成回答
        answer = self.generator.generate(question, context)
        
        # 5. 构建结果
        sources = self._extract_sources(retrieval_result)
        
        return GenerationResult(
            query=question,
            answer=answer,
            sources=sources,
            metadata={
                "engine": "CondensePlusContextEngine",
                "condensed_query": condensed_query,
                "has_history": session is not None and len(session.messages) > 0,
                "context_length": len(context)
            }
        )
    
    async def query_async(self, question: str, session: ChatSession = None) -> GenerationResult:
        """异步处理查询"""
        condensed_query = self._condense_query(question, session)
        retrieval_result = self.retriever.retrieve(condensed_query)
        context = self._build_context(retrieval_result)
        answer = await self.generator.generate_async(question, context)
        sources = self._extract_sources(retrieval_result)
        
        return GenerationResult(
            query=question,
            answer=answer,
            sources=sources,
            metadata={
                "engine": "CondensePlusContextEngine",
                "condensed_query": condensed_query,
                "has_history": session is not None and len(session.messages) > 0,
                "context_length": len(context)
            }
        )
    
    def _condense_query(self, question: str, session: ChatSession = None) -> str:
        """
        压缩查询：将历史对话和当前问题压缩成一个独立的查询
        """
        if not session or not session.messages:
            return question
        
        history = session.get_history(self.max_history_turns)
        if not history:
            return question
        
        # 构建压缩提示
        history_text = "\n".join([
            f"{'用户' if msg.role == 'user' else '助手'}：{msg.content}"
            for msg in history
        ])
        
        condense_prompt = f"""根据以下对话历史和最新问题，生成一个独立的、包含所有必要信息的问题。
要求：
1. 新问题应该能够独立理解，不需要参考对话历史
2. 保留原始问题中的所有关键信息
3. 如果问题涉及之前讨论的内容，需要明确指出

对话历史：
{history_text}

最新问题：{question}

独立问题："""
        
        # 使用生成器压缩查询
        condensed = self.generator.generate(question, condense_prompt)
        
        # 清理结果
        condensed = condensed.strip()
        if condensed.startswith("独立问题："):
            condensed = condensed[5:].strip()
        
        return condensed
    
    def _build_context(self, retrieval_result: RetrievalResult) -> str:
        """构建上下文"""
        if not retrieval_result.results:
            return ""
        
        return "\n\n".join([
            f"[参考文档{i+1}]\n{result.document.content}"
            for i, result in enumerate(retrieval_result.results)
        ])
    
    def _extract_sources(self, retrieval_result: RetrievalResult) -> List[Dict[str, Any]]:
        """提取来源信息"""
        return [
            {
                "content": result.document.content,
                "score": result.score,
                "rank": result.rank,
                "metadata": result.document.metadata
            }
            for result in retrieval_result.results
        ]


class HybridRAGEngine(BaseRAGEngine):
    """
    HybridRAGEngine - 混合RAG引擎
    
    结合ContextChatEngine和CondensePlusContextEngine的优点
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        config: RAGConfig
    ):
        super().__init__(retriever, generator, config)
        self.context_engine = ContextChatEngine(retriever, generator, config)
        self.condense_engine = CondensePlusContextEngine(retriever, generator, config)
    
    def query(self, question: str, session: ChatSession = None) -> GenerationResult:
        """处理查询"""
        # 根据是否有历史选择引擎
        if session and len(session.messages) > 2:
            # 有较多历史时使用压缩引擎
            return self.condense_engine.query(question, session)
        else:
            # 否则使用上下文引擎
            return self.context_engine.query(question, session)
    
    async def query_async(self, question: str, session: ChatSession = None) -> GenerationResult:
        """异步处理查询"""
        if session and len(session.messages) > 2:
            return await self.condense_engine.query_async(question, session)
        else:
            return await self.context_engine.query_async(question, session)


def create_rag_engine(
    retriever: BaseRetriever,
    generator: BaseGenerator,
    config: RAGConfig
) -> BaseRAGEngine:
    """工厂函数：创建RAG引擎"""
    engine_type = config.engine_type
    
    if engine_type == "condense":
        return CondensePlusContextEngine(retriever, generator, config)
    elif engine_type == "hybrid":
        return HybridRAGEngine(retriever, generator, config)
    else:
        return ContextChatEngine(retriever, generator, config)
