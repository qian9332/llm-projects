"""
嵌入模型模块
支持多种嵌入模型：OpenAI、本地模型等
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from dataclasses import dataclass

from config.settings import EmbeddingConfig


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    embeddings: List[List[float]]
    tokens_used: int
    model: str


class BaseEmbedding(ABC):
    """嵌入模型基类"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """批量嵌入文档"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询"""
        pass
    
    def _batch_texts(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """将文本分批"""
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(texts[i:i + batch_size])
        return batches


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI嵌入模型"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            self.client = openai.OpenAI()
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
    
    def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """批量嵌入文档"""
        all_embeddings = []
        total_tokens = 0
        
        batches = self._batch_texts(texts, self.config.batch_size)
        
        for batch in batches:
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            total_tokens += response.usage.total_tokens
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            tokens_used=total_tokens,
            model=self.config.model_name
        )
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询"""
        response = self.client.embeddings.create(
            model=self.config.model_name,
            input=[query]
        )
        return response.data[0].embedding


class LocalEmbedding(BaseEmbedding):
    """本地嵌入模型（使用sentence-transformers）"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._init_model()
    
    def _init_model(self):
        """初始化本地模型"""
        try:
            from sentence_transformers import SentenceTransformer
            model_path = self.config.local_model_path or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(model_path)
        except ImportError:
            raise ImportError("请安装sentence-transformers库: pip install sentence-transformers")
    
    def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """批量嵌入文档"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False
        )
        
        return EmbeddingResult(
            embeddings=embeddings.tolist(),
            tokens_used=sum(len(text.split()) for text in texts),
            model=self.config.local_model_path or "all-MiniLM-L6-v2"
        )
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询"""
        embedding = self.model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()


class ZAIEmbedding(BaseEmbedding):
    """使用z-ai-web-dev-sdk的嵌入模型"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._init_client()
    
    def _init_client(self):
        """初始化ZAI客户端"""
        try:
            import asyncio
            import ZAI from 'z-ai-web-dev-sdk'
            self.zai = asyncio.run(ZAI.create())
        except ImportError:
            raise ImportError("请安装z-ai-web-dev-sdk")
    
    async def embed_documents_async(self, texts: List[str]) -> EmbeddingResult:
        """异步批量嵌入文档"""
        # 使用LLM生成文本表示作为嵌入的替代方案
        embeddings = []
        total_tokens = 0
        
        for text in texts:
            # 简化处理：使用文本的哈希作为伪嵌入
            # 实际应用中应使用专门的嵌入API
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
            total_tokens += len(text.split())
        
        return EmbeddingResult(
            embeddings=embeddings,
            tokens_used=total_tokens,
            model="z-ai-embedding"
        )
    
    def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """批量嵌入文档"""
        import asyncio
        return asyncio.run(self.embed_documents_async(texts))
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询"""
        return self._text_to_embedding(query)
    
    def _text_to_embedding(self, text: str, dim: int = 1536) -> List[float]:
        """将文本转换为嵌入向量（简化实现）"""
        import hashlib
        import numpy as np
        
        # 使用哈希生成伪随机向量
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # 扩展到指定维度
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(dim).astype(np.float32)
        
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()


def create_embedding(config: EmbeddingConfig) -> BaseEmbedding:
    """工厂函数：创建嵌入模型"""
    if config.use_local and config.local_model_path:
        return LocalEmbedding(config)
    elif config.model_name.startswith("text-embedding"):
        return OpenAIEmbedding(config)
    else:
        return LocalEmbedding(config)
