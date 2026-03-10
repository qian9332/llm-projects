"""
检索模块
实现多种检索策略：向量检索、混合检索、重排序等
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from core.vector_store import BaseVectorStore, Document, SearchResult
from core.embeddings import BaseEmbedding
from config.settings import RAGConfig


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    results: List[SearchResult]
    metadata: Dict[str, Any] = None


class BaseRetriever(ABC):
    """检索器基类"""
    
    def __init__(self, vector_store: BaseVectorStore, embedding_model: BaseEmbedding):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """检索相关文档"""
        pass


class VectorRetriever(BaseRetriever):
    """向量检索器"""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbedding,
        top_k: int = 5
    ):
        super().__init__(vector_store, embedding_model)
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """向量检索"""
        top_k = top_k or self.top_k
        
        # 生成查询向量
        query_embedding = self.embedding_model.embed_query(query)
        
        # 向量搜索
        results = self.vector_store.search(query_embedding, top_k)
        
        return RetrievalResult(
            query=query,
            results=results,
            metadata={
                "top_k": top_k,
                "total_results": len(results)
            }
        )


class HybridRetriever(BaseRetriever):
    """混合检索器（向量 + 关键词）"""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbedding,
        top_k: int = 5,
        alpha: float = 0.5
    ):
        super().__init__(vector_store, embedding_model)
        self.top_k = top_k
        self.alpha = alpha  # 向量检索权重
    
    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """混合检索"""
        top_k = top_k or self.top_k
        
        # 向量检索
        query_embedding = self.embedding_model.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k * 2)
        
        # 关键词检索
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # 融合结果
        merged_results = self._merge_results(vector_results, keyword_results, top_k)
        
        return RetrievalResult(
            query=query,
            results=merged_results,
            metadata={
                "top_k": top_k,
                "vector_results": len(vector_results),
                "keyword_results": len(keyword_results),
                "alpha": self.alpha
            }
        )
    
    def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        """简单的关键词搜索"""
        keywords = set(query.lower().split())
        all_docs = []
        
        # 获取所有文档（简化实现）
        # 实际应用中应使用全文搜索引擎
        for i in range(self.vector_store.count()):
            doc = self.vector_store.get(str(i))
            if doc:
                all_docs.append(doc)
        
        # 计算关键词匹配分数
        scores = []
        for doc in all_docs:
            doc_words = set(doc.content.lower().split())
            overlap = len(keywords & doc_words)
            if overlap > 0:
                score = overlap / len(keywords)
                scores.append((doc, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(document=doc, score=score, rank=i + 1)
            for i, (doc, score) in enumerate(scores[:top_k])
        ]
    
    def _merge_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """融合向量检索和关键词检索结果"""
        # 使用RRF（Reciprocal Rank Fusion）算法
        doc_scores: Dict[str, float] = {}
        doc_info: Dict[str, SearchResult] = {}
        
        # 向量检索分数
        for result in vector_results:
            doc_id = result.document.id
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.alpha / (result.rank + 60)
            doc_info[doc_id] = result
        
        # 关键词检索分数
        for result in keyword_results:
            doc_id = result.document.id
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (1 - self.alpha) / (result.rank + 60)
            if doc_id not in doc_info:
                doc_info[doc_id] = result
        
        # 排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(
                document=doc_info[doc_id].document,
                score=score,
                rank=i + 1
            )
            for i, (doc_id, score) in enumerate(sorted_docs[:top_k])
        ]


class RerankingRetriever(BaseRetriever):
    """带重排序的检索器"""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbedding,
        base_retriever: BaseRetriever = None,
        top_k: int = 5,
        rerank_top_k: int = 20
    ):
        super().__init__(vector_store, embedding_model)
        self.base_retriever = base_retriever or VectorRetriever(
            vector_store, embedding_model, rerank_top_k
        )
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
    
    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """检索并重排序"""
        top_k = top_k or self.top_k
        
        # 基础检索
        base_result = self.base_retriever.retrieve(query, self.rerank_top_k)
        
        # 重排序
        reranked_results = self._rerank(query, base_result.results, top_k)
        
        return RetrievalResult(
            query=query,
            results=reranked_results,
            metadata={
                "top_k": top_k,
                "rerank_top_k": self.rerank_top_k,
                "base_results": len(base_result.results)
            }
        )
    
    def _rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """重排序（使用交叉编码器或相似度重计算）"""
        # 获取查询向量
        query_embedding = np.array(self.embedding_model.embed_query(query))
        query_norm = np.linalg.norm(query_embedding)
        
        reranked = []
        for result in results:
            if result.document.embedding:
                doc_embedding = np.array(result.document.embedding)
                doc_norm = np.linalg.norm(doc_embedding)
                
                if query_norm > 0 and doc_norm > 0:
                    # 计算余弦相似度
                    similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
                else:
                    similarity = result.score
                
                reranked.append(SearchResult(
                    document=result.document,
                    score=float(similarity),
                    rank=0
                ))
        
        # 排序
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # 更新排名
        for i, result in enumerate(reranked[:top_k]):
            result.rank = i + 1
        
        return reranked[:top_k]


def create_retriever(
    vector_store: BaseVectorStore,
    embedding_model: BaseEmbedding,
    config: RAGConfig
) -> BaseRetriever:
    """工厂函数：创建检索器"""
    retriever_type = getattr(config, 'retriever_type', 'vector')
    
    if retriever_type == 'hybrid':
        return HybridRetriever(
            vector_store, embedding_model,
            top_k=config.vector_store.top_k
        )
    elif retriever_type == 'reranking':
        return RerankingRetriever(
            vector_store, embedding_model,
            top_k=config.vector_store.top_k
        )
    else:
        return VectorRetriever(
            vector_store, embedding_model,
            top_k=config.vector_store.top_k
        )
