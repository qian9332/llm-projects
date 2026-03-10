"""
向量数据库模块
支持多种向量数据库：Chroma、FAISS、Milvus等
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import json
import os

from config.settings import VectorStoreConfig


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """检索结果"""
    document: Document
    score: float
    rank: int


class BaseVectorStore(ABC):
    """向量数据库基类"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = None) -> List[SearchResult]:
        """相似度搜索"""
        pass
    
    @abstractmethod
    def delete(self, doc_ids: List[str]) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def get(self, doc_id: str) -> Optional[Document]:
        """获取单个文档"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """获取文档数量"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Chroma向量数据库"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self._init_client()
    
    def _init_client(self):
        """初始化Chroma客户端"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            raise ImportError("请安装chromadb库: pip install chromadb")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到Chroma"""
        ids = []
        embeddings = []
        contents = []
        metadatas = []
        
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"文档 {doc.id} 缺少嵌入向量")
            
            ids.append(doc.id)
            embeddings.append(doc.embedding)
            contents.append(doc.content)
            metadatas.append(doc.metadata or {})
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        return ids
    
    def search(self, query_embedding: List[float], top_k: int = None) -> List[SearchResult]:
        """在Chroma中搜索"""
        top_k = top_k or self.config.top_k
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        for i, (doc_id, content, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # 将距离转换为相似度分数
            score = 1 - distance
            
            if score >= self.config.similarity_threshold:
                search_results.append(SearchResult(
                    document=Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata
                    ),
                    score=score,
                    rank=i + 1
                ))
        
        return search_results
    
    def delete(self, doc_ids: List[str]) -> bool:
        """从Chroma删除文档"""
        self.collection.delete(ids=doc_ids)
        return True
    
    def get(self, doc_id: str) -> Optional[Document]:
        """从Chroma获取文档"""
        results = self.collection.get(ids=[doc_id])
        if results['ids']:
            return Document(
                id=results['ids'][0],
                content=results['documents'][0],
                metadata=results['metadatas'][0]
            )
        return None
    
    def count(self) -> int:
        """获取Chroma中的文档数量"""
        return self.collection.count()


class FAISSVectorStore(BaseVectorStore):
    """FAISS向量数据库"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.documents: Dict[str, Document] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.index = None
        self.dimension = None
        self._init_store()
    
    def _init_store(self):
        """初始化FAISS存储"""
        try:
            import faiss
            
            self.faiss = faiss
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # 尝试加载已有索引
            index_path = os.path.join(self.config.persist_directory, "index.faiss")
            meta_path = os.path.join(self.config.persist_directory, "metadata.json")
            
            if os.path.exists(index_path) and os.path.exists(meta_path):
                self.index = faiss.read_index(index_path)
                self.dimension = self.index.d
                
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self.id_to_idx = meta.get('id_to_idx', {})
                    self.idx_to_id = {int(k): v for k, v in meta.get('idx_to_id', {}).items()}
                    # 重建documents字典
                    docs_path = os.path.join(self.config.persist_directory, "documents.json")
                    if os.path.exists(docs_path):
                        with open(docs_path, 'r', encoding='utf-8') as df:
                            docs_data = json.load(df)
                            self.documents = {
                                k: Document(**v) for k, v in docs_data.items()
                            }
                            
        except ImportError:
            raise ImportError("请安装faiss库: pip install faiss-cpu 或 pip install faiss-gpu")
    
    def _save_store(self):
        """保存FAISS索引和元数据"""
        if self.index is not None:
            index_path = os.path.join(self.config.persist_directory, "index.faiss")
            self.faiss.write_index(self.index, index_path)
            
            meta_path = os.path.join(self.config.persist_directory, "metadata.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'id_to_idx': self.id_to_idx,
                    'idx_to_id': self.idx_to_id
                }, f)
            
            docs_path = os.path.join(self.config.persist_directory, "documents.json")
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump({
                    k: {'id': v.id, 'content': v.content, 'metadata': v.metadata}
                    for k, v in self.documents.items()
                }, f, ensure_ascii=False, indent=2)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到FAISS"""
        if not documents:
            return []
        
        # 检查维度
        first_embedding = documents[0].embedding
        if first_embedding is None:
            raise ValueError("文档缺少嵌入向量")
        
        dim = len(first_embedding)
        
        # 初始化索引
        if self.index is None:
            self.dimension = dim
            self.index = self.faiss.IndexFlatIP(dim)  # 内积相似度
        
        # 添加文档
        ids = []
        embeddings = []
        
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"文档 {doc.id} 缺少嵌入向量")
            
            idx = len(self.documents)
            self.documents[doc.id] = doc
            self.id_to_idx[doc.id] = idx
            self.idx_to_id[idx] = doc.id
            ids.append(doc.id)
            embeddings.append(doc.embedding)
        
        # 添加到索引
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.faiss.normalize_L2(embeddings_array)  # 归一化
        self.index.add(embeddings_array)
        
        self._save_store()
        return ids
    
    def search(self, query_embedding: List[float], top_k: int = None) -> List[SearchResult]:
        """在FAISS中搜索"""
        top_k = top_k or self.config.top_k
        
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        self.faiss.normalize_L2(query_array)
        
        scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        search_results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and score >= self.config.similarity_threshold:
                doc_id = self.idx_to_id.get(int(idx))
                if doc_id and doc_id in self.documents:
                    search_results.append(SearchResult(
                        document=self.documents[doc_id],
                        score=float(score),
                        rank=i + 1
                    ))
        
        return search_results
    
    def delete(self, doc_ids: List[str]) -> bool:
        """从FAISS删除文档（需要重建索引）"""
        for doc_id in doc_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                if doc_id in self.id_to_idx:
                    del self.id_to_idx[doc_id]
        
        # 重建索引
        if self.documents:
            embeddings = []
            self.idx_to_id = {}
            for idx, (doc_id, doc) in enumerate(self.documents.items()):
                if doc.embedding:
                    embeddings.append(doc.embedding)
                    self.id_to_idx[doc_id] = idx
                    self.idx_to_id[idx] = doc_id
            
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.faiss.normalize_L2(embeddings_array)
                self.index = self.faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings_array)
        else:
            self.index = self.faiss.IndexFlatIP(self.dimension)
        
        self._save_store()
        return True
    
    def get(self, doc_id: str) -> Optional[Document]:
        """从FAISS获取文档"""
        return self.documents.get(doc_id)
    
    def count(self) -> int:
        """获取FAISS中的文档数量"""
        return len(self.documents)


class SimpleVectorStore(BaseVectorStore):
    """简单的内存向量存储（用于测试和小规模应用）"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.documents: Dict[str, Document] = {}
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档"""
        ids = []
        for doc in documents:
            self.documents[doc.id] = doc
            ids.append(doc.id)
        return ids
    
    def search(self, query_embedding: List[float], top_k: int = None) -> List[SearchResult]:
        """余弦相似度搜索"""
        top_k = top_k or self.config.top_k
        
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        scores = []
        for doc_id, doc in self.documents.items():
            if doc.embedding:
                doc_vec = np.array(doc.embedding)
                doc_norm = np.linalg.norm(doc_vec)
                if doc_norm > 0:
                    similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
                    scores.append((doc_id, similarity))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (doc_id, score) in enumerate(scores[:top_k]):
            if score >= self.config.similarity_threshold:
                results.append(SearchResult(
                    document=self.documents[doc_id],
                    score=score,
                    rank=i + 1
                ))
        
        return results
    
    def delete(self, doc_ids: List[str]) -> bool:
        """删除文档"""
        for doc_id in doc_ids:
            self.documents.pop(doc_id, None)
        return True
    
    def get(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        return self.documents.get(doc_id)
    
    def count(self) -> int:
        """获取文档数量"""
        return len(self.documents)


def create_vector_store(config: VectorStoreConfig) -> BaseVectorStore:
    """工厂函数：创建向量数据库"""
    if config.store_type == "chroma":
        return ChromaVectorStore(config)
    elif config.store_type == "faiss":
        return FAISSVectorStore(config)
    else:
        return SimpleVectorStore(config)
