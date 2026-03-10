"""
文本处理工具模块
包含文本分割、语义分割、问答对提取等功能
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import re
import uuid

from config.settings import TextProcessorConfig


@dataclass
class TextChunk:
    """文本块"""
    id: str
    content: str
    metadata: dict
    start_index: int = 0
    end_index: int = 0


class TextSplitter:
    """文本分割器"""
    
    def __init__(self, config: TextProcessorConfig):
        self.config = config
    
    def split_text(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """分割文本"""
        if metadata is None:
            metadata = {}
        
        # 使用递归字符分割
        chunks = self._recursive_split(text)
        
        # 创建TextChunk对象
        text_chunks = []
        current_index = 0
        
        for chunk_content in chunks:
            # 查找文本中的位置
            start_idx = text.find(chunk_content, current_index)
            if start_idx == -1:
                start_idx = current_index
            end_idx = start_idx + len(chunk_content)
            
            chunk = TextChunk(
                id=str(uuid.uuid4()),
                content=chunk_content.strip(),
                metadata=metadata.copy(),
                start_index=start_idx,
                end_index=end_idx
            )
            text_chunks.append(chunk)
            current_index = end_idx
        
        return text_chunks
    
    def _recursive_split(self, text: str) -> List[str]:
        """递归分割文本"""
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []
        
        # 尝试按分隔符分割
        for separator in self.config.separators:
            if separator in text:
                parts = text.split(separator)
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    if len(current_chunk) + len(part) + len(separator) <= self.config.chunk_size:
                        current_chunk += part + separator
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        if len(part) > self.config.chunk_size:
                            # 递归处理过长的部分
                            sub_chunks = self._recursive_split(part)
                            chunks.extend(sub_chunks)
                        else:
                            current_chunk = part + separator
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                return self._merge_chunks(chunks)
        
        # 无法分割，强制按字符数分割
        return self._force_split(text)
    
    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的块"""
        merged = []
        current = ""
        
        for chunk in chunks:
            if len(current) + len(chunk) <= self.config.chunk_size:
                current += chunk
            else:
                if current:
                    merged.append(current)
                current = chunk
        
        if current:
            merged.append(current)
        
        return merged
    
    def _force_split(self, text: str) -> List[str]:
        """强制按字符数分割"""
        chunks = []
        for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
            chunk = text[i:i + self.config.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


class SemanticSplitter(TextSplitter):
    """语义分割器"""
    
    def __init__(self, config: TextProcessorConfig, embedding_model=None):
        super().__init__(config)
        self.embedding_model = embedding_model
    
    def split_text(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """基于语义的文本分割"""
        if not self.config.use_semantic_split or self.embedding_model is None:
            return super().split_text(text, metadata)
        
        # 首先按句子分割
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return super().split_text(text, metadata)
        
        # 计算句子间的语义相似度
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            
            # 检查长度限制
            if current_length + len(sentence) > self.config.chunk_size:
                # 保存当前块
                chunk_text = "".join(current_chunk_sentences)
                chunks.append(chunk_text)
                current_chunk_sentences = [sentence]
                current_length = len(sentence)
            else:
                current_chunk_sentences.append(sentence)
                current_length += len(sentence)
        
        # 保存最后一个块
        if current_chunk_sentences:
            chunk_text = "".join(current_chunk_sentences)
            chunks.append(chunk_text)
        
        # 创建TextChunk对象
        text_chunks = []
        current_index = 0
        
        for chunk_content in chunks:
            start_idx = text.find(chunk_content, current_index)
            if start_idx == -1:
                start_idx = current_index
            end_idx = start_idx + len(chunk_content)
            
            chunk = TextChunk(
                id=str(uuid.uuid4()),
                content=chunk_content.strip(),
                metadata=metadata.copy() if metadata else {},
                start_index=start_idx,
                end_index=end_idx
            )
            text_chunks.append(chunk)
            current_index = end_idx
        
        return text_chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 中文和英文句子分割
        pattern = r'(?<=[。！？.!?])\s*'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]


class QAExtractor:
    """问答对提取器"""
    
    def __init__(self):
        self.qa_patterns = [
            r'问[：:]\s*(.+?)\s*答[：:]\s*(.+?)(?=问[：:]|$)',
            r'Q[：:]\s*(.+?)\s*A[：:]\s*(.+?)(?=Q[：:]|$)',
            r'问题[：:]\s*(.+?)\s*答案[：:]\s*(.+?)(?=问题[：:]|$)',
        ]
    
    def extract_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        """从文本中提取问答对"""
        qa_pairs = []
        
        for pattern in self.qa_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for q, a in matches:
                q = q.strip()
                a = a.strip()
                if q and a:
                    qa_pairs.append((q, a))
        
        return qa_pairs
    
    def extract_from_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """从文本块中提取问答对并创建新的块"""
        qa_chunks = []
        
        for chunk in chunks:
            qa_pairs = self.extract_qa_pairs(chunk.content)
            for q, a in qa_pairs:
                qa_content = f"问题：{q}\n答案：{a}"
                qa_chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    content=qa_content,
                    metadata={
                        **chunk.metadata,
                        "type": "qa_pair",
                        "question": q,
                        "answer": a
                    }
                )
                qa_chunks.append(qa_chunk)
        
        return qa_chunks


class TextProcessor:
    """文本处理器"""
    
    def __init__(self, config: TextProcessorConfig, embedding_model=None):
        self.config = config
        if config.use_semantic_split and embedding_model:
            self.splitter = SemanticSplitter(config, embedding_model)
        else:
            self.splitter = TextSplitter(config)
        self.qa_extractor = QAExtractor()
    
    def process_documents(
        self,
        documents: List[Tuple[str, dict]],
        extract_qa: bool = False
    ) -> List[TextChunk]:
        """处理文档列表
        
        Args:
            documents: 文档列表，每个元素为 (content, metadata) 元组
            extract_qa: 是否提取问答对
        
        Returns:
            文本块列表
        """
        all_chunks = []
        
        for content, metadata in documents:
            # 分割文本
            chunks = self.splitter.split_text(content, metadata)
            all_chunks.extend(chunks)
            
            # 提取问答对
            if extract_qa:
                qa_chunks = self.qa_extractor.extract_from_chunks(chunks)
                all_chunks.extend(qa_chunks)
        
        return all_chunks
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符（保留中文、英文、数字、标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''、（）《》【】\-—…]', '', text)
        return text.strip()
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """截断文本"""
        max_length = max_length or self.config.chunk_size
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
