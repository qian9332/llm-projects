# RAG项目 - 检索增强生成系统

基于大型语言模型的检索增强生成（RAG）系统实现，支持多种向量数据库、嵌入模型和RAG引擎。

## 项目简介

本项目实现了一个完整的RAG系统，包含以下核心功能：

- **文档处理**：文本分割、语义分割、问答对提取
- **向量存储**：支持Chroma、FAISS、Milvus等多种向量数据库
- **嵌入模型**：支持OpenAI、本地模型等多种嵌入模型
- **检索策略**：向量检索、混合检索、重排序检索
- **RAG引擎**：ContextChatEngine、CondensePlusContextMode、Hybrid引擎
- **多轮对话**：支持上下文感知的多轮对话

## 项目结构

```
rag-project/
├── config/
│   └── settings.py          # 配置文件
├── core/
│   ├── embeddings.py        # 嵌入模型
│   ├── vector_store.py      # 向量数据库
│   ├── retriever.py         # 检索器
│   ├── generator.py         # 生成器
│   └── rag_system.py        # RAG系统主类
├── engines/
│   └── rag_engines.py       # RAG引擎实现
├── utils/
│   └── text_processor.py    # 文本处理工具
├── examples/
│   └── demo.py              # 示例代码
├── data/                    # 数据目录
│   └── vector_store/        # 向量存储目录
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install openai chromadb faiss-cpu sentence-transformers numpy
```

### 基本使用

```python
from core.rag_system import create_rag_system

# 创建RAG系统
rag = create_rag_system(
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo",
    vector_store="chroma",
    engine_type="context"
)

# 索引文档
documents = [
    ("RAG是一种结合检索和生成的技术...", {"topic": "RAG介绍"}),
    ("向量数据库用于存储文档向量...", {"topic": "向量数据库"}),
]
rag.index_documents(documents)

# 查询
result = rag.query("什么是RAG？")
print(result['answer'])
```

### 多轮对话

```python
# 创建会话
session_id = rag.create_session()

# 多轮对话
rag.query("Python是什么？", session_id)
rag.query("它有哪些应用领域？", session_id)  # 会理解"它"指Python

# 获取会话历史
history = rag.get_session_history(session_id)
```

## 核心组件

### 1. 嵌入模型 (Embeddings)

支持多种嵌入模型：

- **OpenAI**: `text-embedding-ada-002`, `text-embedding-3-small`
- **本地模型**: `all-MiniLM-L6-v2`, `bge-large-zh` 等

```python
from core.embeddings import create_embedding
from config.settings import EmbeddingConfig

config = EmbeddingConfig(model_name="all-MiniLM-L6-v2", use_local=True)
embedding = create_embedding(config)
```

### 2. 向量数据库 (Vector Store)

支持多种向量数据库：

- **Chroma**: 轻量级，适合开发测试
- **FAISS**: 高性能，适合大规模数据
- **Simple**: 内存存储，适合小规模应用

```python
from core.vector_store import create_vector_store
from config.settings import VectorStoreConfig

config = VectorStoreConfig(store_type="chroma", top_k=5)
store = create_vector_store(config)
```

### 3. 检索器 (Retriever)

多种检索策略：

- **VectorRetriever**: 纯向量检索
- **HybridRetriever**: 向量+关键词混合检索
- **RerankingRetriever**: 带重排序的检索

### 4. RAG引擎

三种RAG引擎实现：

#### ContextChatEngine
- 检索相关上下文，与历史一起发送给LLM
- 适合简单问答场景

#### CondensePlusContextEngine
- 将历史和当前问题压缩成独立查询
- 提高多轮对话的检索准确性

#### HybridRAGEngine
- 根据对话历史自动选择引擎
- 平衡简单性和准确性

## 配置说明

```python
from config.settings import RAGConfig

config = RAGConfig(
    embedding=EmbeddingConfig(
        model_name="text-embedding-ada-002",
        dimension=1536
    ),
    vector_store=VectorStoreConfig(
        store_type="chroma",
        top_k=5,
        similarity_threshold=0.7
    ),
    llm=LLMConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2048
    ),
    engine_type="context",
    max_history_turns=10
)
```

## 高级用法

### 使用Builder模式

```python
from core.rag_system import RAGSystemBuilder

rag = (RAGSystemBuilder()
       .with_embedding_model("all-MiniLM-L6-v2", use_local=True)
       .with_llm("gpt-4")
       .with_vector_store("faiss", persist_dir="./data/vectors")
       .with_engine("hybrid")
       .with_chunk_size(512, overlap=50)
       .with_top_k(10)
       .build())
```

### 从文件索引

```python
rag.index_from_files([
    "./docs/document1.txt",
    "./docs/document2.txt"
], metadata={"source": "local_files"})
```

### 提取问答对

```python
documents = [
    ("问：什么是RAG？答：RAG是检索增强生成技术...", {}),
]
rag.index_documents(documents, extract_qa=True)
```

## 性能优化建议

1. **向量数据库选择**
   - 小规模数据（<10万）：Chroma或Simple
   - 中等规模（10-100万）：FAISS
   - 大规模（>100万）：Milvus

2. **嵌入模型选择**
   - 追求质量：OpenAI text-embedding-3-large
   - 追求速度：all-MiniLM-L6-v2
   - 中文场景：bge-large-zh

3. **检索参数调优**
   - top_k: 通常5-10即可
   - similarity_threshold: 0.6-0.8之间
   - 使用重排序提升精度

4. **文本分割**
   - chunk_size: 256-1024，根据文档类型调整
   - overlap: chunk_size的10-20%

## 项目来源

本项目基于以下RAG项目经验总结实现：

1. **基于大型语言模型的本地问答系统（LLM-LocalQA system）**
   - 问题提炼、知识融合、推理求解
   - 文本转换与向量化、向量搜索

2. **基于RAG的电商智能问答系统**
   - 数据层优化、模型层调整、训练层创新
   - Qwen7b基座模型应用

3. **RAG-ChatEngine聊天系统**
   - ContextChatEngine实现
   - CondensePlusContextMode实现
   - 上下文压缩技术

4. **智能客服问答系统 LLM-RAG**
   - 知识库构建与管理
   - 向量检索与重排

5. **审计知识库问答系统**
   - 大模型应用、向量数据库
   - 多端支持

6. **智能医疗问答系统**
   - 语音识别集成
   - 知识图谱构建

## 技术栈

- **语言**: Python 3.8+
- **嵌入模型**: OpenAI Embeddings, Sentence Transformers
- **向量数据库**: Chroma, FAISS, Milvus
- **LLM**: OpenAI GPT, 本地模型
- **NLP**: NLTK, spaCy, Transformers

## 许可证

MIT License
