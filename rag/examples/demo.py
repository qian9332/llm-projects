"""
RAG系统示例代码
演示如何使用RAG系统进行文档索引和问答
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_system import RAGSystem, RAGSystemBuilder, create_rag_system
from config.settings import RAGConfig, EmbeddingConfig, VectorStoreConfig, LLMConfig


def example_basic_usage():
    """基本使用示例"""
    print("=" * 50)
    print("基本使用示例")
    print("=" * 50)
    
    # 创建RAG系统
    rag = create_rag_system(
        embedding_model="all-MiniLM-L6-v2",  # 使用本地模型
        llm_model="gpt-3.5-turbo",
        vector_store="simple",  # 使用简单的内存存储
        engine_type="context"
    )
    
    # 索引文档
    documents = [
        ("RAG（检索增强生成）是一种结合检索和生成的技术。它通过检索相关文档来增强大语言模型的生成能力。", 
         {"topic": "RAG介绍"}),
        ("向量数据库是RAG系统的核心组件，用于存储文档的向量表示并支持相似度搜索。常见的向量数据库包括Chroma、FAISS、Milvus等。",
         {"topic": "向量数据库"}),
        ("嵌入模型将文本转换为向量表示，是连接文本和向量数据库的桥梁。常用的嵌入模型有OpenAI的text-embedding-ada-002和本地的sentence-transformers系列。",
         {"topic": "嵌入模型"}),
        ("RAG系统的优势包括：1. 能够利用外部知识库；2. 减少模型幻觉；3. 支持知识更新；4. 提供可追溯的答案来源。",
         {"topic": "RAG优势"}),
    ]
    
    print("\n正在索引文档...")
    count = rag.index_documents(documents)
    print(f"成功索引 {count} 个文档块")
    
    # 查询
    questions = [
        "什么是RAG？",
        "RAG系统有哪些优势？",
        "向量数据库有什么作用？"
    ]
    
    print("\n开始问答：")
    for question in questions:
        print(f"\n问题：{question}")
        result = rag.query(question)
        print(f"回答：result['answer'][:200]}...")
        print(f"来源数量：{len(result['sources'])}")


def example_multi_turn_conversation():
    """多轮对话示例"""
    print("\n" + "=" * 50)
    print("多轮对话示例")
    print("=" * 50)
    
    # 创建RAG系统
    rag = (RAGSystemBuilder()
           .with_embedding_model("all-MiniLM-L6-v2", use_local=True)
           .with_llm("gpt-3.5-turbo")
           .with_vector_store("simple")
           .with_engine("condense")  # 使用压缩引擎支持多轮对话
           .build())
    
    # 索引文档
    documents = [
        ("Python是一种高级编程语言，由Guido van Rossum于1991年创建。Python以其简洁的语法和强大的库生态系统而闻名。",
         {"topic": "Python介绍"}),
        ("Python广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。流行的Python框架包括Django、Flask、FastAPI等。",
         {"topic": "Python应用"}),
        ("Python 3是Python的最新主要版本，引入了许多改进，包括更好的Unicode支持、异步编程等特性。",
         {"topic": "Python版本"}),
    ]
    
    rag.index_documents(documents)
    
    # 创建会话
    session_id = rag.create_session()
    print(f"创建会话：{session_id}")
    
    # 多轮对话
    conversation = [
        "Python是什么时候创建的？",
        "它主要用在哪些领域？",
        "Python 3有什么新特性？"
    ]
    
    for question in conversation:
        print(f"\n用户：{question}")
        result = rag.query(question, session_id)
        print(f"助手：{result['answer'][:150]}...")
    
    # 查看会话历史
    print("\n会话历史：")
    history = rag.get_session_history(session_id)
    for msg in history:
        role = "用户" if msg['role'] == 'user' else "助手"
        print(f"  {role}：{msg['content'][:50]}...")


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 50)
    print("自定义配置示例")
    print("=" * 50)
    
    # 创建自定义配置
    config = RAGConfig(
        embedding=EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            use_local=True,
            dimension=384
        ),
        vector_store=VectorStoreConfig(
            store_type="simple",
            top_k=3,
            similarity_threshold=0.5
        ),
        llm=LLMConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1024
        ),
        engine_type="context",
        max_history_turns=5
    )
    
    # 使用自定义配置创建系统
    rag = RAGSystem(config)
    
    print("配置信息：")
    print(f"  嵌入模型：{config.embedding.model_name}")
    print(f"  向量数据库：{config.vector_store.store_type}")
    print(f"  LLM模型：{config.llm.model_name}")
    print(f"  引擎类型：{config.engine_type}")
    print(f"  检索数量：{config.vector_store.top_k}")
    
    # 获取系统统计
    stats = rag.get_stats()
    print(f"\n系统统计：")
    print(f"  文档数量：{stats['document_count']}")


def example_different_engines():
    """不同引擎对比示例"""
    print("\n" + "=" * 50)
    print("不同RAG引擎对比示例")
    print("=" * 50)
    
    documents = [
        ("机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出决策或预测。",
         {"topic": "机器学习"}),
        ("深度学习是机器学习的一个子领域，使用神经网络来学习数据的表示。",
         {"topic": "深度学习"}),
        ("自然语言处理（NLP）是AI的一个重要领域，专注于计算机与人类语言之间的交互。",
         {"topic": "NLP"}),
    ]
    
    engines = ["context", "condense", "hybrid"]
    
    for engine_type in engines:
        print(f"\n--- {engine_type} 引擎 ---")
        
        rag = create_rag_system(
            embedding_model="all-MiniLM-L6-v2",
            vector_store="simple",
            engine_type=engine_type
        )
        
        rag.index_documents(documents)
        
        result = rag.query("什么是深度学习？")
        print(f"回答：{result['answer'][:100]}...")
        print(f"引擎：{result['metadata'].get('engine', 'unknown')}")


def main():
    """主函数"""
    print("RAG系统示例演示")
    print("=" * 50)
    
    try:
        # 基本使用
        example_basic_usage()
        
        # 多轮对话
        example_multi_turn_conversation()
        
        # 自定义配置
        example_custom_config()
        
        # 不同引擎对比
        example_different_engines()
        
    except Exception as e:
        print(f"\n错误：{e}")
        print("注意：某些示例需要配置API密钥才能完整运行")


if __name__ == "__main__":
    main()
