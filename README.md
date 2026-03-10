# 🤖 llm-projects

大模型(LLM)相关项目集合 - Large Language Model Projects Collection

## 📖 简介

本仓库专门用于存放和管理大语言模型(Large Language Model)相关的项目、实验和研究成果。

## 📁 项目结构

```
llm-projects/
├── rag/                  # RAG检索增强生成系统 ✅ 已实现
├── llm-basics/           # 大模型基础知识与入门项目
├── prompt-engineering/   # 提示工程相关项目
├── fine-tuning/          # 模型微调相关项目
├── agents/               # AI Agent相关项目
├── deployment/           # 模型部署相关项目
└── experiments/          # 实验性项目
```

## 🚀 已实现项目

### [RAG检索增强生成系统](./rag/)

完整的RAG系统实现，包含：

- **文档处理**：文本分割、语义分割、问答对提取
- **向量存储**：支持Chroma、FAISS、Milvus等多种向量数据库
- **嵌入模型**：支持OpenA)、本地模型等多种嵌入模型
- **检索策略**：向量检索、混合检索、重排序检索
- **RAG引擎**：ContextChatEngine、CondensePlusContextMode、Hybrid引擎
- **多轮对话**：支持上下文感知的多轮对话

[查看详细文档](./rag/README.md)

## 🎯 项目方向

- **基础研究**: Transformer架构、注意力机制、模型原理
- **提示工程**: Prompt设计、Few-shot Learning、Chain-of-Thought
- **模型微调**: LoRA、QLoRA、全量微调等技术
- **RAG系统**: 向量数据库、检索增强、知识库构建
- **AI Agent**: 工具调用、多Agent协作、自主决策
- **模型部署**: 量化推理、API服务、性能优化

## 🛠️ 技术栈

- **框架**: PyTorch, TensorFlow, Transformers, LangChain
- **模型**: GPT, LLaMA, ChatGLM, Qwen, Baichuan
- **工具**: vLLM, Ollama, llama.cpp, Hugging Face

## 📝 使用说明

每个子项目都包含独立的README文档，详细介绍项目背景、实现方法和使用方式。

## 📅 更新日志

- 2026-03-10: 添加RAG检索增强生成系统实现
- 2026-03-10: 创建仓库，初始化项目结构

---

⭐ 如果这个仓库对你有帮助，欢迎Star支持！
