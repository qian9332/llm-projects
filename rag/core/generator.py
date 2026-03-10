"""
生成模块
实现多种生成策略：基础生成、带引用生成等
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio

from core.retriever import RetrievalResult
from config.settings import LLMConfig


@dataclass
class GenerationResult:
    """生成结果"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None


class BaseGenerator(ABC):
    """生成器基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """生成回答"""
        pass
    
    @abstractmethod
    async def generate_async(self, query: str, context: str) -> str:
        """异步生成回答"""
        pass


class OpenAIGenerator(BaseGenerator):
    """OpenAI生成器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base
            )
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
    
    def generate(self, query: str, context: str) -> str:
        """生成回答"""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p
        )
        
        return response.choices[0].message.content
    
    async def generate_async(self, query: str, context: str) -> str:
        """异步生成回答"""
        # OpenAI的异步客户端
        import openai
        async_client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base
        )
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)
        
        response = await async_client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p
        )
        
        return response.choices[0].message.content
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个专业的问答助手。请根据提供的上下文信息回答用户问题。
要求：
1. 回答要准确、简洁、有条理
2. 如果上下文中没有相关信息，请明确告知用户
3. 回答时引用相关的上下文内容
4. 使用中文回答"""
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """构建用户提示"""
        return f"""上下文信息：
{context}

用户问题：{query}

请根据上下文信息回答问题："""


class LocalGenerator(BaseGenerator):
    """本地模型生成器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._init_model()
    
    def _init_model(self):
        """初始化本地模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_path = self.config.local_model_path or "Qwen/Qwen2-7B-Instruct"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        except ImportError:
            raise ImportError("请安装transformers库: pip install transformers torch")
    
    def generate(self, query: str, context: str) -> str:
        """生成回答"""
        prompt = self._build_prompt(query, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取回答部分
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        
        return response
    
    async def generate_async(self, query: str, context: str) -> str:
        """异步生成回答"""
        return self.generate(query, context)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """构建提示"""
        return f"""请根据以下上下文信息回答问题。

上下文：
{context}

问题：{query}

回答："""


class ZAIGenerator(BaseGenerator):
    """使用z-ai-web-dev-sdk的生成器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._init_client()
    
    def _init_client(self):
        """初始化ZAI客户端"""
        self.zai = None  # 延迟初始化
    
    async def _get_client(self):
        """获取ZAI客户端"""
        if self.zai is None:
            import asyncio
            # 动态导入以避免问题
            import importlib
            zai_module = importlib.import_module('z-ai-web-dev-sdk')
            ZAI = getattr(zai_module, 'ZAI')
            self.zai = await ZAI.create()
        return self.zai
    
    def generate(self, query: str, context: str) -> str:
        """生成回答"""
        return asyncio.run(self.generate_async(query, context))
    
    async def generate_async(self, query: str, context: str) -> str:
        """异步生成回答"""
        zai = await self._get_client()
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)
        
        completion = await zai.chat.completions.create({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        })
        
        return completion.choices[0].message.content
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个专业的问答助手。请根据提供的上下文信息回答用户问题。
要求：
1. 回答要准确、简洁、有条理
2. 如果上下文中没有相关信息，请明确告知用户
3. 回答时引用相关的上下文内容
4. 使用中文回答"""
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """构建用户提示"""
        return f"""上下文信息：
{context}

用户问题：{query}

请根据上下文信息回答问题："""


class RAGGenerator:
    """RAG生成器（整合检索和生成）"""
    
    def __init__(self, generator: BaseGenerator):
        self.generator = generator
    
    def generate_with_sources(
        self,
        query: str,
        retrieval_result: RetrievalResult
    ) -> GenerationResult:
        """生成带来源的回答"""
        # 构建上下文
        context = self._build_context(retrieval_result)
        
        # 生成回答
        answer = self.generator.generate(query, context)
        
        # 提取来源
        sources = [
            {
                "content": result.document.content,
                "score": result.score,
                "rank": result.rank,
                "metadata": result.document.metadata
            }
            for result in retrieval_result.results
        ]
        
        return GenerationResult(
            query=query,
            answer=answer,
            sources=sources,
            metadata={
                "total_sources": len(sources),
                "context_length": len(context)
            }
        )
    
    async def generate_with_sources_async(
        self,
        query: str,
        retrieval_result: RetrievalResult
    ) -> GenerationResult:
        """异步生成带来源的回答"""
        context = self._build_context(retrieval_result)
        answer = await self.generator.generate_async(query, context)
        
        sources = [
            {
                "content": result.document.content,
                "score": result.score,
                "rank": result.rank,
                "metadata": result.document.metadata
            }
            for result in retrieval_result.results
        ]
        
        return GenerationResult(
            query=query,
            answer=answer,
            sources=sources,
            metadata={
                "total_sources": len(sources),
                "context_length": len(context)
            }
        )
    
    def _build_context(self, retrieval_result: RetrievalResult) -> str:
        """构建上下文"""
        context_parts = []
        for i, result in enumerate(retrieval_result.results, 1):
            context_parts.append(f"[文档{i}]\n{result.document.content}")
        
        return "\n\n".join(context_parts)


def create_generator(config: LLMConfig) -> BaseGenerator:
    """工厂函数：创建生成器"""
    if config.use_local and config.local_model_path:
        return LocalGenerator(config)
    elif config.model_name.startswith("gpt"):
        return OpenAIGenerator(config)
    else:
        return OpenAIGenerator(config)
