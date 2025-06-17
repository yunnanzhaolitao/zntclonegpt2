from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
import os
import json
from datetime import datetime

class ShortTermMemory:
    def __init__(self, llm, max_token_limit=3000):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.message_history = ChatMessageHistory()
        self.memory_key = "chat_history"

    def add_message(self, human_message: str, ai_message: str):
        self.message_history.add_user_message(human_message)
        self.message_history.add_ai_message(ai_message)
        
        # 简单的令牌管理 - 如果历史记录太长，删除最早的消息
        self._trim_history_if_needed()

    def _trim_history_if_needed(self):
        # 简单的令牌计数 - 每个字符算作一个令牌
        total_tokens = sum(len(msg.content) for msg in self.message_history.messages)
        
        # 如果超过最大令牌限制，删除最早的消息对
        while total_tokens > self.max_token_limit and len(self.message_history.messages) >= 2:
            # 删除最早的人类消息和AI消息
            self.message_history.messages.pop(0)  # 人类消息
            if self.message_history.messages:  # 确保还有消息
                self.message_history.messages.pop(0)  # AI消息
            
            # 重新计算令牌数
            total_tokens = sum(len(msg.content) for msg in self.message_history.messages)

    def get_memory_variables(self) -> Dict[str, Any]:
        return {self.memory_key: self.message_history.messages}

    def get_buffer(self) -> List[Dict[str, str]]:
        buffer = []
        for i in range(0, len(self.message_history.messages), 2):
            if i + 1 < len(self.message_history.messages):
                human_msg = self.message_history.messages[i]
                ai_msg = self.message_history.messages[i + 1]
                buffer.append({
                    "human": human_msg.content,
                    "ai": ai_msg.content
                })
        return buffer

    def clear(self):
        self.message_history.clear()

class LongTermMemory:
    def __init__(self, vector_store_path: str = "vector_dbs", openai_api_key: str = None, openai_api_base: str = None):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base
        )
        self.vector_store_path = vector_store_path
        self.vector_store = self._load_or_create_vector_store()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.plain_history: List[Dict[str, str]] = []

    def _load_or_create_vector_store(self) -> FAISS:
        if os.path.exists(self.vector_store_path):
            return FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return FAISS.from_texts(["Initial memory"], self.embeddings)

    def _process_conversation(self, human_message: str, ai_message: str) -> str:
        timestamp = datetime.now().isoformat()
        return f"Time: {timestamp}\nHuman: {human_message}\nAI: {ai_message}\n---\n"

    def add_conversation(self, human_message: str, ai_message: str):
        processed_text = self._process_conversation(human_message, ai_message)
        docs = self.text_splitter.create_documents([processed_text])
        self.vector_store.add_documents(docs)
        self.vector_store.save_local(self.vector_store_path)

        # 保存为原始摘要
        self.plain_history.append({"human": human_message, "ai": ai_message})

    def search_memory(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def summarize(self) -> str:
        if not self.plain_history:
            return "🕳️ 当前没有长期记忆。"
        return "\n\n".join([
            f"{i+1}. 👤 {pair['human']}\n   🤖 {pair['ai']}"
            for i, pair in enumerate(self.plain_history)
        ])

class MemoryManager:
    def __init__(self, llm, vector_store_path: str = "vector_dbs"):
        openai_api_key = None
        if hasattr(llm, 'openai_api_key'):
            if hasattr(llm.openai_api_key, 'get_secret_value'):
                openai_api_key = llm.openai_api_key.get_secret_value()
            else:
                openai_api_key = llm.openai_api_key

        openai_api_base = None
        if hasattr(llm, 'openai_api_base'):
            if hasattr(llm.openai_api_base, 'get_secret_value'):
                openai_api_base = llm.openai_api_base.get_secret_value()
            else:
                openai_api_base = llm.openai_api_base

        self.short_term = ShortTermMemory(llm)
        self.long_term = LongTermMemory(
            vector_store_path=vector_store_path,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base
        )

    def add_interaction(self, human_message: str, ai_message: str):
        self.short_term.add_message(human_message, ai_message)
        self.long_term.add_conversation(human_message, ai_message)

    def get_relevant_history(self, query: str, k: int = 5) -> Dict[str, Any]:
        short_term_memory = self.short_term.get_memory_variables()
        long_term_memory = self.long_term.search_memory(query, k)
        return {
            "short_term": short_term_memory,
            "long_term": long_term_memory
        }

    def clear_short_term(self):
        self.short_term.clear()

    def get_long_term_summary(self) -> str:
        return self.long_term.summarize()