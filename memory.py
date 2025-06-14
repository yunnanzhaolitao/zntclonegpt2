from typing import List, Dict, Any
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
import json
from datetime import datetime

class ShortTermMemory:
    def __init__(self, llm, max_token_limit=3000):
        def fake_token_counter(_text: str) -> int:
            return len(_text)

        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            max_token_limit=max_token_limit,
            get_token_func=fake_token_counter
        )

    def add_message(self, human_message: str, ai_message: str):
        self.memory.save_context(
            {"input": human_message},
            {"output": ai_message}
        )

    def get_memory_variables(self) -> Dict[str, Any]:
        return self.memory.load_memory_variables({})

    def get_buffer(self) -> List[str]:
        return self.memory.buffer

    def clear(self):
        self.memory.clear()

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
