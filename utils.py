# utils.py  — 包含了搜索代理 + 聊天接口
import os
from pathlib import Path
from typing import Optional, Union
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.tools import Tool
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
def online_search_agent(openai_key: str,
                       serp_key: Optional[str] = None,
                       model_name: str = "gpt-3.5-turbo",
                       api_base: str = "https://api.aigc369.com/v1"):
    """基于 SerpAPI 的联网搜索 Agent"""
    if not serp_key:
        serp_key = os.getenv("SERPAPI_API_KEY")
        if not serp_key:
            raise ValueError("需要提供 SERPAPI_API_KEY")
    os.environ["SERPAPI_API_KEY"] = serp_key

    search = SerpAPIWrapper()
    tool = Tool(
        name="Search",
        func=search.run,
        description="用于检索当前事件信息"
    )

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=openai_key,
        openai_api_base=api_base
    )
    return initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

def get_chat_response(prompt: str,
                      memory,
                      openai_api_key: str,
                      model_name: str = "gpt-3.5-turbo",
                      use_docs: bool = False,
                      embed_dir: Optional[Union[str, Path]] = None,
                      chain_of_thought: bool = False,
                      api_base: str = "https://api.aigc369.com/v1"):

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        openai_api_key=openai_api_key,
        openai_api_base=api_base
    )

    if use_docs and embed_dir and Path(embed_dir).exists():
        vectordb = Chroma(
            persist_directory=str(embed_dir),
            embedding_function=HuggingFaceEmbeddings(
                model_name="./models/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
        )
        # 使用 ConversationalRetrievalChain 替代旧的方法
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(),
            memory=memory,
            return_source_documents=True,
            verbose=chain_of_thought
        )
        
        # 从memory中获取chat_history
        chat_history = []
        if hasattr(memory, 'chat_memory'):
            chat_history = memory.chat_memory.messages
        
        # 调用链
        result = chain.invoke({
            "question": prompt,
            "chat_history": chat_history
        })
        
        # 更新streamlit会话状态
        import streamlit as st
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append((prompt, result["answer"]))
        
        return result["answer"]

    # 普通对话
    conv = ConversationChain(llm=llm, memory=memory)
    return conv.invoke({"input": prompt})["response"]