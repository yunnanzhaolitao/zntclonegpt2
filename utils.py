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
        # 使用新版LangChain推荐的方式
        from langchain.chains import create_history_aware_retriever
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        
        # 1. 创建历史感知的检索器
        retriever = vectordb.as_retriever()
        contextualize_q_chain = create_history_aware_retriever(
            llm, retriever, memory
        )
        
        # 2. 创建文档处理链
        qa_chain = create_stuff_documents_chain(llm)
        
        # 3. 组合成完整的检索链
        chain = create_retrieval_chain(contextualize_q_chain, qa_chain)
        
        # 调用链
        result = chain.invoke({"input": prompt})
        
        # 更新streamlit会话状态
        from streamlit import session_state as st_session
        chat_history = st_session.get("chat_history", [])
        chat_history.append((prompt, result["answer"]))
        st_session["chat_history"] = chat_history
        
        return result["answer"]

    # 普通对话
    conv = ConversationChain(llm=llm, memory=memory)
    return conv.invoke({"input": prompt})["response"]