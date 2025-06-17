import os
import streamlit as st
from pathlib import Path
from typing import Optional, Union, Dict, Any
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.tools import Tool
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from memory import MemoryManager


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

    def search_wrapper(query: str) -> str:
        result = search.run(query)
        if isinstance(result, list):
            return "\n".join(result)
        return str(result)

    tool = Tool(
        name="Search",
        func=search_wrapper,
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


def get_chat_response(
    prompt: str,
    memory_manager: Optional["MemoryManager"] = None,
    openai_api_key: str | None = None,
    model_name: str = "gpt-3.5-turbo",
    use_docs: bool = False,
    embed_dir: Optional[Union[str, Path]] = None,
    chain_of_thought: bool = False,
    api_base: str = "https://api.aigc369.com/v1",
) -> str:
    """
    与大语言模型聊天，统一读写短期 / 长期记忆。
    统一调用向量库召回并写入向量库。
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        openai_api_key=openai_api_key,
        openai_api_base=api_base,
    )

    # 初始化 memory manager
    if memory_manager is None:
        if "memory_manager" in st.session_state:
            memory_manager = st.session_state.memory_manager  # type: ignore
        else:
            memory_manager = MemoryManager(llm)
            st.session_state.memory_manager = memory_manager

    answer: str

    # 文档检索模式
    if use_docs and embed_dir and Path(embed_dir).exists():
        vectordb = Chroma(
            persist_directory=str(embed_dir),
            embedding_function=HuggingFaceEmbeddings(
                model_name="./models/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            ),
        )

        relevant = memory_manager.get_relevant_history(prompt)
        short_term_history = relevant["short_term"].get("chat_history", [])

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(),
            verbose=chain_of_thought,
            output_key="answer",
        )

        try:
            result = chain.invoke({"question": prompt, "chat_history": short_term_history})
            answer = (
                result.get("answer")
                if isinstance(result, dict)
                else str(result)
            )
        except Exception as e:
            answer = f"⚠️ 链调用异常：{e}"

    # 普通对话模式
    else:
        try:
            from langchain.memory import ConversationBufferMemory
            relevant = memory_manager.get_relevant_history(prompt)
            messages = relevant["short_term"].get("chat_history", [])
            memory = ConversationBufferMemory(return_messages=True)
            for message in messages:
                if isinstance(message, HumanMessage):
                    memory.chat_memory.add_user_message(message.content)
                elif isinstance(message, AIMessage):
                    memory.chat_memory.add_ai_message(message.content)

            conv = ConversationChain(
                llm=llm,
                memory=memory,
                verbose=chain_of_thought,
            )
            result = conv.invoke({"input": prompt})
            answer = (
                result.get("response")
                if isinstance(result, dict)
                else str(result)
            )
        except Exception as e:
            answer = f"⚠️ 对话模式异常：{e}"

    # 写入所有历史（短期 + 向量库）
    memory_manager.add_interaction(prompt, answer)
    memory_manager.add_to_vector_memory(prompt, answer)

    st.session_state.setdefault("chat_history", [])               # type: ignore
    st.session_state.chat_history.append((prompt, answer))        # type: ignore

    return answer
