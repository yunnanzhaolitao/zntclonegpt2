import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.tools import Tool
from langchain.chains import ConversationChain, ConversationalRetrievalChain
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
        # 保证输出为纯文本格式
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


def get_chat_response(prompt: str,
                      memory_manager: Optional[MemoryManager] = None,
                      openai_api_key: str = None,
                      model_name: str = "gpt-3.5-turbo",
                      use_docs: bool = False,
                      embed_dir: Optional[Union[str, Path]] = None,
                      chain_of_thought: bool = False,
                      api_base: str = "https://api.aigc369.com/v1"):
    """
    获取聊天响应，支持短期和长期记忆，增强健壮性防止 answer 报错
    """
    import streamlit as st
    import json
    from langchain.chains import ConversationChain, ConversationalRetrievalChain
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    # 初始化 LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        openai_api_key=openai_api_key,
        openai_api_base=api_base
    )

    # 初始化记忆管理器
    if memory_manager is None:
        if "memory_manager" in st.session_state:
            memory_manager = st.session_state.memory_manager
        else:
            memory_manager = MemoryManager(llm)
            st.session_state.memory_manager = memory_manager

    # 如果启用文档检索
    if use_docs and embed_dir and Path(embed_dir).exists():
        vectordb = Chroma(
            persist_directory=str(embed_dir),
            embedding_function=HuggingFaceEmbeddings(
                model_name="./models/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
        )

        # 获取相关历史记录
        relevant_memory = memory_manager.get_relevant_history(prompt)
        messages = relevant_memory["short_term"].get("chat_history", [])

        # 创建检索链
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            verbose=chain_of_thought,
            output_key="answer"
        )

        try:
            # 将消息历史转换为元组列表格式
            chat_history = []
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i + 1]
                    chat_history.append((human_msg.content, ai_msg.content))

            # 调用检索链
            result = chain.invoke({
                "question": prompt,
                "chat_history": chat_history
            })

            # 输出调试信息
            st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")
            print("[DEBUG] Result type:", type(result))
            print("[DEBUG] Result content:", result)

            if isinstance(result, dict):
                answer = result.get("answer") or result.get("output") or "🤖 没有获取到回答内容。"
            else:
                answer = str(result)

        except Exception as e:
            answer = f"⚠️ 链调用异常：{str(e)}"

        memory_manager.add_interaction(prompt, answer)

        st.session_state.chat_history = st.session_state.get("chat_history", [])
        st.session_state.chat_history.append((prompt, answer))

        return answer

    # 否则为普通对话模式
    try:
        # 获取历史消息
        messages = memory_manager.short_term.message_history.messages
        
        # 构建对话上下文
        context = "\n".join([
            f"Human: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}"
            for msg in messages[-6:]  # 只使用最近的3轮对话
        ])
        
        # 使用简单的 ConversationChain，不依赖复杂的内存管理
        conv = ConversationChain(
            llm=llm,
            verbose=chain_of_thought
        )
        
        # 将上下文和当前问题组合
        if context:
            full_prompt = f"{context}\n\nHuman: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
            
        result = conv.invoke({"input": full_prompt})
        
        print("[DEBUG] ConversationChain result:", result)

        if isinstance(result, dict):
            response = result.get("response") or list(result.values())[0]
        else:
            response = str(result)

    except Exception as e:
        response = f"⚠️ 对话模式异常：{str(e)}"

    # 添加到内存
    memory_manager.add_interaction(prompt, response)

    st.session_state.chat_history = st.session_state.get("chat_history", [])
    st.session_state.chat_history.append((prompt, response))

    return response