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
    与大语言模型聊天，并自动读写短期 / 长期记忆。

    - 当 `use_docs=True` 且指定了 `embed_dir` 时，自动检索该目录下的向量库。
    - 返回值始终是「纯文本 / emoji」，不会再额外输出 JSON 或调试信息。
    """

    # ───────────────────── 1. 初始化 LLM ─────────────────────
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        openai_api_key=openai_api_key,
        openai_api_base=api_base,
    )

    # ───────────────────── 2. 初始化 Memory ──────────────────
    if memory_manager is None:
        if "memory_manager" in st.session_state:
            memory_manager = st.session_state.memory_manager    # type: ignore
        else:
            from memory import MemoryManager                    # 迟导入，避免循环
            memory_manager = MemoryManager(llm)
            st.session_state.memory_manager = memory_manager

    # ───────────────────── 3. 文档检索模式 ──────────────────
    if use_docs and embed_dir and Path(embed_dir).exists():
        # ① 构建向量数据库检索器
        vectordb = Chroma(
            persist_directory=str(embed_dir),
            embedding_function=HuggingFaceEmbeddings(
                model_name="./models/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            ),
        )

        # ② 拿到短期记忆中的相关对话
        relevant_memory = memory_manager.get_relevant_history(prompt)
        short_term_history = relevant_memory["short_term"].get("chat_history", [])

        # ③ 组装 Retrieval Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(),
            memory=memory_manager.short_term.memory,
            return_source_documents=False,      # 仅要答案，不返回文档
            verbose=chain_of_thought,
            output_key="answer",
        )

        # ④ 调用链
        try:
            result = chain.invoke(
                {"question": prompt, "chat_history": short_term_history}
            )
            answer = (
                result.get("answer")
                if isinstance(result, dict)
                else str(result)
            )
        except Exception as e:
            answer = f"⚠️ 链调用异常：{e}"

    # ───────────────────── 4. 普通对话模式 ───────────────────
    else:
        try:
            conv = ConversationChain(
                llm=llm,
                memory=memory_manager.short_term.memory,
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

    # ───────────────────── 5. 写入记忆 & SessionState ───────
    memory_manager.add_interaction(prompt, answer)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []                      # type: ignore
    st.session_state.chat_history.append((prompt, answer))

    return answer