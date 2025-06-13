# main.py  — znt-GPT 主入口
import pathlib
import asyncio
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import nest_asyncio
nest_asyncio.apply()

from utils import get_chat_response, online_search_agent
from ingest import ingest_folder

# 页面配置
st.set_page_config(page_title="znt-GPT", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
      .stChatMessage { font-size:20px !important; }
      textarea, input { font-size:18px !important; }
    </style>
""", unsafe_allow_html=True)

# 白名单 & 向量库根目录
WHITE_LIST = {"赵黎涛", "吴雨阳", "华清钟", "大家的爸爸88", "赵德胜", "吴汉易"}
BASE_DB_DIR = pathlib.Path("vector_dbs")
BASE_DB_DIR.mkdir(exist_ok=True)

# Session state 初始化
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("user", None)
st.session_state.setdefault("messages", [{"role": "ai", "content": "您好！请在下方输入问题。"}])
st.session_state.setdefault("uploaded_done", False)

# 登录逻辑
if not st.session_state["authenticated"]:
    st.title("🤖 znt-GPT 登录")
    name = st.text_input("请输入用户名（白名单）", key="login_name")
    if st.button("进入", key="login_btn"):
        if name in WHITE_LIST:
            st.session_state.update({"authenticated": True, "user": name})
            st.rerun()
        else:
            st.error("抱歉！不在白名单。")
    st.stop()

# 用户目录初始化
user = st.session_state["user"]
user_db_dir = BASE_DB_DIR / user
user_db_dir.mkdir(parents=True, exist_ok=True)
st.title(f"🤖 你好，{user}！")

# 侧边栏：API 设置
with st.sidebar:
    st.header("🔧 设置")
    api_key = st.text_input("OpenAI API Key", type="password")
    serp_key = st.text_input("SerpAPI Key（联网搜索）", type="password")
    model = st.selectbox("模型", ["gpt-3.5-turbo", "gpt-4", "o3", "o4-mini", "gpt-4.1", "gpt-4.5-preview-2025-02-27"])
    api_base = st.text_input("API Base URL", value="https://api.aigc369.com/v1")

    if st.button("✅ 保存配置", key="setup"):
        if not api_key:
            st.warning("请填写 OpenAI Key")
            st.stop()
        st.session_state.update({
            "openai_api_key": api_key,
            "serp_api_key": serp_key,
            "model_name": model,
            "api_base": api_base
        })

        # 初始化 summary memory（❗注意去掉 input_key/output_key）
        summary_llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_base
        )
        st.session_state["memory"] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )

        st.success("配置完成！")
        st.rerun()

# 等待配置完成
if "openai_api_key" not in st.session_state or "memory" not in st.session_state:
    st.info("请在侧边栏完成配置后再使用")
    st.stop()

# 可选调试：查看历史摘要内容
with st.sidebar:
    if st.button("📄 查看历史摘要"):
        st.sidebar.code(st.session_state["memory"].buffer)

# 展示聊天记录
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 文件上传区域
with st.container():
    uploaded = st.file_uploader(
        "📥 拖文档/图片到此处", type=["pdf", "docx", "pptx", "png", "jpg"],
        accept_multiple_files=True, key="uploader"
    )
    if uploaded and not st.session_state["uploaded_done"]:
        with st.spinner("解析并写入向量库…"):
            try:
                upload_dir = user_db_dir / "uploads"
                upload_dir.mkdir(exist_ok=True)
                for f in uploaded:
                    with open(upload_dir / f.name, "wb") as w:
                        w.write(f.getbuffer())
                ingest_folder(str(upload_dir), persist_dir=str(user_db_dir))
                st.success(f"已写入 {len(uploaded)} 个文件！")
                st.session_state["uploaded_done"] = True
                st.rerun()
            except Exception as e:
                st.error(f"文件处理出错：{e}")

# 聊天输入与开关
col1, col2 = st.columns(2)
online = col1.checkbox("🌐 联网搜索", key="online")
cot = col2.checkbox("🧠 深度思考", key="cot")
prompt = st.chat_input("请输入内容（回车发送）")

# 响应逻辑
if prompt:
    st.session_state["messages"].append({"role": "human", "content": prompt})
    st.chat_message("human").write(prompt)
    with st.spinner("AI思考中…"):
        try:
            if online:
                if not st.session_state.get("serp_api_key"):
                    raise ValueError("请在侧边栏填写 SerpAPI Key")
                agent = online_search_agent(
                    openai_key=st.session_state["openai_api_key"],
                    serp_key=st.session_state["serp_api_key"],
                    model_name=st.session_state["model_name"],
                    api_base=st.session_state["api_base"]
                )
                answer = agent.invoke({"input": prompt})["output"]
            else:
                answer = get_chat_response(
                    prompt=prompt,
                    memory=st.session_state["memory"],
                    openai_api_key=st.session_state["openai_api_key"],
                    model_name=st.session_state["model_name"],
                    use_docs=True,
                    embed_dir=str(user_db_dir),
                    chain_of_thought=cot,
                    api_base=st.session_state["api_base"]
                )
        except Exception as ex:
            answer = f"出错：{ex}"
    st.session_state["messages"].append({"role": "ai", "content": answer})
    st.chat_message("ai").write(answer)