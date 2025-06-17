# ---------- znt-GPT  main.py -- import & 环境修补 ----------
import os, sys, importlib, types, builtins
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"          # 关掉源码监控后台线程
builtins.__dict__["__torch_fake_module__"] = True      # 避免 torch 延迟加载时再报错

# ① 彻底禁用 Streamlit 对模块路径的探测
import streamlit.watcher.local_sources_watcher as _sw
_sw.LocalSourcesWatcher._get_module_paths = lambda *_a, **_k: []

# ② 修补 torch.*
try:
    torch = importlib.import_module("torch")

    def _patch(name: str):
        """为 torch.(classes|_classes) 打补丁：mock __getattr__ & __path__"""
        mod = getattr(torch, name, None)
        if not mod:
            return
        # (a) 让任何 getattr 调用都返回一个哑对象，避免 _get_custom_class_python_wrapper
        mod.__getattr__ = lambda *_, **__: types.SimpleNamespace()         # type: ignore
        # (b) 伪造 __path__，防止 __path__._path AttributeError
        if not getattr(mod, "__path__", None):
            class _FakePath(list): _path = []
            mod.__path__ = _FakePath()                                     # type: ignore

    _patch("_classes")
    _patch("classes")
except Exception:
    pass
# -----------------------------------------------------------------

# ③ 其余正常 import（保持不变）
# ---------- znt-GPT  main.py  (imports & 环境补丁) ----------
import os, sys, types, importlib, builtins
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"      # 彻底关闭源码热重载线程
builtins.__dict__["__torch_fake_module__"] = True  # 配合 torch 延迟加载

# 0️⃣   把 "伪 torch.{_classes, classes}" 先行注入 sys.modules
def _make_fake_torch_branch(name: str):
    fake = types.ModuleType(name)
    class _FakePath(list):         # 让 watcher 能取到 ._path
        _path: list = []
    fake.__path__ = _FakePath()    # type: ignore
    fake.__getattr__ = lambda *a, **k: types.SimpleNamespace()   # 哑响应
    sys.modules[name] = fake

for _n in ("torch.classes", "torch._classes"):
    if _n not in sys.modules:
        _make_fake_torch_branch(_n)

# 1️⃣   禁掉 Streamlit 对源码的路径扫描
import streamlit.watcher.local_sources_watcher as _sw
_sw.LocalSourcesWatcher._get_module_paths = lambda *_a, **_k: []

# 2️⃣   现在再正常 import torch，并把真模块也打补丁（稳妥起见）
try:
    torch = importlib.import_module("torch")
    for _n in ("classes", "_classes"):
        br = getattr(torch, _n, None)
        if br:
            br.__getattr__ = lambda *a, **k: types.SimpleNamespace()  # type: ignore
            if not getattr(br, "__path__", None):
                class _FakePath2(list): _path = []
                br.__path__ = _FakePath2()                            # type: ignore
except Exception:
    # torch 可能根本用不到；忽略即可
    pass
# -----------------------------------------------------------------

# 3️⃣   其余常规 import
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import pathlib, asyncio, streamlit as st
from langchain_openai import ChatOpenAI
import nest_asyncio
nest_asyncio.apply()

from utils import get_chat_response, online_search_agent
from ingest import ingest_folder
from memory import MemoryManager
# ----------  import 补丁结束 --------------------------------------
# ---------- 头部结束 --------------------------------------------



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

        # 初始化 LLM 和记忆管理器
        summary_llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_base
        )
        # 创建记忆管理器，包含短期和长期记忆，使用用户特定的向量存储路径
        memory_vector_path = str(user_db_dir / "memory_vectors")
        st.session_state["memory_manager"] = MemoryManager(
            summary_llm,
            vector_store_path=memory_vector_path
        )

        st.success("配置完成！")
        st.rerun()

# 等待配置完成
if "openai_api_key" not in st.session_state or "memory_manager" not in st.session_state:
    st.info("请在侧边栏完成配置后再使用")
    st.stop()

# 可选调试：查看历史摘要内容
#with st.sidebar:
#    if st.button("📄 查看历史摘要"):
#        if "memory_manager" in st.session_state:
#            short_term = st.session_state["memory_manager"].short_term.memory.buffer
#            st.sidebar.subheader("短期记忆")
#            st.sidebar.code(short_term)
#            
#            long_term = st.session_state["memory_manager"].get_long_term_summary()
#            st.sidebar.subheader("长期记忆摘要")
#            st.sidebar.code(long_term)

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
                    memory_manager=st.session_state["memory_manager"],
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