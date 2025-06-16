# AI 助手应用

这是一个基于 Streamlit 和 LangChain 构建的智能对话应用，支持文档问答和在线搜索功能。

## 功能特点

- 💬 智能对话：基于 GPT-3.5 的自然语言交互
- 📚 文档问答：支持对上传文档进行智能问答
- 🔍 在线搜索：集成 SerpAPI 实现实时信息检索
- 📄 多格式支持：可处理 PDF、Word、Excel、PowerPoint 等多种文档格式
- 💾 向量存储：使用 Chroma 进行高效的文档向量存储
- 🚀 本地 Embeddings：集成 HuggingFace 模型，支持本地文本向量化

## 安装说明

1. 克隆项目并创建虚拟环境：
```bash
git clone [repository-url]
cd [project-directory]
python -m venv venv
```

2. 激活虚拟环境：
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载必要的模型文件：
- 确保 `./models/all-MiniLM-L6-v2` 目录下包含所需的 HuggingFace 模型文件

## 环境变量配置

在运行应用前，需要设置以下环境变量：

- `OPENAI_API_KEY`：OpenAI API 密钥
- `SERPAPI_API_KEY`：（可选）用于在线搜索功能的 SerpAPI 密钥

## 使用说明

1. 启动应用：
```bash
streamlit run main.py
```

2. 文档处理：
```bash
python ingest.py [文档目录路径]
```

## 主要组件

- `main.py`：Streamlit 应用程序入口
- `utils.py`：核心功能实现，包括搜索代理和聊天接口
- `ingest.py`：文档处理和向量化模块

## 目录结构

```
.
├── ingest.py          # 文档处理模块
├── main.py           # 主程序入口
├── models/           # 模型文件目录
├── utils.py          # 工具函数模块
├── vector_dbs/       # 向量数据库存储
└── requirements.txt  # 项目依赖
```

## 技术栈

- Streamlit：Web 界面框架
- LangChain：大语言模型应用框架
- ChromaDB：向量数据库
- HuggingFace Transformers：本地 Embeddings 模型
- OpenAI GPT-3.5：对话模型

## 注意事项

- 确保有足够的磁盘空间用于存储向量数据库
- 首次运行可能需要下载模型文件
- 建议使用 GPU 加速文档处理（如果可用）

## License

MIT