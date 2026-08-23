# 智能导购助手

基于 RAG（检索增强生成）的智能导购 / 客服助手。项目通过 ChromaDB 本地知识库和混合检索（向量相似度 + 关键词匹配）召回相关内容，再由 DeepSeek 大模型生成回答，支持终端问答、流式 Web 界面和离线索引构建。

## 功能特性

- 混合检索：结合 ChromaDB 向量检索与关键词打分，兼顾语义理解与精确匹配。
- RAG 客服模式：命中知识库时严格基于参考文档回答，并返回引用来源。
- 自由聊天模式：知识库未命中时自动切换到通用 AI 对话。
- 流式 Web 界面：FastAPI + SSE 边生成边显示，交互更流畅。
- 自动重试：网络抖动或 API 限流时指数退避重试，最多 3 次。
- 离线友好：内置轻量哈希 Embedding，无需额外下载模型即可启动。

## 技术栈

| 组件 | 用途 |
| --- | --- |
| Python 3 | 开发语言 |
| FastAPI / Uvicorn | Web 服务与流式接口 |
| ChromaDB | 本地向量数据库 |
| langchain-text-splitters | 知识库文本切分 |
| DeepSeek API | 大模型生成 |
| tenacity | API 调用重试 |
| python-dotenv | 环境变量管理 |

## 项目结构

| 路径 | 说明 |
| --- | --- |
| `api.py` | FastAPI 入口：Web 页面、`/chat`、`/chat/stream` |
| `chatbot.py` | 问答流程编排：混合检索、模式选择、答案输出 |
| `generator.py` | 大模型调用：RAG 模式、自由聊天模式、流式生成 |
| `indexer.py` | 离线索引构建：读取知识库、切分、写入 ChromaDB |
| `knowledge_base.txt` | 客服知识库原始文本 |
| `templates/ai_assistant.html` | 全屏 Web 前端 |
| `chroma_data/` | 本地向量数据库（已加入 `.gitignore`） |

## 快速开始

### 1. 安装依赖

```bash
cd 智能客服
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=sk-你的密钥
```

### 3. 构建知识库索引

```bash
python indexer.py
```

脚本会把 `knowledge_base.txt` 切分后写入本地 ChromaDB。

### 4. 启动服务

Web 界面：

```bash
python api.py
```

浏览器访问 `http://127.0.0.1:8000`。

终端交互：

```bash
python chatbot.py
```

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 全屏 Web 界面 |
| GET | `/mini` | 迷你卡片页面 |
| POST | `/chat` | 非流式问答 |
| POST | `/chat/stream` | SSE 流式问答 |

请求体：

```json
{"question": "退货政策是什么？"}
```

非流式响应：

```json
{
  "code": 0,
  "data": {
    "answer": "回答内容",
    "citations": ["chunk_0"]
  }
}
```

`/chat/stream` 返回 `data: {"delta": "..."}`、`data: {"citations": [...]}`，最后以 `data: [DONE]` 结束。

## 核心参数

| 参数 | 位置 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `MIN_SCORE_THRESHOLD` | `chatbot.py` | `0.4` | 触发 RAG 客服模式的最低综合得分 |
| `CHUNK_SIZE` | `indexer.py` | `80` | 知识库切分最大字符数 |
| `CHUNK_OVERLAP` | `indexer.py` | `20` | 切分重叠字符数 |
| `MODEL_NAME` | `generator.py` | `deepseek-chat` | DeepSeek 模型名称 |

## 常见问题

- 修改 `knowledge_base.txt` 后回答没有更新：重新运行 `python indexer.py`。
- 报 API Key 错误：确认项目根目录存在 `.env`，且变量名是 `DEEPSEEK_API_KEY`。
- 端口被占用：修改 `api.py` 末尾 `uvicorn.run(...)` 的 `port` 参数。
