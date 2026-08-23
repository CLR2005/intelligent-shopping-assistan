# ====================================================================
# api.py —— 智能导购助手（全屏 + 流式 Web 版）
# ====================================================================

import sys
import json
import asyncio
import concurrent.futures
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.stdout.reconfigure(line_buffering=True)

from chatbot import ask_question, ask_question_stream

app = FastAPI(title="智能导购助手", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

# -------------------- 首页（全屏 + 流式） --------------------
# HTML 每次请求时从磁盘读取，修改模板后刷新浏览器即可生效，无需重启服务。
TEMPLATE_DIR = Path(__file__).parent / "templates"

@app.get("/", response_class=HTMLResponse)
async def home():
    return (TEMPLATE_DIR / "ai_assistant.html").read_text(encoding="utf-8")


# -------------------- 迷你卡片页（仿淘宝风格，非全屏，保留版） --------------------
@app.get("/mini", response_class=HTMLResponse)
async def mini():
    html_content = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能导购助手</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: #f5f6fa; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .app {
            width: 100%;
            max-width: 480px;
            min-height: 600px;
            background: #ffffff;
            border-radius: 32px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.12);
            padding: 24px 24px 30px;
            margin: 20px;
            display: flex;
            flex-direction: column;
        }
        .header {
            font-weight: 700;
            font-size: 24px;
            color: #1a1a2e;
            margin-bottom: 4px;
        }
        .header span { color: #ff6a00; }
        .sub {
            font-size: 14px;
            color: #8c8f9c;
            margin-bottom: 16px;
        }
        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
        }
        .chip {
            background: #f0f2f8;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            cursor: pointer;
            transition: 0.2s;
            user-select: none;
        }
        .chip:hover { background: #e6e9f2; }
        .chip.primary { background: #ff6a00; color: white; }
        .chat {
            flex: 1;
            background: #fafbfc;
            border-radius: 16px;
            padding: 16px;
            min-height: 300px;
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #e9ecf2;
            margin-bottom: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .msg {
            max-width: 85%;
            padding: 10px 14px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.5;
            animation: fadeIn 0.2s ease;
        }
        .msg.user {
            align-self: flex-end;
            background: #ff6a00;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .msg.assistant {
            align-self: flex-start;
            background: white;
            border: 1px solid #e9ecf2;
            border-bottom-left-radius: 4px;
        }
        .msg .citation {
            margin-top: 6px;
            padding: 6px 10px;
            background: #f0f2f8;
            border-radius: 6px;
            font-size: 13px;
            color: #5a5d6e;
            border-left: 3px solid #ff6a00;
        }
        @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px 16px;
            border: 1.5px solid #e9ecf2;
            border-radius: 24px;
            outline: none;
            font-size: 15px;
            background: #f8f9fc;
            transition: 0.2s;
        }
        .input-area input:focus { border-color: #ff6a00; background: white; box-shadow: 0 0 0 4px rgba(255,106,0,0.08); }
        .input-area button {
            padding: 12px 24px;
            background: #ff6a00;
            color: white;
            border: none;
            border-radius: 24px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s;
        }
        .input-area button:hover { background: #e55d00; }
        .input-area button:disabled { background: #d0d4e0; cursor: not-allowed; }
        .chat::-webkit-scrollbar { width: 4px; }
        .chat::-webkit-scrollbar-thumb { background: #d0d4e0; border-radius: 8px; }
    </style>
</head>
<body>
<div class="app">
    <div class="header">🤖 智能<span>导购</span></div>
    <div class="sub">👋 输入问题或点击快捷指令</div>
    <div class="chips">
        <span class="chip primary" onclick="sendChip('帮我推荐一款性价比高的手机')">📱 推荐手机</span>
        <span class="chip" onclick="sendChip('送女朋友什么礼物好？')">🎁 送礼推荐</span>
        <span class="chip" onclick="sendChip('2000元以内的蓝牙耳机推荐')">🎧 蓝牙耳机</span>
    </div>
    <div class="chat" id="chat"></div>
    <div class="input-area">
        <input id="input" placeholder="输入您的问题..." />
        <button id="send">发送</button>
    </div>
</div>
<script>
    const chat = document.getElementById('chat');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');

    function addMessage(text, type, citations) {
        const div = document.createElement('div');
        div.className = 'msg ' + type;
        div.textContent = text;
        if (citations && citations.length) {
            const cite = document.createElement('div');
            cite.className = 'citation';
            cite.textContent = '📚 引用来源：' + citations.join('、');
            div.appendChild(cite);
        }
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
    }

    async function sendQuestion() {
        const q = input.value.trim();
        if (!q) return;
        addMessage(q, 'user');
        input.value = '';
        sendBtn.disabled = true;
        addMessage('⏳ 思考中...', 'assistant');
        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q })
            });
            const data = await res.json();
            // 移除“思考中”
            const last = chat.lastChild;
            if (last && last.textContent === '⏳ 思考中...') chat.removeChild(last);
            if (data.code === 0) {
                addMessage(data.data.answer || '(空)', 'assistant', data.data.citations || []);
            } else {
                addMessage('❌ ' + (data.message || '未知错误'), 'assistant');
            }
        } catch (e) {
            const last = chat.lastChild;
            if (last && last.textContent === '⏳ 思考中...') chat.removeChild(last);
            addMessage('❌ 请求失败，请检查服务', 'assistant');
        } finally {
            sendBtn.disabled = false;
            input.focus();
        }
    }

    function sendChip(q) {
        input.value = q;
        sendQuestion();
    }

    sendBtn.addEventListener('click', sendQuestion);
    input.addEventListener('keydown', e => { if (e.key === 'Enter') sendQuestion(); });

    addMessage('您好！我是您的智能导购助手，可以帮您推荐商品、回答购物问题。', 'assistant');
</script>
</body>
</html>
    """
    return html_content

# -------------------- 非流式接口 --------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        try:
            result = await loop.run_in_executor(pool, ask_question, request.question)
            return {
                "code": 0,
                "data": {
                    "answer": result.get('answer', ''),
                    "citations": result.get('citations', [])
                }
            }
        except Exception as e:
            return {"code": -1, "message": str(e)}

# -------------------- 流式接口（SSE，全屏界面使用） --------------------
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """以 SSE（Server-Sent Events）方式流式返回回答。

    与 /chat 的区别：不等模型生成完整答案，边生成边推送，
    前端可以像打字机一样实时显示。

    每条消息为 data: <JSON> 形式：
      {"delta": "文本增量"}     # 回答正文，可能多次
      {"citations": [...]}      # 引用文档 ID，结束前一次
      {"error": "错误说明"}     # 出错时一次
    最后发送 data: [DONE] 表示流结束。
    """
    queue: asyncio.Queue = asyncio.Queue()

    def producer():
        # 在后台线程里执行检索和模型调用，避免阻塞事件循环。
        # put_nowait 从其他线程调用是安全的，无需再包一层 run_in_executor。
        try:
            for event in ask_question_stream(request.question):
                queue.put_nowait(event)
        except BaseException as e:
            # 兜底：即使包装函数漏了异常，也要给前端一个交代。
            queue.put_nowait({"error": f"服务暂时异常（{type(e).__name__}），请稍后重试。"})
        finally:
            queue.put_nowait(None)  # None 作为流结束哨兵

    asyncio.get_running_loop().run_in_executor(None, producer)

    async def sse_stream():
        while True:
            event = await queue.get()
            if event is None:
                yield "data: [DONE]\n\n"
                break
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 经 nginx 等代理时禁用缓冲，保证实时性
        },
    )

# -------------------- 启动 --------------------
if __name__ == "__main__":
    import webbrowser, threading, time
    print("🌟 智能导购启动，请访问 http://127.0.0.1:8000")
    threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open("http://127.0.0.1:8000")), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)