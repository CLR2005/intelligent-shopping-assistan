# ====================================================================
# generator.py —— 大模型生成器（RAG模式 + 自由聊天模式）
# ====================================================================
# 【核心功能】
#   - 如果传入 docs 非空：进入 RAG 客服模式（基于文档回答）
#   - 如果传入 docs 为空：进入自由聊天模式（通用AI回答）
#   - generate_answer：一次性生成完整答案（终端 / 非流式接口使用）
#   - stream_answer：流式逐字生成答案（Web 界面实时显示使用）
# 【健壮性】
#   - 强制刷新输出缓冲区，确保调试信息实时显示
#   - 捕获所有可能的异常（包括 Ctrl+C），绝不静默崩溃
# ====================================================================

import os
import json
import re
import sys
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# 强制 Python 实时刷新 print 输出（解决终端卡住不显示的问题）
sys.stdout.reconfigure(line_buffering=True)

# 加载 .env 文件中的 API Key
load_dotenv()

# -------------------- 配置区 --------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"          # 稳定模型
CONNECT_TIMEOUT = 10                  # 连接超时（秒）
READ_TIMEOUT = 30                     # 读取超时（秒）


# -------------------- 1. 带重试的 API 调用函数 --------------------
@retry(
    stop=stop_after_attempt(3),        # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数退避 1s, 2s, 4s
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def call_deepseek_with_retry(messages):
    """
    发送消息给 DeepSeek，带自动重试机制
    如果网络抖动或限流（429），会自动等待重试
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,             # 低温度让回答更严谨
        "response_format": {"type": "json_object"}  # 强制输出合法 JSON
    }

    print("   📡 正在向 DeepSeek API 发送请求...")  # 关键调试信息

    response = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )

    # 如果状态码不是 200，抛出异常让 tenacity 触发重试
    if response.status_code != 200:
        raise Exception(f"API 返回错误: {response.status_code} - {response.text}")

    print("   ✅ DeepSeek API 响应成功，正在解析结果...")
    return response.json()


# -------------------- 2. 核心生成函数（你只需要调用这一个） --------------------
def generate_answer(query: str, docs: list) -> dict:
    """
    根据是否传入文档，自动选择模式生成回答

    参数:
        query: 用户问题（字符串）
        docs: 文档列表，格式为 [{"id": "chunk_0", "content": "..."}, ...]
              - 传入非空列表 -> RAG 客服模式（基于文档回答）
              - 传入空列表 [] -> 自由聊天模式（通用 AI 回答）

    返回:
        {"answer": "回答内容", "citations": ["引用的文档ID"]}
    """
    try:
        # ========== 模式 A：没有文档 → 自由聊天 ==========
        if not docs:
            print("   [模式] 自由聊天（使用大模型通用知识）")
            system_prompt = """你是一个智能AI助手。请用中文直接回答用户的问题。
规则：
1. 如果知道答案就准确回答，如果不知道就明确说"我不知道"。
2. 你必须输出合法的JSON格式：{"answer": "回答内容", "citations": []}
3. citations 字段固定为空列表，不要填任何内容。"""
            user_prompt = query

        # ========== 模式 B：有文档 → RAG 客服 ==========
        else:
            print("   [模式] RAG 客服（基于检索到的文档回答）")
            # 最多取前 3 段文档，防止超出上下文窗口
            context_parts = []
            valid_ids = []
            for doc in docs[:3]:
                context_parts.append(f"[文档ID: {doc['id']}]\n{doc['content']}")
                valid_ids.append(doc['id'])

            final_context = "\n\n".join(context_parts)

            system_prompt = """你是一个专业的智能客服助手。请严格基于以下"参考文档"回答用户的问题。
规则：
1. 如果文档中有相关信息，请根据这些信息组织成流畅、礼貌的客服话术。
2. 如果文档内容完全与用户问题无关，请直接说"未找到相关答案"。
3. 你必须输出合法的JSON格式，包含两个字段：
   - "answer": 字符串，你的回答内容
   - "citations": 字符串列表，列出你实际引用的文档ID（必须来源于下面的文档）
4. 不要输出任何Markdown标记或额外解释，只输出纯JSON。"""

            user_prompt = f"""【参考文档】
{final_context}

【用户问题】
{query}
"""

        # 组装消息体
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # ---------- 调用大模型（所有异常都会被下面的 except 捕获） ----------
        result_json = call_deepseek_with_retry(messages)
        raw_content = result_json["choices"][0]["message"]["content"]

        # 清洗可能被包裹的 Markdown 标记（如 ```json ... ```）
        cleaned = raw_content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # 解析为 Python 字典
        parsed = json.loads(cleaned)

        # ---------- 校验 citations（防大模型编造不存在的 ID） ----------
        if docs:
            valid_ids = [doc['id'] for doc in docs]
            if "citations" in parsed and isinstance(parsed["citations"], list):
                parsed["citations"] = [cid for cid in parsed["citations"] if cid in valid_ids]
            else:
                parsed["citations"] = []
        else:
            # 自由聊天模式，强制 citations 为空
            parsed["citations"] = []

        # 如果 answer 字段为空，给一个默认值
        if not parsed.get("answer"):
            parsed["answer"] = "未能生成有效回复，请稍后重试。"

        return parsed

    # ---------- 捕获 Ctrl+C（用户主动中断） ----------
    except KeyboardInterrupt:
        print("\n   ⚠️ 用户中断了当前请求（按下了 Ctrl+C）")
        return {"answer": "请求已取消", "citations": []}

    # ---------- 捕获所有其他异常（包括系统级错误） ----------
    except BaseException as e:
        print(f"   ❌ 生成回答时发生异常: {type(e).__name__} - {e}")
        # 返回一个友好的错误信息给 chatbot.py 显示
        return {
            "answer": f"服务暂时异常（{type(e).__name__}），请稍后重试。",
            "citations": []
        }


# -------------------- 3. 流式生成函数（Web 界面实时显示用） --------------------
# 【与 generate_answer 的区别】
#   - generate_answer 要求模型输出 JSON 并整体解析，只能一次性返回；
#   - stream_answer 要求模型直接输出纯文本，边生成边 yield，前端实时显示。
# 【引用方案】
#   流式模式下无法边输出边解析 JSON，因此让模型在回答末尾另起一行输出
#   引用标记 [CITATIONS: chunk_0, chunk_1]，后端在流结束后解析该标记：
#   - 标记之前的文本才是正文（已实时发送给前端）；
#   - 标记中的文档 ID 会经过校验，只保留真实存在于检索结果里的 ID；
#   - 如果模型没有输出标记（RAG 模式），退回引用全部检索文档。

CITATION_PATTERN = re.compile(r"\[CITATIONS\s*:\s*([^\]]*)\]")
STREAM_HOLD_BACK = 30   # 尾部保留字符数，防止 [CITATIONS 被分块截断后把半截发给前端


@retry(
    stop=stop_after_attempt(3),        # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数退避 1s, 2s, 4s
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def call_deepseek_stream_with_retry(messages):
    """
    以流式方式请求 DeepSeek，建立连接阶段失败会自动重试。
    返回保持连接的 response，正文由调用方逐行读取。

    注意：流式请求不能携带 response_format=json_object，
    否则模型会输出 JSON 文本，前端无法获得自然的打字机效果。
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,             # 与非流式保持一致
        "stream": True                  # 开启流式输出
    }

    print("   📡 正在向 DeepSeek API 发送流式请求...")

    response = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        stream=True
    )

    # 如果状态码不是 200，抛出异常让 tenacity 触发重试
    if response.status_code != 200:
        raise Exception(f"API 返回错误: {response.status_code} - {response.text}")

    print("   ✅ DeepSeek 流式连接建立，开始接收内容...")
    return response


def stream_answer(query: str, docs: list):
    """
    流式生成回答，逐个产出事件字典（供 chatbot.py / api.py 的流式接口使用）

    参数:
        query: 用户问题（字符串）
        docs: 文档列表，格式同 generate_answer（非空 -> RAG 模式，空 -> 自由聊天）

    产出格式:
        {"delta": "文本增量"}        # 可能多次
        {"citations": [...]}        # 流结束时恰好一次（自由聊天模式恒为空列表）
        {"error": "错误说明"}        # 出错时恰好一次，之后流立即结束
    """
    try:
        # ========== 组织提示词（模式判断与 generate_answer 一致） ==========
        if not docs:
            print("   [模式] 自由聊天（使用大模型通用知识）")
            system_prompt = """你是一个智能AI助手。请用中文直接回答用户的问题。
规则：
1. 如果知道答案就准确回答，如果不知道就明确说"我不知道"。
2. 使用纯文本输出，不要使用 JSON、Markdown 代码块或任何特殊标记。"""
            user_prompt = query
        else:
            print("   [模式] RAG 客服（基于检索到的文档回答）")
            # 最多取前 3 段文档，防止超出上下文窗口
            context_parts = []
            for doc in docs[:3]:
                context_parts.append(f"[文档ID: {doc['id']}]\n{doc['content']}")
            final_context = "\n\n".join(context_parts)

            system_prompt = """你是一个专业的智能客服助手。请严格基于以下"参考文档"回答用户的问题。
规则：
1. 如果文档中有相关信息，请根据这些信息组织成流畅、礼貌的客服话术。
2. 如果文档内容完全与用户问题无关，请直接说"未找到相关答案"。
3. 使用纯文本输出，不要使用 Markdown 标记、不要输出 JSON。
4. 回答结束后，另起一行输出引用标记，格式为：[CITATIONS: 文档ID1, 文档ID2]
   只列出你实际引用的文档 ID（必须来源于参考文档中的 [文档ID: xxx]）；
   如果没有任何文档被引用，输出 [CITATIONS: ]"""

            user_prompt = f"""【参考文档】
{final_context}

【用户问题】
{query}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # ---------- 建立流式连接（连接失败由 tenacity 自动重试） ----------
        response = call_deepseek_stream_with_retry(messages)
        # DeepSeek 的 SSE 响应没有声明字符集，强制按 UTF-8 解码中文。
        response.encoding = "utf-8"

        full_text = ""          # 模型输出的完整文本（含末尾引用标记）
        pending = ""            # 尚未发送给前端的尾部文本
        marker_started = False  # 是否已检测到 [CITATIONS 开头
        sent_any = False        # 是否已向调用方发送过正文增量

        # 逐行解析 SSE 数据：{"choices":[{"delta":{"content":"..."}}]}
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue    # 忽略解析失败的行，不中断整个流

            delta = obj.get("choices", [{}])[0].get("delta") or {}
            piece = delta.get("content")
            if not piece:
                continue

            full_text += piece

            if marker_started:
                # 引用标记已开始，后续内容全部缓存，流结束后统一解析。
                pending += piece
                continue

            pending += piece
            pos = pending.find("[CITATIONS")
            if pos != -1:
                # 检测到引用标记开头：把标记之前的正文发出去，之后进入缓存模式。
                if pos > 0:
                    yield {"delta": pending[:pos]}
                    sent_any = True
                pending = pending[pos:]
                marker_started = True
            elif len(pending) > STREAM_HOLD_BACK:
                # 正常流式输出。发送前检查将要发出去的头部是否出现 "["，
                # 防止 [CITATIONS 跨越多个分块时把标记的开头误发给前端。
                head = pending[:-STREAM_HOLD_BACK]
                tail = pending[-STREAM_HOLD_BACK:]
                cut = head.find("[")
                if cut == -1:
                    yield {"delta": head}
                    sent_any = True
                    pending = tail
                else:
                    # 从 "[" 处截断，剩下的并入尾部缓存，等流结束后统一处理。
                    if cut > 0:
                        yield {"delta": head[:cut]}
                        sent_any = True
                    pending = head[cut:] + tail

        # ---------- 流结束：解析引用标记，校验 ID ----------
        citations = []
        match = CITATION_PATTERN.search(full_text)
        if match:
            # 标记之前的内容是正文，已全部发出；pending 里只剩标记及残留文本。
            if docs:
                valid_ids = [doc['id'] for doc in docs]
                citations = [cid.strip() for cid in match.group(1).split(",")
                             if cid.strip() in valid_ids]
        else:
            # 模型没有输出合法标记：
            #   - RAG 模式退回引用全部检索文档，保证引用功能仍可用；
            #   - 若标记只写了一半（流被截断），丢弃残留的半个标记。
            if docs:
                citations = [doc['id'] for doc in docs]
            if pending and not marker_started:
                yield {"delta": pending}
                sent_any = True

        # 整个流没有产出任何正文时给一个兜底提示，避免前端出现空气泡。
        if not sent_any:
            yield {"delta": "未能生成有效回复，请稍后重试。"}

        yield {"citations": citations}

    # ---------- 捕获 Ctrl+C（用户主动中断） ----------
    except KeyboardInterrupt:
        print("\n   ⚠️ 用户中断了当前请求（按下了 Ctrl+C）")
        yield {"error": "请求已取消"}

    # ---------- 捕获所有其他异常（包括系统级错误） ----------
    except BaseException as e:
        print(f"   ❌ 流式生成时发生异常: {type(e).__name__} - {e}")
        yield {"error": f"服务暂时异常（{type(e).__name__}），请稍后重试。"}


# -------------------- 4. 单独测试本文件（可选） --------------------
if __name__ == "__main__":
    print("="*50)
    print("测试 generator.py（单独运行）")
    print("="*50)

    # 测试 1：自由聊天模式
    print("\n>> 测试 1: 自由聊天（1+1=？）")
    result1 = generate_answer("1+1=？", [])
    print(json.dumps(result1, ensure_ascii=False, indent=2))

    # 测试 2：RAG 模式（模拟文档）
    print("\n>> 测试 2: RAG 模式（怎么退货？）")
    mock_docs = [
        {"id": "chunk_0", "content": "退货政策：用户在签收商品后7天内，在不影响二次销售的情况下可以申请无理由退货，邮费由用户承担。"}
    ]
    result2 = generate_answer("怎么退货？", mock_docs)
    print(json.dumps(result2, ensure_ascii=False, indent=2))