"""智能客服程序入口。

本文件只负责协调用户交互和业务流程，具体工作由其他模块完成：

* ``indexer.hybrid_search``：从本地知识库检索相关文档；
* ``generator.generate_answer``：根据问题和检索结果调用大模型生成答案。

每个问题的处理流程如下：

1. 接收并清理用户输入；
2. 使用混合检索获取候选文档；
3. 根据最高综合得分决定使用 RAG 客服模式还是自由聊天模式；
4. 生成答案并显示引用来源；
5. 捕获单次请求异常，保证主循环可以继续服务下一个问题。

ask_question_stream 是流式版本，供 api.py 的 Web 界面实时显示使用。
"""

import sys

# 混合检索函数会同时结合向量相似度和关键词匹配结果。
from indexer import hybrid_search

# 生成器负责组织提示词、请求 DeepSeek，并解析/流式返回模型结果。
from generator import generate_answer, stream_answer

# 让终端中的状态信息立即显示，便于观察检索和 API 请求进度。
sys.stdout.reconfigure(line_buffering=True)

# 只有最高综合得分达到该值，才认为知识库中存在足够相关的内容。
# 分数低于阈值时传入空文档列表，生成器会自动切换到自由聊天模式。
# 【校准依据】离线哈希向量下实测：退货问题约 0.477，客服电话问题约 1.074，
# 无关问题最高约 0.175，因此取 0.4 可以让相关问题进入 RAG，过滤无关结果。
# （旧值 1.8 过高，导致 RAG 客服模式无法触发；0.6 又会误判退货问题。）
MIN_SCORE_THRESHOLD = 0.4


def _retrieve_docs(question: str) -> list:
    """执行混合检索并根据得分阈值选择回答模式。

    返回:
        检索到的候选文档列表；得分不足（或没有结果）时返回空列表，
        此时 generator 会自动切换到自由聊天模式。
    """
    # 只取两个候选文档，兼顾上下文完整性和 API 请求长度。
    docs = hybrid_search(question, top_k=2)

    # 检索结果按综合得分从高到低排列，因此第一个结果就是最高分。
    # 没有任何结果时使用 0.0，确保后面的比较不会访问空列表。
    best_score = docs[0]['score'] if docs else 0.0

    print(f"📊 当前最高综合得分: {best_score:.3f} (阈值: {MIN_SCORE_THRESHOLD})")

    # 分数不足说明知识库可能没有覆盖当前问题。
    # 返回空列表后，generator.py 会使用通用对话提示词回答。
    if not docs or best_score < MIN_SCORE_THRESHOLD:
        print("   → 触发【自由聊天模式】")
        return []

    # 分数足够高时，将检索文档作为上下文交给生成器，
    # 让模型严格依据客服知识库作答，并返回引用文档 ID。
    print("   → 进入【RAG 客服模式】")
    return docs


def ask_question(question: str):
    """处理一个用户问题并将最终答案打印到终端。

    参数:
        question: 已经由主循环读取并去除首尾空白的用户问题。

    注意:
        本函数会隔离单个问题产生的异常，因此一次 API 错误或检索错误
        不会直接终止整个客服程序。
    """
    try:
        # 先回显问题，让用户知道当前正在处理哪一条请求。
        print(f"\n👤 用户问: {question}")
        print("🤔 正在使用【混合检索】查找知识库...")

        # 混合检索 + 得分阈值判断（自由聊天模式返回空列表）。
        docs = _retrieve_docs(question)

        final_answer = generate_answer(question, docs)

        # 统一格式输出答案，便于用户阅读，也便于后续替换成 Web 界面。
        print("\n" + "="*50)
        print(f"🤖 AI 回复: {final_answer.get('answer', '无内容')}")

        # 自由聊天模式通常没有引用；只有存在有效引用时才显示该行。
        if final_answer.get('citations'):
            print(f"📚 引用来源: {', '.join(final_answer['citations'])}")
        print("="*50 + "\n")

        # 【关键】把结果返回给调用方（api.py 的 /chat 接口依赖这个返回值）。
        # 只打印不返回会让 Web 接口拿到 None，误报"内部逻辑返回格式异常"。
        return final_answer

    except KeyboardInterrupt:
        # 用户在处理请求期间按 Ctrl+C，只取消当前请求。
        print("\n⚠️ 用户中断了当前问题处理")
        # 必须返回统一格式，否则 api.py 会误报"内部逻辑返回格式异常"。
        return {"answer": "请求已取消", "citations": []}

    except BaseException as e:
        # 兜底捕获异常，避免一次请求失败导致整个程序退出。
        # 这里保留异常类型，便于定位网络、数据或解析问题。
        print(f"\n❌ 处理问题时发生意外错误: {type(e).__name__} - {e}")
        # 把错误信息装进 answer 返回给 Web 前端，用户才能看到真实原因。
        return {
            "answer": f"服务暂时异常（{type(e).__name__}），请稍后重试。",
            "citations": []
        }


def ask_question_stream(question: str):
    """处理一个用户问题，并以流式方式逐步产出事件（供 api.py 的流式接口使用）。

    与 ask_question 的区别：不等模型生成完整答案，而是边生成边产出，
    Web 前端可以像打字机一样实时显示。

    参数:
        question: 已经由前端读取并去除首尾空白的用户问题。

    产出格式（逐个 yield 字典）:
        {"delta": "文本增量"}       # 可能多次
        {"citations": [...]}        # 流结束时一次
        {"error": "错误说明"}       # 出错时一次，之后流结束
    """
    try:
        # 与 ask_question 相同的检索流程。
        print(f"\n👤 用户问: {question}")
        print("🤔 正在使用【混合检索】查找知识库...")
        docs = _retrieve_docs(question)

        # 把 generator 产出的事件原样转发给上层（api.py 的 SSE 接口）。
        yield from stream_answer(question, docs)

    except BaseException as e:
        # 兜底捕获异常（主要是检索阶段的），保证流以错误事件正常收尾。
        print(f"\n❌ 流式处理问题时发生意外错误: {type(e).__name__} - {e}")
        yield {"error": f"服务暂时异常（{type(e).__name__}），请稍后重试。"}

# -------------------- 交互入口 --------------------
if __name__ == "__main__":
    # 只有直接运行 chatbot.py 时才启动交互循环；
    # 被其他模块 import 时不会意外阻塞等待输入。
    print("🌟 智能客服已启动（输入 quit 退出）")
    print(f"💡 当前检索降级阈值: {MIN_SCORE_THRESHOLD}\n")
    
    while True:
        try:
            # strip() 去掉用户误输入的首尾空格和换行符。
            user_input = input("请输入您的问题: ").strip()

            # 支持多个常见退出命令，并统一按小写比较，减少输入限制。
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            # 空输入不发送检索和 API 请求，直接重新显示输入提示。
            if not user_input:
                continue

            # 每次只处理一条问题；函数内部异常不会破坏循环。
            ask_question(user_input)
        except KeyboardInterrupt:
            # 主循环中的 Ctrl+C 表示用户希望退出整个程序。
            print("\n👋 检测到中断，程序退出。")
            break
        except BaseException as e:
            # 处理输入阶段的意外错误，并继续等待下一条问题。
            print(f"❌ 顶层循环异常: {e}")
            # 不退出，继续运行。