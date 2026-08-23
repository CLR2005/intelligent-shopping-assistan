# ====================================================================
# indexer.py —— 离线知识库构建器（混合检索 + 单字符过滤）
# ====================================================================
# 【核心升级】
#   - 关键词提取时，强制过滤掉长度 < 2 的英文/数字（如 "1", "a"）
#   - 彻底解决 "1+1=？" 因数字 "1" 误匹配 "7天/15天" 的问题
# ====================================================================

import os
import re
import hashlib
import math
import shutil
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================================================================
# 第一部分：全局配置（你可以在这里随意调参）
# ====================================================================
CHUNK_SIZE = 80                 # 每段最大字符数
CHUNK_OVERLAP = 20              # 段落间重叠字符数
COLLECTION_NAME = "customer_service_kb"
SOURCE_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_data"
PERSIST_DIR = Path(os.getenv("LOCALAPPDATA", Path.home())) / "智能客服" / "chroma_data"
KEYWORD_WEIGHT = 1.5            # 关键词命中率的权重系数

# 工作区目录可能来自只读位置（例如受保护的共享目录），将已有索引
# 复制到当前用户目录，保证 ChromaDB 可以创建 SQLite 临时文件和锁文件。
PERSIST_DIR.parent.mkdir(parents=True, exist_ok=True)
if SOURCE_PERSIST_DIR.exists() and not PERSIST_DIR.exists():
    shutil.copytree(SOURCE_PERSIST_DIR, PERSIST_DIR)

# ====================================================================
# 第二部分：初始化 Embedding 模型（全局只加载一次）
# ====================================================================
print(">> [1/5] 初始化本地 Embedding 模型（all-MiniLM-L6-v2）...")


class LocalEmbeddingFunction:
    """无需下载模型的确定性向量，供离线环境启动和检索使用。"""

    dimension = 384

    def name(self) -> str:
        return "local_hash_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            vector = [0.0] * self.dimension
            normalized_text = text.lower()
            terms = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", normalized_text)
            for term in terms:
                digest = hashlib.sha256(term.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            embeddings.append(
                [value / norm for value in vector] if norm else vector
            )
        return embeddings


try:
    if os.getenv("USE_REMOTE_EMBEDDING") != "1":
        raise RuntimeError("默认使用离线 embedding；设置 USE_REMOTE_EMBEDDING=1 才加载模型")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
except Exception as error:
    print(f"   ⚠️ 本地模型不可用（{type(error).__name__}），切换到离线检索向量。")
    embedding_fn = LocalEmbeddingFunction()

# ====================================================================
# 第三部分：连接 / 创建向量数据库（供 chatbot.py 导入）
# ====================================================================
print(">> [2/5] 连接本地 ChromaDB 向量数据库...")
client = chromadb.PersistentClient(path=str(PERSIST_DIR))
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    USE_EXPLICIT_EMBEDDINGS = True
except Exception:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    USE_EXPLICIT_EMBEDDINGS = False
print(f"   数据库连接成功！当前集合: {COLLECTION_NAME}")

# ====================================================================
# 第四部分：核心函数（关键词命中率 + 混合检索）
# ====================================================================

def _keyword_score(query_text: str, document: str) -> float:
    """
    计算关键词命中率（已过滤单字符噪声）

    原理：
        1. 中文拆分为两两组合（Bigram），如“退货” -> “退货”
        2. 英文/数字提取单词，但【只保留长度 >= 2 的】
           - 例如 "1+1=？" 会提取出 ['1', '1']，但被过滤掉，剩空列表
           - 例如 "VIP客服" 会提取出 ['VIP']，正常匹配
        3. 统计命中比例
    """
    # 1. 提取中文 Bigram（双字词）
    chinese_terms = re.findall(r"[\u4e00-\u9fff]+", query_text.lower())
    terms = {
        term[index:index + 2]
        for term in chinese_terms
        for index in range(len(term) - 1)
    }

    # 2. 提取英文/数字，但【只保留长度 >= 2 的】（关键修复！）
    raw_eng_terms = re.findall(r"[A-Za-z0-9]+", query_text.lower())
    terms.update([t for t in raw_eng_terms if len(t) >= 2])

    # 3. 如果没有提取到任何有效关键词，直接返回 0
    if not terms:
        return 0.0

    # 4. 统计命中比例
    matched_terms = sum(term in document.lower() for term in terms)
    return matched_terms / len(terms)


def hybrid_search(query_text: str, top_k: int = 2) -> list:
    """
    混合检索：向量相似度 + 关键词加权（已过滤单字符）

    返回格式：
        [
            {
                "id": "chunk_0",
                "content": "退货政策...",
                "score": 1.523,
                "vector_score": 0.823,
                "keyword_score": 0.467
            },
            ...
        ]
    """
    # 1. 向量检索：全量捞取（让关键词排序重新洗牌）
    query_args = {
        "n_results": collection.count(),
    }
    if USE_EXPLICIT_EMBEDDINGS:
        query_args["query_embeddings"] = embedding_fn([query_text])
    else:
        query_args["query_texts"] = [query_text]
    results = collection.query(**query_args)

    # 2. 如果数据库为空，直接返回空列表
    if not results['documents'] or not results['documents'][0]:
        return []

    # 3. 遍历每条结果，计算综合得分
    ranked_results = []
    for idx, doc_text in enumerate(results['documents'][0]):
        # 提取向量距离（余弦距离，越接近0越相似）
        dist = results['distances'][0][idx]
        vector_similarity = 1 - dist  # 转为相似度（越大越好）

        # 计算关键词命中率（此时已自动过滤单字符）
        keyword_score = _keyword_score(query_text, doc_text)

        # 综合得分 = 向量相似度 + 权重 × 关键词命中率
        combined_score = vector_similarity + KEYWORD_WEIGHT * keyword_score

        ranked_results.append({
            "id": results['ids'][0][idx],
            "content": doc_text,
            "score": combined_score,
            "vector_score": vector_similarity,
            "keyword_score": keyword_score
        })

    # 4. 按综合得分降序排序，取 Top-K
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results[:top_k]


# ====================================================================
# 第五部分：离线索引构建（只有直接运行本文件时才执行）
# ====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  开始执行离线索引构建（备菜流程）")
    print("="*60 + "\n")

    # ---------- 5.1 读取原始知识库 ----------
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
        print(f">> [3/5] 读取知识库成功，共 {len(raw_text)} 个字符。")
    except FileNotFoundError:
        print("❌ 致命错误：找不到 knowledge_base.txt 文件！")
        exit(1)

    if not raw_text.strip():
        print("❌ 知识库文件为空，请写入有效内容。")
        exit(1)

    # ---------- 5.2 智能切分 ----------
    print(">> [4/5] 正在切分文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"   切分完成，共生成 {len(chunks)} 个片段。")

    if not chunks:
        print("⚠️ 警告：切分结果为空。")
        exit(0)

    # ---------- 5.3 清理旧数据（幂等性） ----------
    if collection.count() > 0:
        print(f">> 检测到旧数据 ({collection.count()} 条)，正在清空...")
        old_ids = collection.get()['ids']
        if old_ids:
            collection.delete(ids=old_ids)
        print("   旧数据已清空。")

    # ---------- 5.4 准备新数据并入库 ----------
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": "knowledge_base.txt",
            "index": i,
            "id": ids[i]      # 把 id 也存进元数据，方便追溯
        }
        for i in range(len(chunks))
    ]

    print(">> [5/5] 正在计算向量并存入 ChromaDB（可能需要几秒钟）...")
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=embedding_fn(chunks) if USE_EXPLICIT_EMBEDDINGS else None
    )

    print(f"\n✅ 大功告成！共入库 {len(chunks)} 条向量数据。")
    print(f"   数据存储位置: {os.path.abspath(PERSIST_DIR)}")
    print("   💡 现在你可以运行 python chatbot.py 来提问了！")

    # ---------- 5.5 自动验证：混合检索测试 ----------
    print("\n" + "="*60)
    print("  自动验证：混合检索测试（含单字符过滤效果）")
    print("="*60)

    test_queries = [
        "我想退货怎么操作？",
        "你们客服电话多少？",
        "1+1=？"   # 这个测试会展示过滤效果
    ]

    for q in test_queries:
        print(f"\n>> 测试查询: '{q}'")
        results = hybrid_search(q, top_k=2)

        if not results:
            print("   ⚠️ 未找到任何相关文档。")
            continue

        for rank, res in enumerate(results, start=1):
            # 打印详细分数，让你看清关键词得分是否为 0
            print(f"  第{rank}名 (综合: {res['score']:.3f} | 向量: {res['vector_score']:.3f} | 关键词: {res['keyword_score']:.3f})")
            print(f"    内容: {res['content'][:50]}...")

    print("\n" + "="*60)
    print("  索引构建及测试全部完成！")
    print("="*60)