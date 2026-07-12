import json
from rag.BM25 import bm25_search_law_qa, bm25_search_law, ChineseAnalyzer
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import torch
from sentence_transformers import CrossEncoder
import numpy as np


# 加载 embedding 模型（GTE-base-zh）
batch_size = 64
device = os.getenv("RAG_DEVICE", "cpu")

model_kwargs = {"attn_implementation": "eager"}

if device == "cpu":
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.graphs.is_current_stream_capturing = lambda: False

model = SentenceTransformer(
    "BAAI/bge-large-zh-v1.5",
    device=device,
    model_kwargs=model_kwargs,
)
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-large",
            device=device,
            model_kwargs=model_kwargs,
        )
    return _reranker

# 连接 chromadb（本地持久化）
client_law = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_law_db"))
collection_law = client_law.get_or_create_collection(name="legal_corpus")

client_law_qa = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_law_qa_db"))
collection_law_qa = client_law_qa.get_or_create_collection(name="legal_QA")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,          # 每块最大字符数
    chunk_overlap=100,       # 块之间重叠的字符数
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    length_function=len,
)

def rerank(query, candidates, top_k):
    """
    对候选文档进行重排序
    query: 用户查询字符串
    candidates: 候选文档列表，每个元素是一个 dict，至少包含 "id" 和 "text"（文档内容）字段
               例如: [{"id": "123", "text": "文档内容..."}, ...]
    top_k: 返回重排后得分最高的前 k 个文档
    返回: 列表，每个元素为 (doc_id, score, text)，按得分降序排列
    """
    # 构建 (query, 文档文本) 对
    if not candidates:
        return []

    if os.getenv("RAG_ENABLE_RERANK", "0") != "1":
        return candidates[:top_k]

    pairs = [(query, cand["text"]) for cand in candidates]

    # 预测相关性分数（返回 float 列表，值越大越相关）
    scores = get_reranker().predict(pairs)

    # 将分数与原始候选文档绑定
    scored_candidates = []
    for i, cand in enumerate(candidates):
        scored_candidates.append({
            "id": cand["id"],
            "text": cand["text"],
            "score": float(scores[i])
        })

    # 按分数降序排序
    sorted_results = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

    # 返回前 top_k 个
    return sorted_results[:top_k]

def join_corpus_to_db():
    if collection_law.count() == 0:
        data_path = PROJECT_ROOT / "data" / "legal_statutes.jsonl"
        count = 0
        ids = []
        embeddings = []
        sentences = []
        documents = []
        metadatas = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)

                    ids.append(str(count))
                    count += 1
                    sentences.append(item['name'])
                    documents.append(item['content'])
                    metadatas.append({'name': item['name'], 'content': item['content']})

                    chunks = text_splitter.split_text(item['content'])

                    for i in range(0, len(chunks)):
                        ids.append(str(count))
                        count += 1
                        sentences.append(chunks[i])
                        documents.append(item['name'])
                        metadatas.append({'name': item['name'], 'content': item['content']})

            embeddings = model.encode(sentences, batch_size=batch_size).tolist()
            print("向量生成完毕")
            # 分批次添加
            Batch_size = 5000  # 安全值，小于 Chroma 限制 5461
            total = len(ids)
            for start in range(0, total, Batch_size):
                end = start + Batch_size
                collection_law.add(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end]
                )
                print(f"已添加 {end}/{total} 条记录")

        print(f"已存入 {collection_law.count()} 条法律条文")

def join_qa_to_db():
    if collection_law_qa.count() == 0:
        data_path = PROJECT_ROOT / "data" / "legal_qa_cases.jsonl"
        count = 0
        ids = []
        embeddings = []
        sentences = []
        documents = []
        metadatas = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)

                    chunks = text_splitter.split_text(item['input'])

                    for chunk in chunks:
                        ids.append(str(count))
                        count += 1
                        sentences.append(chunk)
                        documents.append(item['output'])
                        metadatas.append({'query': item['input'], 'answer': item['output']})

            embeddings = model.encode(sentences, batch_size=batch_size).tolist()
            print("向量生成完毕")
            # 分批次添加
            Batch_size = 5000  # 安全值，小于 Chroma 限制 5461
            total = len(ids)
            for start in range(0, total, Batch_size):
                end = start + Batch_size
                collection_law_qa.add(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end]
                )
                print(f"已添加 {end}/{total} 条记录")
        print(f"已存入 {collection_law_qa.count()} 条法律问答对话")

def corpus_query(query, n_results):
    embedding = model.encode(query).tolist()
    result1 = collection_law.query(query_embeddings=embedding, n_results=n_results)
    result2 = bm25_search_law(query, str(PROJECT_ROOT / "bm25_index_law"), top_k=n_results)
    candidates = []
    for i in range(len(result1['metadatas'][0])):
        candidates.append({'id':result1['metadatas'][0][i]['name'],'text':result1['metadatas'][0][i]['content']})
    for law_name, content, score in result2:
        candidates.append({'id':law_name, 'text':content})
    results = rerank(query, candidates, n_results)
    new_results = []
    for item in results:
        new_item = {
            'law_name': item['id'],
            'content': item['text'],
            'score': item.get('score')  # 如果有 score 字段也保留
        }
        new_results.append(new_item)
    return new_results

def qa_query(query, n_results):
    embedding = model.encode(query).tolist()
    result1 = collection_law_qa.query(query_embeddings=embedding, n_results=n_results)
    result2 = bm25_search_law_qa(query, str(PROJECT_ROOT / "bm25_index_law_qa"), top_k=n_results)
    candidates = []
    for i in range(len(result1['metadatas'][0])):
        candidates.append({'id':result1['metadatas'][0][i]['query'],'text':result1['metadatas'][0][i]['answer']})
    for law_question, answer, score in result2:
        candidates.append({'id':law_question, 'text':answer})
    results = rerank(query, candidates, n_results)
    new_results = []
    for item in results:
        new_item = {
            'law_question': item['id'],
            'answer': item['text'],
            'score': item.get('score')  # 如果有 score 字段也保留
        }
        new_results.append(new_item)
    return new_results

if __name__ == "__main__":
    #join_corpus_to_db()
    #join_qa_to_db()
    results = corpus_query("中华人民共和国民法典", 3)
    print(results)
    for out in results:
        print(out['law_name'], out['content'], out['score'])
    results = qa_query("开发商延期交房违约金怎么计算？", 3)
    print(results)
    for out in results:
        print(out['law_question'], out['answer'], out['score'])

