import os
import json
from pathlib import Path
from whoosh.fields import Schema, ID, TEXT
from whoosh.index import create_in, open_dir
from whoosh.analysis import Analyzer, Token
from whoosh.qparser import QueryParser
from whoosh.index import open_dir
import jieba
import re
import sys

def clean_query_string(query):
    """
    清洗查询字符串，移除可能干扰 Whoosh 解析的特殊字符
    """
    # 保留：中文、字母、数字、空格、常见标点（句号、逗号、问号等）
    # 移除：% & | ( ) = > < * ? ~ ^ 等特殊符号
    query = re.sub(r'[%&|()=<>*?~^]', ' ', query)
    # 将多个空格合并为一个
    query = re.sub(r'\s+', ' ', query)
    # 去除首尾空格
    query = query.strip()
    return query

# 自定义中文分词器（使用 jieba）
class ChineseAnalyzer(Analyzer):
    """
    基于 jieba 的 Whoosh 中文分词器
    """
    def __call__(self, value, **kwargs):
        # 使用 jieba 搜索引擎模式分词
        words = jieba.lcut_for_search(value)
        # 为每个词生成 Token，并添加必需的 pos 属性（位置索引）
        for idx, word in enumerate(words):
            token = Token(text=word,
                          original=word,
                          pos=idx,           # 必须提供位置信息
                          startchar=None,
                          endchar=None,
                          stopped=False)
            yield token

    def __eq__(self, other):
        return isinstance(other, ChineseAnalyzer)

    def __repr__(self):
        return "ChineseAnalyzer()"

def _prepare_legacy_analyzer_pickle():
    # 旧索引是直接运行脚本构建的，Whoosh 反序列化时可能会查找
    # "__main__.ChineseAnalyzer"，这里补一个兼容映射。
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "ChineseAnalyzer"):
        setattr(main_module, "ChineseAnalyzer", ChineseAnalyzer)

def build_law_bm25_index(jsonl_path, index_dir):
    """
    jsonl_path: 原始数据文件路径（每行一个 JSON 对象）
    index_dir:  存放 Whoosh 索引的目录
    text_field: 要建立 BM25 索引的字段名（如 "content" 或 "name"）
    """
    # 定义 Schema：ID 作为唯一标识，text 字段使用中文分词
    schema = Schema(
        law_name=ID(stored=True, unique=True),
        content=TEXT(analyzer=ChineseAnalyzer(), stored=True)
    )

    # 创建索引目录
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # 如果索引已存在则打开，否则创建
    if not os.path.exists(os.path.join(index_dir, "MAIN_WRITELOCK")):
        ix = create_in(index_dir, schema)
    else:
        ix = open_dir(index_dir)
    writer = ix.writer()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            law_name = item['name']  # 确保与 ChromaDB 中的 ID 一致
            text = item['name'] + ':' + item["content"]
            if not text:
                continue
            # 更新索引（如果 ID 已存在则覆盖）
            writer.update_document(law_name=law_name, content=text)

    writer.commit()
    print(f"BM25law 索引已构建完成，共 {ix.doc_count()} 条记录，保存在 {index_dir}")

def build_law_qa_bm25_index(jsonl_path, index_dir):
    """
    jsonl_path: 原始数据文件路径（每行一个 JSON 对象）
    index_dir:  存放 Whoosh 索引的目录
    """
    # 定义 Schema：ID 作为唯一标识，text 字段使用中文分词
    schema = Schema(
        law_question=TEXT(analyzer=ChineseAnalyzer(), stored=True),
        answer=ID(stored=True, unique=True)
    )

    # 创建索引目录
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # 如果索引已存在则打开，否则创建
    if not os.path.exists(os.path.join(index_dir, "MAIN_WRITELOCK")):
        ix = create_in(index_dir, schema)
    else:
        ix = open_dir(index_dir)
    writer = ix.writer()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            law_question = item['input']  # 确保与 ChromaDB 中的 ID 一致
            text = item['output']
            if not text:
                continue
            # 更新索引（如果 ID 已存在则覆盖）
            writer.update_document(law_question=law_question, answer=text)

    writer.commit()
    print(f"BM25lawqa 索引已构建完成，共 {ix.doc_count()} 条记录，保存在 {index_dir}")

def bm25_search_law(query, index_dir, top_k=10):
    """
    使用 BM25 检索
    query: 用户查询字符串（中文）
    index_dir: 索引目录
    top_k: 返回前 k 个文档 ID
    返回: 列表，每个元素为 (law_name, content, score)
    """
    _prepare_legacy_analyzer_pickle()
    query = clean_query_string(query)
    ix = open_dir(index_dir)
    with ix.searcher() as searcher:
        # 使用默认的 BM25 相似度（Whoosh 内置）
        parser = QueryParser("content", ix.schema)
        q = parser.parse(query)
        results = searcher.search(q, limit=top_k)
        hits = []
        for hit in results:
            law_name = hit["law_name"]
            content = hit["content"]
            score = hit.score
            hits.append((law_name, content, score))
        return hits
def bm25_search_law_qa(query, index_dir, top_k=10):
    """
    使用 BM25 检索
    query: 用户查询字符串（中文）
    index_dir: 索引目录
    top_k: 返回前 k 个文档 ID
    返回: 列表，每个元素为 (law_question, answer, score)
    """
    _prepare_legacy_analyzer_pickle()
    query = clean_query_string(query)
    ix = open_dir(index_dir)
    with ix.searcher() as searcher:
        # 使用默认的 BM25 相似度（Whoosh 内置）
        parser = QueryParser("law_question", ix.schema)
        q = parser.parse(query)
        results = searcher.search(q, limit=top_k)
        hits = []
        for hit in results:
            law_question = hit["law_question"]
            answer = hit["answer"]
            score = hit.score
            hits.append((law_question, answer, score))
        return hits
if __name__ == "__main__":
    print("BM25 工具模块已加载。")

