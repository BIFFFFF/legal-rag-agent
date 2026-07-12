import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


JUDGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = JUDGE_DIR.parent

DEFAULT_RETRIEVAL_SET = JUDGE_DIR / "retrieval_testset.jsonl"
DEFAULT_GENERATION_SET = JUDGE_DIR / "generation_testset.jsonl"
DEFAULT_RETRIEVAL_OUTPUT = JUDGE_DIR / "retrieval_scores.json"
DEFAULT_GENERATION_OUTPUT = JUDGE_DIR / "judge_scores.jsonl"
DEFAULT_GENERATION_SUMMARY = JUDGE_DIR / "judge_summary.json"


load_dotenv(PROJECT_ROOT / ".env")
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(PROJECT_ROOT / ".hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def read_jsonl(path: Path) -> List[Dict]:
    """读取 jsonl 测试集。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return rows


def write_json(path: Path, data: Dict) -> None:
    """写入格式化后的 JSON 文件。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    """写入 jsonl 文件。"""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def result_to_text(result: Dict) -> str:
    """把检索结果转成便于关键词匹配的文本。"""
    return json.dumps(result, ensure_ascii=False)


def is_hit(results: List[Dict], expected_keywords: List[str], k: int) -> bool:
    """判断前 k 条检索结果中是否包含所有期望关键词。"""
    joined = "\n".join(result_to_text(item) for item in results[:k])
    return all(keyword in joined for keyword in expected_keywords)


def evaluate_retrieval(
    testset_path: Path = DEFAULT_RETRIEVAL_SET,
    output_path: Path = DEFAULT_RETRIEVAL_OUTPUT,
    max_k: int = 5,
) -> Dict:
    """
    评估检索命中率 Hit@K。

    retrieval_testset.jsonl 每行字段：
    - query：用户问题
    - source_type：statute 表示法条检索，qa 表示问答案例检索
    - expected_keywords：期望在检索结果中出现的关键词列表
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from rag.Rag import corpus_query, qa_query

    samples = read_jsonl(testset_path)
    counters = {1: 0, 3: 0, 5: 0}
    details = []
    top_k = max(max_k, 5)

    for index, sample in enumerate(samples, start=1):
        query = sample["query"]
        source_type = sample.get("source_type", "statute")
        expected_keywords = sample.get("expected_keywords", [])

        if source_type == "qa":
            results = qa_query(query, top_k)
        else:
            results = corpus_query(query, top_k)

        hit_map = {f"hit@{k}": is_hit(results, expected_keywords, k) for k in counters}
        for k in counters:
            counters[k] += int(hit_map[f"hit@{k}"])

        details.append(
            {
                "id": index,
                "query": query,
                "source_type": source_type,
                "expected_keywords": expected_keywords,
                **hit_map,
                "top_results": results[:5],
            }
        )
        print(f"[检索评测] {index}/{len(samples)} {query} {hit_map}")

    summary = {
        "total": len(samples),
        "hit@1": round(counters[1] / len(samples), 4) if samples else 0,
        "hit@3": round(counters[3] / len(samples), 4) if samples else 0,
        "hit@5": round(counters[5] / len(samples), 4) if samples else 0,
        "details": details,
    }
    write_json(output_path, summary)
    print(f"检索评测完成，结果已保存：{output_path}")
    return summary


class JudgeAgent:
    """用大模型对 Agent 回答做三维度评分。"""

    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，请先在 .env 中配置。")

        self.llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0,
        )

    def score_triplet(self, query: str, reference: str, answer: str) -> Dict:
        system_prompt = """
你是法律领域 Legal Agent 的评测员。请只根据用户问题、检索依据和 Agent 回答进行评分。

评分维度：
1. query_relevance：回答是否回应了用户问题。
2. reference_faithfulness：回答是否忠实于检索依据，不能编造检索依据里没有的法律结论。
3. answer_quality：回答是否完整、清晰、可操作，是否给出风险提示或下一步建议。

每个维度 0-100 分：
- 90-100：非常好，基本没有明显问题。
- 70-89：可用，但存在轻微缺漏。
- 50-69：部分可用，但问题明显。
- 0-49：严重偏题、无依据或不可用。

必须返回 JSON，不要输出任何额外文字。格式：
{
  "query_relevance": 0,
  "reference_faithfulness": 0,
  "answer_quality": 0,
  "comment": "一句话说明主要扣分点"
}
"""
        user_prompt = f"""
用户问题：
{query}

检索依据：
{reference}

Agent 回答：
{answer}
"""
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return parse_score_json(response.content)


def parse_score_json(raw_text: str) -> Dict:
    """解析大模型返回的评分 JSON，并做兜底清洗。"""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    return {
        "query_relevance": clamp_score(data.get("query_relevance", 0)),
        "reference_faithfulness": clamp_score(data.get("reference_faithfulness", 0)),
        "answer_quality": clamp_score(data.get("answer_quality", 0)),
        "comment": str(data.get("comment", "")),
    }


def clamp_score(value) -> int:
    """把评分限制在 0 到 100 之间。"""
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def evaluate_generation(
    testset_path: Path = DEFAULT_GENERATION_SET,
    output_path: Path = DEFAULT_GENERATION_OUTPUT,
    summary_path: Path = DEFAULT_GENERATION_SUMMARY,
) -> Dict:
    """
    使用 LLM-as-Judge 评估生成质量。

    generation_testset.jsonl 每行字段：
    - query：用户问题
    - reference：检索依据或参考材料
    - answer：Agent 生成答案
    """
    samples = read_jsonl(testset_path)
    judge = JudgeAgent()
    rows = []

    for index, sample in enumerate(samples, start=1):
        query = sample.get("query", "")
        reference = sample.get("reference", "")
        answer = sample.get("answer", "")
        scores = judge.score_triplet(query, reference, answer)

        row = {
            "id": index,
            "query": query,
            "scores": scores,
        }
        rows.append(row)
        print(f"[生成评测] {index}/{len(samples)} {scores}")

    write_jsonl(output_path, rows)
    summary = summarize_generation_scores(rows)
    write_json(summary_path, summary)
    print(f"生成评测明细已保存：{output_path}")
    print(f"生成评测汇总已保存：{summary_path}")
    return summary


def summarize_generation_scores(rows: List[Dict]) -> Dict:
    """汇总三维评分平均值。"""
    if not rows:
        return {
            "total": 0,
            "query_relevance": 0,
            "reference_faithfulness": 0,
            "answer_quality": 0,
        }

    return {
        "total": len(rows),
        "query_relevance": round(mean(row["scores"]["query_relevance"] for row in rows), 2),
        "reference_faithfulness": round(mean(row["scores"]["reference_faithfulness"] for row in rows), 2),
        "answer_quality": round(mean(row["scores"]["answer_quality"] for row in rows), 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legal Agent 评测脚本")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "generation", "all"],
        default="retrieval",
        help="retrieval 评估 Hit@5；generation 评估生成质量；all 依次执行两者。",
    )
    parser.add_argument("--retrieval-set", default=str(DEFAULT_RETRIEVAL_SET), help="检索评测集路径")
    parser.add_argument("--generation-set", default=str(DEFAULT_GENERATION_SET), help="生成评测集路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"retrieval", "all"}:
        evaluate_retrieval(Path(args.retrieval_set))
    if args.mode in {"generation", "all"}:
        evaluate_generation(Path(args.generation_set))


if __name__ == "__main__":
    main()
