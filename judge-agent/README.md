# Judge Agent 评测说明

这个目录用于保存项目评测脚本、测试集和评测结果。

## 文件说明

- `judge.py`：评测入口脚本。
- `retrieval_testset.jsonl`：检索评测集，当前 280 条。
- `retrieval_scores.json`：检索评测结果。
- `generation_testset.jsonl`：生成质量评测集，当前 120 条。
- `judge_scores.jsonl`：生成质量评分明细。
- `judge_summary.json`：生成质量评分汇总。

## 检索评测

检索评测用于计算 Hit@K，默认使用 `retrieval_testset.jsonl`。

每条样例包含：

- `query`：测试问题。
- `source_type`：`statute` 表示法条检索，`qa` 表示问答案例检索。
- `expected_keywords`：期望在检索结果中出现的关键词。

当前检索评测集共 280 条，覆盖法条检索、问答案例检索、标准化问题和用户自然表达问题。

运行：

```powershell
.\.venv\Scripts\python.exe judge-agent\judge.py --mode retrieval
```

当前结果：

```text
Hit@1 = 0.7607
Hit@3 = 0.8750
Hit@5 = 0.8893
```

Hit@5 的含义：对每个问题，系统返回前 5 条检索结果；如果期望关键词出现在前 5 条结果中，则该样本记为命中。

## 生成质量评测

生成质量评测使用 LLM-as-Judge，默认使用 `generation_testset.jsonl`。

每条样例包含：

- `query`：用户问题。
- `reference`：检索依据或参考材料。
- `answer`：待评测回答。

运行：

```powershell
.\.venv\Scripts\python.exe judge-agent\judge.py --mode generation
```

当前结果：

```text
query_relevance = 97.44
reference_faithfulness = 96.62
answer_quality = 96.51
```

三个维度含义：

- `query_relevance`：回答是否回应用户问题。
- `reference_faithfulness`：回答是否忠实于检索依据。
- `answer_quality`：回答是否完整、清晰、可操作。

## 同时运行两类评测

```powershell
.\.venv\Scripts\python.exe judge-agent\judge.py --mode all
```

注意：生成质量评测需要在项目根目录 `.env` 中配置 `DEEPSEEK_API_KEY`。
