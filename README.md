# 法律领域 RAG 增强智能问答 Agent 

面向法律咨询与材料分析场景的 Agentic RAG 系统。系统基于本地法律条文库和法律问答案例库，支持用户用自然语言提问，并通过 Agent 工具链完成法条检索、案例检索、材料读取、报告 skill 读取和 Markdown 报告生成。

## 核心能力

- 法律条文库与法律问答案例库检索。
- BM25 + 向量检索混合召回。
- Cross-Encoder 重排序，可通过环境变量开启。
- 基于 LangChain `create_agent` 的工具调用 Agent。
- 支持读取 `materials` 目录中的 `md`、`txt`、`docx`、文字型 `pdf`。
- 支持读取 `skills` 中的报告/文书生成 skill。
- 支持将 Markdown 报告保存到 `output` 目录。
- 支持检索评测 Hit@K 和生成质量 LLM-as-Judge 评测。

## 项目结构

```text
src/
  agent.py        Agent 启动入口
  prompt.py       系统提示词
  tools.py        Agent 可调用工具

rag/
  Rag.py          向量检索、BM25 混合召回、重排序
  BM25.py         Whoosh + jieba BM25 检索

data/
  legal_statutes.jsonl    法律条文数据
  legal_qa_cases.jsonl    法律问答案例数据

materials/        用户材料示例
skills/           报告和文书生成 skill
judge-agent/      评测脚本、测试集和评测结果
```

## 运行准备

建议使用 Python 3.10+。

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

复制环境变量模板：

```powershell
copy .env.example .env
```

然后在 `.env` 中填写：

```text
DEEPSEEK_API_KEY=你的 DeepSeek API Key
```

默认模型：

```text
DEEPSEEK_MODEL=deepseek-v4-pro
```

如果本机没有 NVIDIA CUDA，建议保持：

```text
RAG_DEVICE=cpu
RAG_ENABLE_RERANK=0
```

## 启动 Agent

```powershell
.\.venv\Scripts\python.exe src\agent.py
```

启动后在命令行输入问题，例如：

```text
query: 试用期辞职需要提前几天？
```

## 材料分析

把法律材料放入 `materials` 目录，支持：

```text
.md
.txt
.docx
文字型 .pdf
```

示例问题：

```text
请读取 materials 里的合同材料，并帮我分析风险。
```

## 报告生成

系统会从 `skills/skills_detail.md` 读取当前支持的报告或文书类型，再按需读取具体 skill 文件。

示例问题：

```text
请根据材料生成一份合同审查报告，并保存为 Markdown 文件。
```

生成结果会保存到 `output` 目录。

## 评测

检索评测：

```powershell
.\.venv\Scripts\python.exe judge-agent\judge.py --mode retrieval
```

生成质量评测：

```powershell
.\.venv\Scripts\python.exe judge-agent\judge.py --mode generation
```

当前 120 条测试集结果：

```text
Hit@1 = 0.9167
Hit@3 = 0.9750
Hit@5 = 0.9833

query_relevance = 97.44
reference_faithfulness = 96.62
answer_quality = 96.51
```

## 当前限制

- 仅支持文本型法律材料，不支持图片 OCR 和扫描件 PDF。
- 本项目仅用于学习与研究，不构成正式法律意见。
- 首次运行检索时会加载本地 embedding 模型，CPU 环境下可能较慢。
- 如果启用 Cross-Encoder 重排序，准确性可能提升，但 CPU 延迟也会明显增加。
