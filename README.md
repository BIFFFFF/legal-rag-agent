# 法律领域 RAG 增强智能问答 Agent

面向法律咨询与法律材料分析场景的 Agentic RAG 系统。系统基于本地法律条文库和法律问答案例库，支持用户通过自然语言提问，并通过 Agent 工具链完成法条检索、案例检索、材料读取、报告 Skill 读取与 Markdown 报告生成。

项目重点放在法律知识库构建、混合检索、Agent 工具调用、可溯源回答和评测体系设计上，用于验证 RAG 在法律问答场景中降低幻觉、提升回答依据性的效果。

## 核心能力

- 法律条文库与法律问答案例库检索。
- 法律条文采用“条款名称、完整内容、内容片段”的多粒度索引方式。
- BM25 关键词检索与向量检索混合召回。
- Cross-Encoder 重排序，可通过环境变量按需开启。
- 基于 LangChain `create_agent` 构建工具调用型 Agent。
- 支持读取 `materials` 目录中的 `md`、`txt`、`docx`、文字型 `pdf` 材料。
- 支持读取 `skills` 目录中的报告和文书生成 Skill。
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
skills/           报告和文书生成 Skill
judge-agent/      评测脚本、测试集和评测结果
output/           示例 Markdown 报告输出
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

系统会先从 `skills/skills_detail.md` 读取当前支持的报告或文书类型，再按需读取具体 Skill 文件。

示例问题：

```text
请根据材料生成一份合同审查报告，并保存为 Markdown 文件。
```

生成结果会保存到 `output` 目录。仓库中保留了少量示例 Markdown 报告，便于查看报告生成效果。

## 评测

检索评测：

```powershell
.\.venv\Scripts\python.exe judge-agent\judge.py --mode retrieval
```

生成质量评测：

```powershell
.\.venv\Scripts\python.exe judge-agent\judge.py --mode generation
```

当前检索评测集共 280 条，覆盖法条检索、问答案例检索、标准化问题和用户自然表达问题。当前整体检索结果：

```text
Hit@1 = 0.7607
Hit@3 = 0.8750
Hit@5 = 0.8893
```

生成质量评测集共 120 条，使用 LLM-as-Judge 从三个维度评分：

```text
query_relevance = 97.44
reference_faithfulness = 96.62
answer_quality = 96.51
```

其中：

- `query_relevance`：回答是否回应用户问题。
- `reference_faithfulness`：回答是否忠实于检索依据。
- `answer_quality`：回答是否完整、清晰、可操作。

## 当前限制

- 仅支持文本型法律材料，不支持图片 OCR 和扫描件 PDF。
- 本项目用于学习与研究，不构成正式法律意见。
- 首次运行检索时会加载本地 embedding 模型，CPU 环境下可能较慢。
- 如果启用 Cross-Encoder 重排序，准确性可能提升，但 CPU 延迟也会明显增加。
