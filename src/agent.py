import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import (
    list_material_files,
    list_report_skills,
    read_legal_material,
    read_report_skill,
    save_markdown_report,
    search_legal_QA,
    search_legal_statutes,
)
from dotenv import load_dotenv
from langchain.agents.middleware import ToolCallLimitMiddleware
from prompt import system_prompt
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class Agent:
    def __init__(self):
        # ==================== Agent 配置 ====================
        model = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            openai_api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0,
            max_tokens=2048,
            timeout=120,
            max_retries=3,
        )

        middleware = [
            ToolCallLimitMiddleware(
                run_limit=10,
                exit_behavior="continue"
            )
        ]

        checkpointer = MemorySaver()

        self.Agent = create_agent(
            model=model,
            tools=[
                search_legal_statutes,
                search_legal_QA,
                list_material_files,
                read_legal_material,
                list_report_skills,
                read_report_skill,
                save_markdown_report,
            ],
            system_prompt=system_prompt,
            middleware=middleware,
            checkpointer=checkpointer,   # 短期记忆
        )
    def invoke(self, question):
        run_config = {
            "configurable": {
                "thread_id": "test_session_001",
                "user_id": "user_123"
            }
        }
        response = self.Agent.invoke(
            {"messages": [("user", question)]},  # 修正消息格式
            config=run_config,
            recursion_limit=70  # 防止步数超限
        )
        return response

if __name__ == "__main__":
    # 必须传入 thread_id（对话线程）和 user_id（用户标识）
    agent = Agent()
    while True:
        try:
            question = input("query: ")
        except EOFError:
            break
        if not question.strip():
            continue
        response = agent.invoke(question)
        print(response['messages'][-1].content)
