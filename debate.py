import operator
import json
import logging
from typing import TypedDict, Annotated, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define output schema for debaters
class DebaterOutput(BaseModel):
    argument: str = Field(description="提出的論點 (Argument)")
    emotion_tone: str = Field(description="情緒語氣 (Emotional tone)")
    attack_point: str = Field(description="攻擊點或是反駁的論點 (Attack points or refutations)")

class DebateState(TypedDict):
    topic: str
    messages: Annotated[List[BaseMessage], operator.add]
    round_count: int
    max_rounds: int
    current_speaker: str
    final_report: str

import os
from dotenv import load_dotenv

load_dotenv()

# Config & Model
MODEL_NAME = os.getenv("MODEL_NAME")
llm = ChatOllama(
    model=MODEL_NAME, 
    temperature=0.7, 
    base_url="http://127.0.0.1:11434",
    keep_alive=-1      
)

# ==========================================
# Prompt 
# ==========================================

PRO_SYSTEM = """你是一名人 AI 辯論家 (正方)。
你的性格：邏輯嚴密、數據導向、言辭具備攻擊性。
你的任務：提出支持論點並強烈質疑對方的邏輯漏洞。"""

PRO_USER = """目前辯論主題是：{topic}
這是第 {round_count} / {max_rounds} 輪。

【過去的辯論紀錄】
---
{debate_history}
---

現在輪到你了！請直接以純文字發言，不要使用 JSON，並必定包含以下三個段落：
[情緒與語氣]：(描述你現在的語氣)
[我的論點]：(你的主要論點)
[攻擊與反駁]：(如果是第一輪開場請強調立論，否則請猛烈攻擊對方的具體點)
"""

CON_SYSTEM = """你是一名人 AI 辯論家 (反方)。
你的性格：充滿同理心、從倫理或不可預見風險角度出發。
你的任務：反駁正方論點，並提出對立面的證據或顧慮。"""

CON_USER = """目前辯論主題是：{topic}
這是第 {round_count} / {max_rounds} 輪。

【過去的辯論紀錄】
---
{debate_history}
---

現在輪到你了！請直接以純文字發言，不要使用 JSON，並必定包含以下三個段落：
[情緒與語氣]：(描述你現在的語氣)
[我的論點]：(你的主要論點)
[攻擊與反駁]：(請針對上方正方的發言進行防禦與反駁)
"""

JUDGE_SYSTEM = "你是一場 AI 辯論會的裁判與主持人。"
JUDGE_USER = """辯論主題是：{topic}
辯論已經結束。請你根據雙方的發言，對他們的表現進行總結與評分。

【辯論全紀錄】
---
{debate_history}
---

評分標準包含：1. 論點強度 2. 修辭技巧 3. 反應速度
請詳細寫出你的分析報告，並宣佈勝負結果。"""

def get_history_text(messages: List[BaseMessage]) -> str:
    if not messages:
        return "無 (這是第一輪開場)"
    return "\n\n".join([m.content for m in messages])

def stream_and_collect(chain, inputs, prefix=""):
    print(f"\n{prefix}")
    print("-" * 50)
    full_response = ""
    try:
        # Stream token by token to console
        for chunk in chain.stream(inputs):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += content
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        
    print("\n" + "-" * 50)
    
    if not full_response.strip():
        logger.warning(f"偵測到模型 {prefix} 回傳空白！嘗試使用非 stream 的 invoke 重試...")
        try:
            fallback = chain.invoke(inputs)
            full_response = fallback.content
            print(f"(Retry) {full_response}")
            print("-" * 50)
        except Exception as retry_e:
            logger.error(f"Invoke retry failed: {retry_e}")
            
    return full_response

def node_pro(state: DebateState):
    logger.info(f"進入 node_pro (第 {state['round_count']} 輪)")
    prompt = ChatPromptTemplate.from_messages([
        ("system", PRO_SYSTEM),
        ("human", PRO_USER)
    ])
    chain = prompt | llm
    
    history = get_history_text(state["messages"])
    inputs = {
        "topic": state["topic"],
        "round_count": state["round_count"],
        "max_rounds": state["max_rounds"],
        "debate_history": history
    }
    
    full_text = stream_and_collect(chain, inputs, prefix=f"=== 第 {state['round_count']} 輪：[正方] 發言 ===")
    msg = AIMessage(content=f"[正方]:\n{full_text}")
    return {"messages": [msg], "current_speaker": "pro"}

def node_con(state: DebateState):
    logger.info(f"進入 node_con (第 {state['round_count']} 輪)")
    prompt = ChatPromptTemplate.from_messages([
        ("system", CON_SYSTEM),
        ("human", CON_USER)
    ])
    chain = prompt | llm
    
    history = get_history_text(state["messages"])
    inputs = {
        "topic": state["topic"],
        "round_count": state["round_count"],
        "max_rounds": state["max_rounds"],
        "debate_history": history
    }
    
    full_text = stream_and_collect(chain, inputs, prefix=f"=== 第 {state['round_count']} 輪：[反方] 發言 ===")
    msg = AIMessage(content=f"[反方]:\n{full_text}")
    return {"messages": [msg], "current_speaker": "con", "round_count": state["round_count"] + 1}

def node_judge(state: DebateState):
    logger.info("進入 node_judge")
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM),
        ("human", JUDGE_USER)
    ])
    chain = prompt | llm
    
    history = get_history_text(state["messages"])
    inputs = {"topic": state["topic"], "debate_history": history}
    
    full_text = stream_and_collect(chain, inputs, prefix="=== [裁判] 總結與裁決 ===")
    return {"final_report": full_text, "current_speaker": "judge"}

def router(state: DebateState) -> Literal["node_pro", "node_judge"]:
    if state["round_count"] > state["max_rounds"]:
        return "node_judge"
    else:
        return "node_pro"

builder = StateGraph(DebateState)

builder.add_node("node_pro", node_pro)
builder.add_node("node_con", node_con)
builder.add_node("node_judge", node_judge)

builder.add_edge(START, "node_pro")
builder.add_edge("node_pro", "node_con")
builder.add_conditional_edges("node_con", router)
builder.add_edge("node_judge", END)

graph = builder.compile()

if __name__ == "__main__":
    TOPIC = os.getenv("TOPIC")
    max_rounds = 3
    print(f"=== 辯論開始 ===")
    print(f"主題：{TOPIC}")
    print(f"總輪數：{max_rounds}\n")

    initial_state = {
        "topic": TOPIC,
        "messages": [],
        "round_count": 1,
        "max_rounds": max_rounds,
        "current_speaker": "init",
        "final_report": ""
    }

    for s in graph.stream(initial_state):
        pass
