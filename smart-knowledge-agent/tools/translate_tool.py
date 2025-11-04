# tools/translate_tool.py
from typing import Literal
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


def translate(
    text: str,
    target_lang: Literal["en", "zh"] = "zh",
) :
    """用 OpenAI 模型做中英互译"""
    if not text:
        return "[translate] 请输入要翻译的文本。"

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)

    # 这里是关键：不要再用 {{ }} 了，直接用正常的 f-string
    lang_name = "中文" if target_lang == "zh" else "英文"
    sys_prompt = (
        f"你是一个翻译助手。请把用户的内容准确翻译成{lang_name}，"
        "不要解释，不要添加多余内容，保留专有名词。"
    )

    resp = llm.invoke(
        [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=text),
        ]
    )

    # langchain 的 ChatOpenAI 返回的是 AIMessage，有 .content
    return resp.content

