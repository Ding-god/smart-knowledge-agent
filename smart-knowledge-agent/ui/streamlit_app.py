from dotenv import load_dotenv
# 加载 .env 文件
load_dotenv()
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
import main as core
from main import SmartKnowledgeAgent

# ============ 页面配置 ============
st.set_page_config(
    page_title="Smart Knowledge Agent 🧠",
    page_icon="✨",
    layout="wide"
)

# ============ 自定义样式 ============
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #E8E8E8;
    font-family: 'Segoe UI', sans-serif;
}
.chat-container {
    width: 70%;
    margin: auto;
    padding: 25px;
}
.user-bubble {
    background: linear-gradient(90deg, #2E8BFD, #1453D1);
    padding: 12px 18px;
    border-radius: 15px;
    color: white;
    margin: 8px 0;
    text-align: right;
    box-shadow: 0px 4px 10px rgba(46,139,253,0.2);
}
.ai-bubble {
    background: #1A1C23;
    padding: 12px 18px;
    border-radius: 15px;
    color: #E8E8E8;
    margin: 8px 0;
    text-align: left;
    border: 1px solid #2A2D35;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
}
.header-title {
    font-size: 34px;
    text-align: center;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 5px;
    background: linear-gradient(90deg, #00D4FF, #6E7CFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-header {
    text-align: center;
    font-size: 16px;
    color: #B8B8B8;
    margin-bottom: 40px;
}
</style>
""", unsafe_allow_html=True)

# ============ 标题区域 ============
st.markdown('<div class="header-title">Smart Knowledge Agent 🧠</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">📚 本地知识库 + 🌐 联网增强问答系统</div>', unsafe_allow_html=True)

# ============ 输入区 ============
query = st.text_input("💬 请输入你的问题：", placeholder="例如：强化学习的核心思想是什么？")
mode = st.radio(
    "选择回答模式：",
    ["🧩 本地知识库问答（仅查向量库）", "🌐 智能混合问答（本地优先 + 联网补全）"],
    horizontal=True
)

@st.cache_resource
def get_agent():
    return SmartKnowledgeAgent()

agent = get_agent()

if st.button("🚀 开始回答", use_container_width=True):
    if not query.strip():
        st.warning("请输入一个问题再试～")
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="user-bubble">👤 你：{query}</div>', unsafe_allow_html=True)

        import io, contextlib
        with st.spinner("AI 正在检索并组织回答中... ⏳"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                # —— 如果你的 smart_answer 目前只返回纯文本，这里先做兼容 ——
                result = None
                if "本地" in mode:
                    txt = agent.rag_answer(query)
                else:
                    # 优先尝试“结构化返回”的新版本
                    try:
                        result = agent.smart_answer(query)  # 期望是 dict
                    except TypeError:
                        # 老版本只返回字符串
                        txt = agent.smart_answer(query)

            debug_txt = buf.getvalue()

        # 统一渲染
        if isinstance(result, dict):
            txt = result.get("text", "")
            refs = result.get("sources", [])
        else:
            refs = []

        st.markdown(f'<div class="ai-bubble">🤖 智能助手：{txt}</div>', unsafe_allow_html=True)

        # if refs:
        #     st.markdown("**参考来源**")
        #     for s in refs:
        #         st.markdown(f"- {s}")

        if st.checkbox("显示调试信息"):
            st.code(debug_txt or "（无输出）", language="text")

        st.markdown('</div>', unsafe_allow_html=True)


# ============ 页脚 ============
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color: #777;">✨ 支持两种模式：纯本地检索 / 联网增强检索</p>',
    unsafe_allow_html=True
)





