# Smart Knowledge Agent

一个结合 **LangChain + RAG + Tool Calling + 多文档检索** 的智能知识助手。支持上传 PDF/Markdown/网页内容，构建向量检索库，进行问答、总结与多工具调用，并提供简单的 Web UI（Streamlit）。

## 快速开始
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # 然后填 OPENAI_API_KEY
streamlit run ui/streamlit_app.py
```

## 目录结构
```
smart-knowledge-agent/
├── main.py
├── requirements.txt
├── .env.example
├── tools/
├── retriever/
├── memory/
└── ui/
```
