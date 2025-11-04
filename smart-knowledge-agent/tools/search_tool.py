# tools/search_tool.py
from ddgs import DDGS 
from tools.translate_tool import translate


def _maybe_to_english(query: str) -> str:
    has_cn = any('\u4e00' <= ch <= '\u9fff' for ch in query)
    if has_cn:
        en = translate(query, target_lang="en")
        print(f"[RAG DEBUG] 检测到中文查询，已翻译用于搜索：{en}")
        return en
    return query


def search(query: str, max_results: int = 5) -> str:
    q = _maybe_to_english(query)
    texts = []

    # 第一次尝试：正常的 text 搜索
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=max_results, region="wt-wt", safesearch="moderate"):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                texts.append(f"🔹 {title}\n{body}\n🔗 {href}\n")
    except Exception as e:
        print("[RAG DEBUG] 第一次搜索失败：", e)

    # 如果还没有结果，再用 html 后端再抓一遍（有些地区这个反而能拿到）
    if not texts:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=max_results, backend="html"):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    href = r.get("href", "")
                    texts.append(f"🔹 {title}\n{body}\n🔗 {href}\n")
        except Exception as e:
            print("[RAG DEBUG] html 后端也失败：", e)

    # 还是空，就返回一个占位符，别传空串
    if not texts:
        return f"【联网搜索为空：{q}】"

    return "\n".join(texts)
