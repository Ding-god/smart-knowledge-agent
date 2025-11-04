import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from memory.vector_store import VectorStore
from retriever.pdf_loader import load_pdf_dir
from retriever.md_loader import load_md_dir
from tools.search_tool import search
from tools.translate_tool import translate
from tools.calc_tool import safe_eval

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

class SmartKnowledgeAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
        self.embeddings = OpenAIEmbeddings()
        self.vs = VectorStore(self.embeddings)

         # 👇 关键：启动就尝试把本地向量库读回来
        try:
            self.vs.load()
            print("[DEBUG] 已加载本地向量库。")
        except Exception as e:
            print("[DEBUG] 暂无已有向量库，将在下一次 ingest 时创建：", e)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". "]
        )

    def ingest(self, paths):
        docs = []
        for p in paths:
            if os.path.isdir(p):
                docs += load_pdf_dir(p)
                docs += load_md_dir(p)
            elif p.lower().endswith(".pdf"):
                docs += load_pdf_dir(os.path.dirname(p) or ".", only_file=os.path.basename(p))
            elif p.lower().endswith((".md", ".txt")):
                docs += load_md_dir(os.path.dirname(p) or ".", only_file=os.path.basename(p))
        if not docs:
            print("[Ingest] 未找到可用文档")
            return
        chunks = self.splitter.split_documents([Document(page_content=d.page_content, metadata=d.metadata) for d in docs])
        self.vs.build(chunks)
        self.vs.save("./memory")
        print(f"[DEBUG] 收到路径: {paths}")
        print(f"[DEBUG] 读取文档数: {len(docs)}")
        print(f"[DEBUG] 拆分块数: {len(chunks)}")

        print(f"[Ingest] 已构建向量库：{len(chunks)} chunks")

    def rag_answer(self, query: str, k: int = 5):
        # 1. 检索
        refs = self.vs.search(query, k=k)

        # 2. 打印一下检索结果，方便调试
        print("[DEBUG] 检索到的文档片段：")
        for i, r in enumerate(refs):
            print(f"  - Top-{i+1} 来源: {r.metadata.get('source','unknown')}")
            print(f"    内容前60字: {r.page_content[:60]!r}")

        # 3. 拼上下文
        context = "\n\n".join([r.page_content for r in refs])

        # 4. 调用 LLM
        sys_prompt = (
            "你是一个严谨的知识助手。请主要依据“上下文”来回答；"
            "如果上下文确实不足，就说明不足，不要胡编。"
            "最后请输出一个“参考来源”列表。"
        )
        user_prompt = f"问题：{query}\n\n上下文：\n{context}"
        resp = self.llm.invoke([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ])

        # 5. 把引用拼到最后
        citations_lines = []
        for r in refs:
            src = r.metadata.get("source", "unknown")
            citations_lines.append(f"- {src}")
        citations = "\n".join(citations_lines) if citations_lines else "- （无检索结果）"

        final_answer = resp.content + "\n\n参考来源：\n" + citations
        return final_answer

        # >>> 新增：智能回答（本地优先 + 不足再联网）
    def smart_answer(self, query: str, k: int = 5) -> str:
    # """
    # 智能版问答（带调试）：
    # 1. 先从本地向量库里查 k 段
    # 2. 看本地内容够不够（长度 + 数量）
    # 3. 不够就去联网搜索
    # 4. 最后把本地 + 联网 一起喂给 llm
    # """
        DEBUG_RAG = True  # 打开/关闭调试输出
        if DEBUG_RAG:
            print("\n" + "=" * 80)
            print(f"[RAG DEBUG] 用户问题：{query}")

        # ① 先查本地
        refs = self.vs.search(query, k=k)   # 这里就是你的向量库检索
        local_texts = []   # 只在下面循环里构建，避免重复
        

        # 打印本地检索详情
        if DEBUG_RAG:
            print(f"[RAG DEBUG] 本地检索返回 {len(refs)} 段：")
        for i, r in enumerate(refs, start=1):
            src = r.metadata.get("source") if hasattr(r, "metadata") else f"样本_{i}"
            chunk_text = f"[{i}] 来源: {src}\n{r.page_content}"
            local_texts.append(chunk_text)
            if DEBUG_RAG:
                preview = r.page_content.replace("\n", " ")[:120]
                print(f"  [{i}] 来源: {src}")
                print(f"      内容: {preview}...")
        # ② 判断本地是否足够（新规则）
        # >>> 修改开始：新的联网判定逻辑（相似度阈值 + 占比 + 关键词覆盖）

        import os, re
        from statistics import median

        # 可通过环境变量调参；没有就用默认值
        SIM_TAU   = float(os.getenv("RAG_SIM_TAU", "0.52"))  # 单段“靠谱”的相似度阈值（≥ 认为相关）
        MIN_GOOD  = int(os.getenv("RAG_MIN_GOOD", "3"))      # 至少需要这么多“靠谱段”
        MIN_PROP  = float(os.getenv("RAG_MIN_PROP", "0.40")) # 靠谱段占比下限

        def _cn_keywords(q: str, topn: int = 6):
            # 从中文问题里提取长度≥2的词片段；英文问题可以自己扩展
            return re.findall(r"[\u4e00-\u9fa5]{2,}", q)[:topn]

        # 取检索分数（不同后端有的放在 r.score，有的放在 r.metadata["score"]，都兼容）
        scores = []
        for r in refs:
            s = getattr(r, "score", None)
            if s is None and hasattr(r, "metadata"):
                s = r.metadata.get("score")
            try:
                s = float(s) if s is not None else 0.0
            except Exception:
                s = 0.0
            scores.append(s)

        good_idx = [i for i, s in enumerate(scores) if s >= SIM_TAU]
        prop = len(good_idx) / max(len(refs), 1)

        # 关键词覆盖：前 8 段里至少出现过一个关键词，避免被“AI/文化”这类泛段误判为相关
        kws = _cn_keywords(query)
        kw_hits = sum(
            1 for r in refs[:min(8, len(refs))]
            if (kws and any(kw in r.page_content for kw in kws))
        )

        # 综合判定：满足“靠谱段数量 + 占比 + 关键词覆盖”才认为本地足够
        has_enough_local = (
            (len(good_idx) >= MIN_GOOD) and
            (prop >= MIN_PROP) and
            (kw_hits > 0 or not kws)     # 没有中文关键词时放宽这一条
        )

        if DEBUG_RAG:
            max_score = max(scores) if scores else 0.0
            med_score = median(scores) if scores else 0.0
            print(f"[RAG DEBUG] 本地检索分数: max={max_score:.3f}  median={med_score:.3f}  "
                f"good={len(good_idx)}  prop={prop:.2f}  kw_hits(top8)={kw_hits}")
            print(f"[RAG DEBUG] 判定本地是否足够: {has_enough_local}")

        # <<< 修改结束：新的联网判定逻辑


        # ③ 不够就去联网
        web_text = ""
        if not has_enough_local:
            if DEBUG_RAG:
                print("[RAG DEBUG] → 判定为本地资料不足，准备调用联网搜索 ...")
            raw_web_text = search(query)                 # 你的搜索函数
            zh = translate(raw_web_text)                 # 翻译成中文

            # 给联网结果加上“标题 + 链接 + 摘要”的壳，便于模型引用
            web_text = (
                f"### 联网检索摘要：{query}\n"
                f"链接：无（DuckDuckGo 摘要）\n"
                f"摘要：{zh}\n"
            )
            if DEBUG_RAG:
                print(f"[RAG DEBUG] 联网返回内容长度: {len(web_text)}")
        else:
            if DEBUG_RAG:
                print("[RAG DEBUG] → 本地资料已足够，本次不联网。")


        # ④ 组织最终 prompt
        context_blocks = []
        if web_text:   # 先放联网，让预览能看到
            context_blocks.append("【联网结果】\n" + web_text)
        if local_texts:
            context_blocks.append("【本地资料】\n" + "\n\n".join(local_texts))

        final_context = "\n\n".join(context_blocks) if context_blocks else "（无资料）"

        if DEBUG_RAG:
            print("[RAG DEBUG] 最终送入 LLM 的上下文（前 400 字每块）：")
            for blk in context_blocks:
                print("-" * 40)
                print(blk[:400])  # 每块各截 400 字
            print("-" * 40)


        # ⑤ 调 LLM
        sys_prompt = (
            "你是一个检索增强问答助手。\n"
            "你只能根据给你的资料作答，不要自己编资料。\n"
            "必须在回答结尾给出“参考来源”这一段，把你用到的本地/联网片段都列出来。\n"
            "如果资料确实没有，就说“资料里没有相关内容”，不要瞎猜。\n"
            # """你是检索增强助手。只允许使用我提供的【上下文】作答：
            # - 如果上下文不足以回答，请明确告诉我“未找到相关资料”；
            # - 不要凭常识或外部记忆编造答案；
            # - 如使用了资料，请在结尾列出“参考来源：<文件或链接>”。"""
        )

        user_prompt = (
            f"用户问题：{query}\n\n"
            "下面是你能用到的资料：\n"
            f"{final_context}\n\n"
            "请先给出简洁回答，然后换行写：\n"
            "参考来源：\n"
            "- 列出你上面用到的资料标题/文件名/URL能看见的那一段；\n"
            "- 如果只有一份资料，就写一个；\n"
            "- 如果完全没资料，就写“（无）”。\n"
        )

        ans = self.llm.invoke([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]).content

        if DEBUG_RAG:
            print("[RAG DEBUG] LLM 最终回答：")
            print(ans)
            print("=" * 80 + "\n")

        return ans


    def call_tool(self, name: str, **kwargs):
        if name == "search":
            return search(kwargs.get("q", ""))
        if name == "translate":
            return translate(kwargs.get("text", ""), kwargs.get("target", "zh"))
        if name == "calc":
            return safe_eval(kwargs.get("expr", ""))
        return f"未知工具：{name}"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", nargs="*", help="文档路径或目录（可多选）")
    ap.add_argument("--ask", type=str, help="问答问题", default=None)
    # >>> 新增：命令行里也可以走智能回答
    ap.add_argument("--smart-ask", type=str, help="智能问答（本地+联网）", default=None)
    ap.add_argument("--tool", type=str, help="调用工具：search/translate/calc", default=None)
    ap.add_argument("--tool-args", type=str, help="工具参数，示例：q=langchain expr=1+2*3", default=None)
    return ap.parse_args()


def parse_kv(s):
    if not s:
        return {}
    out = {}
    for kv in s.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


if __name__ == "__main__":
    args = parse_args()
    agent = SmartKnowledgeAgent()

    if args.ingest:
        agent.ingest(args.ingest)

    if args.ask:
        print("\n[RAG Answer]\n" + agent.rag_answer(args.ask))

    # >>> 新增：命令行测试智能问答
    if args.smart_ask:
        print("\n[Smart Answer]\n" + agent.smart_answer(args.smart_ask))

    if args.tool:
        kwargs = parse_kv(args.tool_args)
        print("\n[Tool Result]\n", agent.call_tool(args.tool, **kwargs))
