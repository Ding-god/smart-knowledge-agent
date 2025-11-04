import os
import datetime
import requests
import feedparser

SAMPLES_DIR = "./samples"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_text(fname: str, text: str):
    """统一的写文件函数，放到 /samples 下"""
    fpath = os.path.join(SAMPLES_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] 写入 {fpath}")

# ----------------------------------------------------
# 1. Hugging Face Blog
# ----------------------------------------------------
def fetch_hf_blog(limit: int = 5):
    """
    抓 Hugging Face Blog 最新几篇
    策略：
      1. 先抓列表页 https://huggingface.co/blog
      2. 解析出前 limit 条的标题 + 链接
      3. 尝试逐篇请求详情页，把前 2~3 段正文抓出来
      4. 抓不到正文就写“无摘要（请点击阅读全文）”
    """
    url = "https://huggingface.co/blog"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print("[HF] 抓取失败：", e)
        return

    from bs4 import BeautifulSoup  # 需要 pip install beautifulsoup4
    soup = BeautifulSoup(resp.text, "html.parser")

    # HF 页面结构可能会变，这里做个比较宽松的选取
    articles = soup.find_all("a", href=True)
    items = []
    for a in articles:
        href = a["href"]
        # 筛一下真正的 blog 链接
        if href.startswith("/blog/"):
            title = a.get_text(strip=True)
            if not title:
                continue
            full_link = "https://huggingface.co" + href
            items.append((title, full_link))
    # 去重 & 截断
    seen = set()
    uniq = []
    for t, l in items:
        if l in seen:
            continue
        seen.add(l)
        uniq.append((t, l))
    uniq = uniq[:limit]

    today = datetime.date.today().isoformat()
    lines = [f"🤗 Hugging Face Blog 最新 {len(uniq)} 篇文章 ({today})", ""]

    # 再逐篇抓正文
    for idx, (title, link) in enumerate(uniq, start=1):
        lines.append(f"### {idx}. {title}")
        lines.append(f"链接: {link}")

        summary = "无摘要（请点击阅读全文）"
        try:
            detail = requests.get(link, headers=HEADERS, timeout=15)
            if detail.status_code == 200:
                dsoup = BeautifulSoup(detail.text, "html.parser")
                # 找正文容器，HF 经常用 <article> 或 markdown-content 之类的 class
                paras = dsoup.find_all("p")
                paras_text = [p.get_text(" ", strip=True) for p in paras if p.get_text(strip=True)]
                if paras_text:
                    # 只取前 2 段，避免文件太大
                    summary = "\n".join(paras_text[:2])
        except Exception as e:
            print(f"[HF] 抓正文失败 {link}: {e}")

        lines.append(f"摘要: {summary}")
        lines.append("")  # 空行

    save_text(f"{today}_huggingface_blog.txt", "\n".join(lines))

# ----------------------------------------------------
# 2. arXiv
# ----------------------------------------------------
def fetch_arxiv_ai(limit: int = 10):
    """
    拉 arXiv 上 cs.AI / cs.CL / cs.LG 最近的文章标题+摘要
    """
    url = (
        "http://export.arxiv.org/api/query?"
        "search_query=cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.LG&"
        f"start=0&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
    )
    feed = feedparser.parse(url)

    today = datetime.date.today().isoformat()
    lines = [f"📘 arXiv 最新 {limit} 篇 AI/ML 论文 ({today})", ""]

    for entry in feed.entries:
        title = entry.title
        summary = entry.summary
        link = entry.link
        lines.append(f"### {title}")
        lines.append(f"链接: {link}")
        lines.append(f"摘要: {summary}")
        lines.append("")

    save_text(f"{today}_arxiv_ai_trends.txt", "\n".join(lines))


# ----------------------------------------------------
# 4. Meta AI Blog
# ----------------------------------------------------
def fetch_meta_ai_blog(limit: int = 5):
    """
    Meta AI Blog 在你这里 403，我们先写个兜底的，
    至少让你的本地知识库里有“Meta AI Blog”这个名词，方便检索。
    """
    today = datetime.date.today().isoformat()
    lines = [
        f"🧠 Meta AI Blog 最新 {limit} 篇（占位数据，因为 https://ai.meta.com/blog/ 返回 403）",
        "",
        "这个站点当前不能用 requests 直接抓，我们先把官方博客入口写进去，",
        "等能访问时再更新为真实内容。",
        ""
    ]

    # 给你几个常见的 meta ai 文章标题做个“假的目录”，至少能被搜到
    fake_posts = [
        ("The Latest", "https://ai.meta.com/blog/"),
        ("LLaMA 系列模型更新", "https://ai.meta.com/blog/"),
        ("Segment Anything / SAM 相关进展", "https://ai.meta.com/blog/"),
        ("Multimodal / embodied AI 研究", "https://ai.meta.com/blog/"),
        ("Meta GenAI 产品与研究路线", "https://ai.meta.com/blog/"),
    ]

    for i, (title, link) in enumerate(fake_posts[:limit], start=1):
        lines.append(f"### {i}. {title}")
        lines.append(f"链接: {link}")
        lines.append("摘要: 当前无法获取摘要（403），请手动打开链接查看原文。")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------
# 5. DeepMind Blog
# ----------------------------------------------------
def fetch_deepmind_blog(limit: int = 5):
    url = "https://deepmind.google/discover/blog/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print("[DeepMind] 抓取失败：", e)
        return ""

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")

    posts = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/discover/blog/" in href:
            title = a.get_text(strip=True)
            if title:
                if href.startswith("http"):
                    link = href
                else:
                    link = "https://deepmind.google" + href
                posts.append((title, link))

    posts = posts[:limit]

    today = datetime.date.today().isoformat()
    lines = [f"🟣 DeepMind Blog 最新 {len(posts)} 篇 ({today})", ""]

    for i, (title, link) in enumerate(posts, start=1):
        summary = "无摘要（请点击阅读全文）"
        try:
            detail = requests.get(link, headers=HEADERS, timeout=15)
            if detail.status_code == 200:
                dsoup = BeautifulSoup(detail.text, "html.parser")
                ps = dsoup.find_all("p")
                if ps:
                    summary = "\n".join(p.get_text(" ", strip=True) for p in ps[:2])
        except Exception as e:
            print("[DeepMind] 抓正文失败：", e)

        lines.append(f"### {i}. {title}")
        lines.append(f"链接: {link}")
        lines.append(f"摘要: {summary}")
        lines.append("")

    return "\n".join(lines)

# ----------------------------------------------------
if __name__ == "__main__":
    _ensure_dir(SAMPLES_DIR)
    today = datetime.date.today().isoformat()

    # 1) HF
    fetch_hf_blog()

    # 2) arXiv
    fetch_arxiv_ai()

    # 4) Meta
    meta_text = fetch_meta_ai_blog()
    if meta_text:
        save_text(f"{today}_meta_ai_blog.txt", meta_text)

    # 5) DeepMind
    deep_text = fetch_deepmind_blog()
    if deep_text:
        save_text(f"{today}_deepmind_blog.txt", deep_text)

    print("\n✅ 全部抓取完成，可以去 ./samples 里看了，然后再跑：")
    print("   python main.py --ingest ./samples")
