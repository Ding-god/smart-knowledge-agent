from typing import List, Optional
from langchain.schema import Document
import os

def load_md_dir(dir_path: str, only_file: Optional[str] = None) -> List[Document]:
    docs = []
    if not os.path.isdir(dir_path):
        return docs
    for name in os.listdir(dir_path):
        if only_file and name != only_file:
            continue
        if name.lower().endswith((".md", ".txt")):
            path = os.path.join(dir_path, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    docs.append(Document(page_content=content, metadata={"source": path}))
            except Exception:
                pass
    print(f"[DEBUG] 读取文档: {name}, 长度: {len(content)} 字符")

    return docs
