from typing import List, Optional
from langchain.schema import Document
from pypdf import PdfReader
import os

def load_pdf_dir(dir_path: str, only_file: Optional[str] = None) -> List[Document]:
    docs = []
    if not os.path.isdir(dir_path):
        return docs
    for name in os.listdir(dir_path):
        if only_file and name != only_file:
            continue
        if name.lower().endswith(".pdf"):
            path = os.path.join(dir_path, name)
            try:
                reader = PdfReader(path)
                texts = []
                for page in reader.pages:
                    try:
                        texts.append(page.extract_text() or "")
                    except Exception:
                        pass
                content = "\n".join(texts)
                if content.strip():
                    docs.append(Document(page_content=content, metadata={"source": path}))
            except Exception:
                # 忽略损坏或无法解析的 PDF
                pass
    return docs
