import os
from typing import List
import json  # 【新增】存一些元信息
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

BASE_DIR = os.path.dirname(__file__)
MEMORY_DIR = os.path.join(BASE_DIR, "memory")      # 而不是 "./memory"
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
try:
    import faiss
    USE_FAISS = True
except Exception:
    USE_FAISS = False


import numpy as np  # 【新增】外面统一 import，下面要多次用
class VectorStore:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.db_type = os.getenv("VECTOR_DB", "faiss")
        self.docs: List[Document] = []
        self.index = None

    def build(self, docs: List[Document]):
        self.docs = docs
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        import numpy as np
        vecs = np.array(vectors, dtype="float32")
        if self.db_type == "faiss" and USE_FAISS:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
            # 向量归一化以提升内积检索稳定性
            faiss.normalize_L2(vecs)
            index.add(vecs)
            self.index = index
        else:
            # 简易后备：内存线性检索
            self.index = vecs
        print(f"[DEBUG] 嵌入向量维度: {len(vectors[0])}")
        print(f"[DEBUG] 索引构建完成, 共 {len(docs)} 个文档块")

    # =========================================================
    # 【新增】把当前向量库保存到本地
    # =========================================================
    def save(self, path: str = "./memory"):
        """
        把 docs + 向量索引 落盘，方便下次直接加载
        会生成：
          path/docs.json
          path/index.faiss   (faiss 才有)
          path/index.npy     (非 faiss 用)
        """
        os.makedirs(path, exist_ok=True)

        # 1) 存文档
        docs_data = [
            {
                "page_content": d.page_content,
                "metadata": d.metadata,
            }
            for d in self.docs
        ]
        with open(os.path.join(path, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        # 2) 存向量
        if isinstance(self.index, np.ndarray):
            np.save(os.path.join(path, "index.npy"), self.index)
            index_type = "ndarray"
        else:
            # faiss
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
            index_type = "faiss"

        # 3) 存个 meta，告诉我们当时用的是什么向量后端
        meta = {
            "db_type": self.db_type,
            "index_type": index_type,
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[DEBUG] 向量库已保存到 {path}")

    # =========================================================
    # 【新增】从本地加载向量库
    # =========================================================
    def load(self, path: str = "./memory"):
        """
        从本地把上次保存的向量库加载回来
        """
        meta_file = os.path.join(path, "meta.json")
        docs_file = os.path.join(path, "docs.json")

        if not os.path.exists(meta_file) or not os.path.exists(docs_file):
            raise FileNotFoundError(f"[DEBUG] 在 {path} 里没有找到已保存的向量库，请先运行 ingest")

        # 1) 读文档
        with open(docs_file, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        self.docs = [
            Document(page_content=d["page_content"], metadata=d.get("metadata") or {})
            for d in docs_data
        ]

        # 2) 读 meta
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        index_type = meta.get("index_type", "ndarray")

        # 3) 读向量
        if index_type == "ndarray":
            self.index = np.load(os.path.join(path, "index.npy"))
        else:  # faiss
            self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        print(f"[DEBUG] 从 {path} 加载向量库成功，共 {len(self.docs)} 个文档块")


    def search(self, query: str, k: int = 5) -> List[Document]:
        if self.index is None or not self.docs:
            return []
        qv = self.embeddings.embed_query(query)
        import numpy as np
        q = np.array([qv], dtype="float32")
        if isinstance(self.index, np.ndarray):
            # 线性遍历（余弦相似近似）
            from numpy.linalg import norm
            mat = self.index
            sims = mat @ (q.T / (norm(q) + 1e-9))
            sims = sims.reshape(-1)
            idx = sims.argsort()[::-1][:k]
            return [self.docs[i] for i in idx]
        else:
            # FAISS 内积
            import faiss
            faiss.normalize_L2(q)
            _, I = self.index.search(q, k)
            return [self.docs[i] for i in I[0] if i >= 0]
