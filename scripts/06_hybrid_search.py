import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, models
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
NAMESPACE = "science-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5   # фінальна кількість результатів для виводу
VIEW_WIDTH = 120  # кількість символів для попереднього перегляду результатів

def build_model() -> SentenceTransformer:
    transformer = models.Transformer(MODEL_NAME, max_seq_length=512)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[transformer, pooling])

def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Об'єднує кілька ранжованих списків через RRF.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def vector_search(query, model, index, namespace, top_k):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    results = index.query(
        namespace=namespace,
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
    )
    return [
        (int(m.id.split("_")[1]), m.score, f"{m.metadata['title']} {m.metadata['abstract']}")
        for m in results.matches
    ]

def bm25_search(
    query: str,
    bm25: BM25Okapi,
    top_k: int,
) -> List[Tuple[int, float]]:
    """Повертає список (doc_idx, score), відсортований за score спадаюче."""
    scores = bm25.get_scores(query.lower().split())
    ranking = list(np.argsort(scores)[::-1][:top_k])
    return [(idx, scores[idx]) for idx in ranking]

def hybrid_search(query, model, index, namespace, bm25, corpus, top_k=100):
    vec_results = vector_search(query, model, index, namespace, top_k)
    vector_ranking = [doc_idx for doc_idx, _, _ in vec_results]

    bm25_results = bm25_search(query, bm25, top_k)
    bm25_ranking = [doc_idx for doc_idx, _ in bm25_results]

    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
    return [(corpus[doc_id], score) for doc_id, score in fused]

# Ініціалізація Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

# Ініціалізація моделі ембеддінгів
model = build_model()

# Створення індексу BM25
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)
df['text'] = df['title'] + " " + df['abstract']
corpus = df['text'].tolist()
bm25_idx = BM25Okapi([doc.lower().split() for doc in corpus])

query_list = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

for query in query_list:

    print("\n" + "*" * 50)
    print(f"Запит: '{query}'\n")

    # --- Тільки векторний пошук ---
    print("\n=== Тільки Pinecone (векторний) ===")
    vec_results = vector_search(query, model, index, NAMESPACE, TOP_K)
    for _, score, text in vec_results:
        print(f"  [{score:.3f}]{text[:VIEW_WIDTH]}...")

    # --- BM25 пошук ---
    print("\n=== Тільки BM25 ===")
    bm25_sc = bm25_idx.get_scores(query.lower().split())
    for idx in np.argsort(bm25_sc)[::-1][:TOP_K]:
        print(f"  [{bm25_sc[idx]:.3f}]{corpus[idx][:VIEW_WIDTH]}...")

    # --- Гібридний RRF ---
    print("\n=== Гібридний RRF ===")
    results = hybrid_search(query, model, index, NAMESPACE, bm25_idx, corpus)
    for doc, score in results[:TOP_K]:
        print(f"  [RRF={score:.4f}]{doc[:VIEW_WIDTH]}...")