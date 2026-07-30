# scripts/04_search.py
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, models

load_dotenv()

INDEX_NAME = "arxiv-papers"
NAMESPACE = "science-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"

def build_model() -> SentenceTransformer:
    transformer = models.Transformer(MODEL_NAME, max_seq_length=512)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[transformer, pooling])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

model = build_model()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

#####################
print(f"\nЧистий семантичний пошук\n")

query = "attention mechanism in neural networks"
query_vector = model.encode(query, normalize_embeddings=True,).tolist()

results = index.query(
    namespace=NAMESPACE,
    vector=query_vector,
    top_k=TOP_K,
    include_metadata=True,
    # filter={
    #     "$and": [
    #         {"language": {"$eq": "en"}},
    #         {"year": {"$gte": 2023}},
    #     ]
    # },
)

print(f"Запит: '{query}'\n")
for match in results.matches:
    print(f"ID: {match.id} | Score: {match.score:.4f}")
    print(f"  Title: {match.metadata['title']}")
    print(f"  Category: {match.metadata['category']}")
    print(f"  Year: {int(match.metadata['year'])}")
    print(f"  Abstract: {match.metadata['abstract'][:100]}...")
    print()


#########################

input("\nНатисніть Enter, щоб продовжити до пошуку з фільтрацією... \n")
print("\n" + "*" * 50)
print(f"\nПошук з фільтрацією. Приклад 1\n")

query = "reinforcement learning"
query_vector = model.encode(query, normalize_embeddings=True,).tolist()

results = index.query(
    namespace=NAMESPACE,
    vector=query_vector,
    top_k=TOP_K,
    include_metadata=True,
    filter={
        "$and": [
            {"category": {"$eq": "cs.LG"}},
            {"year": {"$gte": int(datetime.now().year) - 5}},
        ]
    },
)

print(f"Запит: '{query}'\n")
for match in results.matches:
    print(f"ID: {match.id} | Score: {match.score:.4f}")
    print(f"  Title: {match.metadata['title']}")
    print(f"  Category: {match.metadata['category']}")
    print(f"  Year: {int(match.metadata['year'])}")
    print(f"  Abstract: {match.metadata['abstract'][:100]}...")
    print()


#################

input("\nНатисніть Enter, щоб продовжити до пошуку з фільтрацією... \n")
print("\n" + "*" * 50)
print(f"\nПошук з фільтрацією. Приклад 2\n")

query = "reinforcement learning"
query_vector = model.encode(query, normalize_embeddings=True,).tolist()

results = index.query(
    namespace=NAMESPACE,
    vector=query_vector,
    top_k=TOP_K,
    include_metadata=True,
    filter=
    # {"$and": [
        #     {"category": {"$eq": "cs.LG"}},
            {"year": {"$lt": 2015}},
        # ]
    # },
)

print(f"Запит: '{query}'\n")
for match in results.matches:
    print(f"ID: {match.id} | Score: {match.score:.4f}")
    print(f"  Title: {match.metadata['title']}")
    print(f"  Category: {match.metadata['category']}")
    print(f"  Year: {int(match.metadata['year'])}")
    print(f"  Abstract: {match.metadata['abstract'][:100]}...")
    print()


##################

input("\nНатисніть Enter, щоб порівняти метрики схожості... \n")
print("\n" + "*" * 50)
print(f"\nПорівння метрик схожості\n")

embeddings = np.load(INPUT_EMBEDDINGS)
query = "transformer architecture for NLP"
query_vector = model.encode(query, normalize_embeddings=True,).tolist()

cosine_similarities = []
dot_prod_similarities = []
l2_dist_similarities = []

for embedding in tqdm(embeddings):
    cosine_similarities.append(cosine_similarity(query_vector, embedding))  
    dot_prod_similarities.append(dot_product(query_vector, embedding))  
    l2_dist_similarities.append(l2_distance(query_vector, embedding))

cosine_similarities = np.argsort(np.array(cosine_similarities), descending=False)
dot_prod_similarities = np.argsort(np.array(dot_prod_similarities), descending=False)
l2_dist_similarities = np.argsort(np.array(l2_dist_similarities), descending=True)


print(f"Запит: '{query}'\n")
print(f"Топ 5 за косинусною схожістю:")
for idx in cosine_similarities[:5]:
    print(f"  ID: {idx} | Cosine Similarity: {cosine_similarity(query_vector, embeddings[idx]):.4f}")

print(f"\nТоп 5 за скалярним добутком:")
for idx in dot_prod_similarities[:5]:
    print(f"  ID: {idx} | Dot Product: {dot_product(query_vector, embeddings[idx]):.4f}")

print(f"\nТоп 5 за L2 відстанню:")
for idx in l2_dist_similarities[:5]:
    print(f"  ID: {idx} | L2 Distance: {l2_distance(query_vector, embeddings[idx]):.4f}")
