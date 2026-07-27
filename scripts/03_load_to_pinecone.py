# scripts/03_load_to_pinecone.py
import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Constants
INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
NAMESPACE = "science-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Helper function to batch data
def batch(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    batch = tuple(itertools.islice(it, batch_size))
    while batch:
        yield batch
        batch = tuple(itertools.islice(it, batch_size))


# Ініціалізація клієнта Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


# Перевіряємо, чи існує індекс (щоб не створювати дублікат)
if INDEX_NAME not in pc.list_indexes().names():
# Створення serverless-індексу
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,        # повинна збігатися з розмірністю моделі
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # обирайте регіон ближче до ваших користувачів
        ),
    )

index = pc.Index(INDEX_NAME)

# Завантажуємо ембединги та документи
embeddings = np.load(INPUT_EMBEDDINGS)
documents = pd.read_parquet(INPUT_PARQUET)
# documents['year'] = documents['year'].astype(str)  # Перетворюємо рік на string для збереження в метаданих

# Перевірка відповідності кількості ембедингів та документів
if len(embeddings) != len(documents):
    raise ValueError("Кількість ембедингів не збігається з кількістю документів!")  

# Отримуємо ембединги і документи та записуємо в індекс
vectors_to_upsert = [
    {
        "id": f"paper_{doc_idx}",
        "values": embeddings[doc_idx].tolist(),
        "metadata": {
            "arxiv_id": documents.iloc[doc_idx]["id"],
            "title": documents.iloc[doc_idx]["title"],
            "authors": documents.iloc[doc_idx]["authors"][:200],  # обмежуємо довжину списку авторів для метаданих
            "abstract": documents.iloc[doc_idx]["abstract"][:500],  # обмежуємо довжину абстракту для метаданих
            "year": int(documents.iloc[doc_idx]["year"]),
        },
    }
    for doc_idx in tqdm(range(0, len(documents)), desc="Підготовка векторів до upsert")
]

# Upsert data with 200 vectors per upsert request
for ids_vectors_batch in tqdm(batch(vectors_to_upsert, batch_size=BATCH_SIZE), desc="Запис векторів у Pinecone"):
    for attempt in range(3):
        try:
            index.upsert(vectors=ids_vectors_batch, namespace=NAMESPACE)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                raise


print(f"Записано {len(vectors_to_upsert)} векторів у namespace '{NAMESPACE}' індексу '{INDEX_NAME}'")
