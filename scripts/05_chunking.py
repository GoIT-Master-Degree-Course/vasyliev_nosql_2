# scripts/05_chunking.py
import os
import re
import itertools
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, models
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
FIXED_CHUNKS_INDEX_NAME = "arxiv-chunks-fixed"
SEMANTIC_CHUNKS_INDEX_NAME = "arxiv-chunks-semantic"
# INDEX_NAME = "arxiv-papers"
NAMESPACE = "science-papers"
VECTOR_DIM = 768
BATCH_SIZE = 64
TOP_K = 5

# Helper function to batch data
def batch(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    batch = tuple(itertools.islice(it, batch_size))
    while batch:
        yield batch
        batch = tuple(itertools.islice(it, batch_size))



def build_model() -> SentenceTransformer:
    transformer = models.Transformer(MODEL_NAME, max_seq_length=512)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[transformer, pooling])

def fixed_chunking(text, chunk_size=200, chunk_overlap=50):
    """
    Splits the input text into chunks of specified size with overlap.
    
    Parameters:
    - text (str): The input text to be chunked.
    - chunk_size (int): The maximum size of each chunk.
    - chunk_overlap (int): The number of characters to overlap between chunks.
    
    Returns:
    - List[str]: A list of text chunks.
    """
    if not text:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text.strip())
    return chunks


def semantic_chunking(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.7,
    min_chunk_size: int = 50,
) -> List[str]:
    """
    Ділить текст на семантично зв'язні блоки.
    Новий chunk починається, коли косинусна схожість
    між сусідніми реченнями падає нижче threshold.
    """
    # Просте розділення на речення
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if len(sentences) < 2:
        return sentences

    # Отримуємо ембеддинги речень
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Косинусна схожість між сусідніми реченнями
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    chunks, current_chunk = [], [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold and len(" ".join(current_chunk)) >= min_chunk_size:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks




df = pd.read_parquet("data/arxiv_subset.parquet")
df['abstract_length'] = df['abstract'].fillna("").str.len()

df.sort_values(by='abstract_length', ascending=False, inplace=True)

max_len_df = df.iloc[0:30]



# Ініціалізація клієнта Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Ініціалізація моделі для отримання ембедингів
model = build_model()


# Перевіряємо, чи існує індекс (щоб не створювати дублікат)
if FIXED_CHUNKS_INDEX_NAME not in pc.list_indexes().names():
# Створення serverless-індексу
    pc.create_index(
        name=FIXED_CHUNKS_INDEX_NAME,
        dimension=VECTOR_DIM,        # повинна збігатися з розмірністю моделі
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # обирайте регіон ближче до ваших користувачів
        ),
    )
fixed_chunks_index = pc.Index(FIXED_CHUNKS_INDEX_NAME)

# Перевіряємо, чи існує індекс (щоб не створювати дублікат)
if SEMANTIC_CHUNKS_INDEX_NAME not in pc.list_indexes().names():
# Створення serverless-індексу
    pc.create_index(
        name=SEMANTIC_CHUNKS_INDEX_NAME,
        dimension=VECTOR_DIM,        # повинна збігатися з розмірністю моделі
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # обирайте регіон ближче до ваших користувачів
        ),
    )
semantic_chunks_index = pc.Index(SEMANTIC_CHUNKS_INDEX_NAME)



fixed_chunks_vectors_to_upsert = []

for doc_idx in tqdm(range(0, len(max_len_df)), desc="Підготовка fixed chunks векторів до upsert"):
    abstract = max_len_df.iloc[doc_idx]["abstract"]
    arxiv_id = max_len_df.iloc[doc_idx]["id"]
    title = max_len_df.iloc[doc_idx]["title"]
    category = max_len_df.iloc[doc_idx]["category"]
    authors = max_len_df.iloc[doc_idx]["authors"][:200]
    year = int(max_len_df.iloc[doc_idx]["year"])

    chunks = fixed_chunking(abstract, chunk_size=200, chunk_overlap=50)

    for chunk_idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk, normalize_embeddings=True)
        fixed_chunks_vectors_to_upsert.append(
            {
                "id": f"{arxiv_id}_chunk_{chunk_idx}",
                "values": embedding.tolist(),
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "chunk": chunk,
                    "chunk_index": chunk_idx,
                    "year": year,
                    "category": category,
                },
            }
        )


semantic_chunks_vectors_to_upsert = []

for doc_idx in tqdm(range(0, len(max_len_df)), desc="Підготовка semantic chunks векторів до upsert"):
    abstract = max_len_df.iloc[doc_idx]["abstract"]
    arxiv_id = max_len_df.iloc[doc_idx]["id"]
    title = max_len_df.iloc[doc_idx]["title"]
    category = max_len_df.iloc[doc_idx]["category"]
    authors = max_len_df.iloc[doc_idx]["authors"][:200]
    year = int(max_len_df.iloc[doc_idx]["year"])

    chunks = semantic_chunking(abstract, model=model, threshold=0.85)

    for chunk_idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk, normalize_embeddings=True)
        semantic_chunks_vectors_to_upsert.append(
            {
                "id": f"{arxiv_id}_chunk_{chunk_idx}",
                "values": embedding.tolist(),
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "chunk": chunk,
                    "chunk_index": chunk_idx,
                    "year": year,
                    "category": category,
                },
            }
        )

# Upsert data with upsert request
for ids_vectors_batch in tqdm(batch(fixed_chunks_vectors_to_upsert, batch_size=BATCH_SIZE), desc="Запис векторів у Pinecone"):
    for attempt in range(3):
        try:
            fixed_chunks_index.upsert(vectors=ids_vectors_batch, namespace=NAMESPACE)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                raise


print(f"Записано {len(fixed_chunks_vectors_to_upsert)} векторів у namespace '{NAMESPACE}' індексу '{FIXED_CHUNKS_INDEX_NAME}'")


# Upsert data with upsert request
for ids_vectors_batch in tqdm(batch(semantic_chunks_vectors_to_upsert, batch_size=BATCH_SIZE), desc="Запис векторів у Pinecone"):
    for attempt in range(3):
        try:
            semantic_chunks_index.upsert(vectors=ids_vectors_batch, namespace=NAMESPACE)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                raise


print(f"Записано {len(semantic_chunks_vectors_to_upsert)} векторів у namespace '{NAMESPACE}' індексу '{SEMANTIC_CHUNKS_INDEX_NAME}'")


# query_list = [
#     "What are the latest advancements in quantum computing?",
#     "Explain the significance of the Higgs boson discovery.",
#     "How does CRISPR technology work and what are its applications?",
#     "What are the challenges in developing fusion energy?",
#     "Describe the process of photosynthesis in plants.",
# ]

query_list = [
    "inflation debt crisis",
    "attention mechanism in neural networks",
    "star formation in galaxies",
    "reinforcement learning",
    "zoo of exotic animals",
    "space clouds and nebulae clusters",
]

print("\n" + "*" * 50)
print(f"\nДемонстрація пошуку по фіксованих фрагментах та семантичних фрагментах\n")

for query in query_list:
    query_vector = model.encode(query, normalize_embeddings=True,).tolist()

    fixed_chunks_results = fixed_chunks_index.query(
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

    print(f"Запит: '{query}' до бази фіксованих фрагментів\n")
    for match in fixed_chunks_results.matches:
        print(f"ID: {match.id} | Score: {match.score:.4f}")
        print(f"  Title: {match.metadata['title']}")
        # print(f"  Category: {match.metadata['category']}")
        # print(f"  Year: {int(match.metadata['year'])}")
        print(f"  Chunk: {match.metadata['chunk'][:100]}...")
        print()

    print("\n" + "-" * 50)

    semantic_chunks_results = semantic_chunks_index.query(
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
    
    print(f"Запит: '{query}' до бази семантичних фрагментів\n")
    for match in semantic_chunks_results.matches:
        print(f"ID: {match.id} | Score: {match.score:.4f}")
        print(f"  Title: {match.metadata['title']}")
        # print(f"  Category: {match.metadata['category']}")
        # print(f"  Year: {int(match.metadata['year'])}")
        print(f"  Chunk: {match.metadata['chunk'][:100]}...")
        print()

    print("\n" + "=" * 50)


