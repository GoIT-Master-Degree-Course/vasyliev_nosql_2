import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, models

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
# MODEL_NAME = "sentence-transformers/allenai-specter"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

def make_text(row: pd.Series) -> str:
    title = str(row["title"]).strip()
    abstract = str(row["abstract"]).strip()
    return f"{title} [SEP] {abstract}"

def build_model() -> SentenceTransformer:
    transformer = models.Transformer(MODEL_NAME, max_seq_length=512)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[transformer, pooling])

def main() -> None:

    df = pd.read_parquet(INPUT_FILE)

    model = build_model()

    texts = [make_text(row) for _, row in df.iterrows()]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    os.makedirs("embeddings", exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)

    print(f"\nCreated {embeddings.shape[0]} embeddings for {len(df)} documents.")
    print(f"\nEmbedding shape: {embeddings.shape[1]}")
    print(f"\nSaved {embeddings.shape[0]} embeddings to {OUTPUT_FILE}")
    print(f"\nThe first embedding norm: {np.linalg.norm(embeddings[0])}")
    print(f"\nThe first embedding vector:\n{embeddings[0]}")

if __name__ == "__main__":
    main()