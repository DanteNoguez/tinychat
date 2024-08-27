import os

import pandas as pd

from tinychat.rag.models import VectorDBConfig
from tinychat.rag.registry import VectorDBRegistry

current_dir = os.path.dirname(os.path.abspath(__file__))

csv_file_path = os.path.join(current_dir, "cars.csv")
with open(csv_file_path, "r", encoding="utf-8") as f:
    df = pd.read_csv(f)

KAVAK_CONFIG = VectorDBConfig(
    path=current_dir + "/kavak_db",
    collection_name="kavak",
)

KAVAK_VECTOR_REGISTRY = VectorDBRegistry(
    configs={"default": KAVAK_CONFIG}, default_collection_name="default"
)

if __name__ == "__main__":
    import json

    from dotenv import load_dotenv

    load_dotenv(override=True)

    from tinychat.rag.chroma import ChromaCollection
    from tinychat.rag.registry import VectorDBRegistry

    json_file_path = os.path.join(current_dir, "text_chunks.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    kavak_collection = VectorDBRegistry.create_chroma_collection(
        source_documents=chunks, name="kavak", path=current_dir + "/kavak_db"
    )

    kavak_collection = ChromaCollection(
        config=KAVAK_CONFIG,
    )
    print(kavak_collection.similarity_search("qué es kavak?", 2))
