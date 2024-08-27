import os

from loguru import logger

from tinychat.rag.chroma import ChromaCollection
from tinychat.rag.models import VectorDBConfig


class VectorDBRegistry:
    def __init__(
        self, configs: dict[str, VectorDBConfig], default_collection_name: str
    ):
        self.configs = configs
        self.default_collection_name = default_collection_name

    @staticmethod
    def create_chroma_collection(
        source_documents: list,
        name: str,
        path: str,
    ) -> ChromaCollection:
        try:
            # isort: off
            import chromadb
            from chromadb.utils.embedding_functions.openai_embedding_function import (
                OpenAIEmbeddingFunction,
            )

            OPENAI_EMBEDDING_FUNCTION = OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-3-small",
            )
        except ImportError as e:
            raise ImportError("Missing required dependancies for chromadb") from e
        # isort: on
        client = chromadb.PersistentClient(path=path)
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=OPENAI_EMBEDDING_FUNCTION,
        )
        ids = [str(i) for i in range(len(source_documents))]
        collection.add(documents=source_documents, ids=ids)
        chroma_collection = ChromaCollection(
            config=VectorDBConfig(collection_name=name, path=path)
        )
        return chroma_collection

    def register_collection_config(self, collection_name: str, config: VectorDBConfig):
        self.configs[collection_name] = config

    def set_default_collection_name(self, collection_name: str):
        if collection_name not in self.configs:
            raise ValueError(
                f"Collection config '{collection_name}' not found in registry"
            )
        self.default_collection_name = collection_name

    def get_collection(self, collection_name: str) -> ChromaCollection:
        config = self.configs.get(
            collection_name, self.configs[self.default_collection_name]
        )
        return ChromaCollection(config)

    def get_config(self, collection_name: str) -> VectorDBConfig:
        return self.configs.get(
            collection_name, self.configs[self.default_collection_name]
        )
