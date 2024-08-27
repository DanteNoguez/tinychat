import os

from dotenv import load_dotenv

load_dotenv()

from tinychat.rag.models import VectorDBConfig


class ChromaCollection:
    def __init__(self, config: VectorDBConfig):
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
        self.client = chromadb.PersistentClient(path=config.path)
        self.collection = self.client.get_collection(
            name=config.collection_name, embedding_function=OPENAI_EMBEDDING_FUNCTION
        )

    def similarity_search(self, query: str, n_results: int = 1):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        return results.get("documents")
