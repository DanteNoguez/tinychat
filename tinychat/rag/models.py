from typing import Optional

from pydantic import BaseModel


class VectorDBConfig(BaseModel):
    collection_name: str
    path: Optional[str] = None
