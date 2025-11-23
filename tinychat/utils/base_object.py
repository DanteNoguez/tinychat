from abc import ABC
from typing import Optional

from tinychat.utils.utils import random_id


class BaseObject(ABC):
    """
    Base class for Python objects with self-identifying attributes.
    """

    def __init__(self, name: Optional[str] = None):
        self._id = random_id()
        self._name = name or f"{self.__class__.__name__}_{self._id}"

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name})"
