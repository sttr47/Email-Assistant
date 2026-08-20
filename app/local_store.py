import os
import pickle
from typing import Iterator, Optional, Sequence, Tuple
from langchain_core.stores import BaseStore
from langchain_core.documents import Document

class PickleDocStore(BaseStore[str, Document]):
    def __init__(self, file_path="./docstore.pkl"):
        self.file_path = file_path
        self.memory = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                self.memory = pickle.load(f)

    def mget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        return [self.memory.get(k) for k in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        for k, v in key_value_pairs:
            self.memory[k] = v
        with open(self.file_path, "wb") as f:
            pickle.dump(self.memory, f)

    def mdelete(self, keys: Sequence[str]) -> None:
        for k in keys:
            if k in self.memory:
                del self.memory[k]
        if self.memory:
            with open(self.file_path, "wb") as f:
                pickle.dump(self.memory, f)
        elif os.path.exists(self.file_path):
            os.remove(self.file_path)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        for k in self.memory.keys():
            if prefix is None or k.startswith(prefix):
                yield k