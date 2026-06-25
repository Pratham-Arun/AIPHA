import hashlib

class RetrievalCache:
    """
    A simple in-memory cache to store retrieval results for repeated queries.
    In a real production system, this could be backed by Redis.
    """
    def __init__(self):
        self._cache = {}

    def _hash_query(self, query: str, filter_dict: dict) -> str:
        key = f"{query}_{filter_dict}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, query: str, filter_dict: dict = None):
        key = self._hash_query(query, filter_dict)
        return self._cache.get(key)

    def set(self, query: str, results: tuple, filter_dict: dict = None):
        key = self._hash_query(query, filter_dict)
        self._cache[key] = results
