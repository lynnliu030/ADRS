"""
Mock SnapshotCache for use when caching is disabled.
This allows the reference implementation to import without the full scripts infrastructure.
"""

class SnapshotCache:
    """
    Mock cache that does nothing when disable_cache=True.
    """
    
    def __init__(self, cache_dir: str, disable_cache: bool = False):
        self.cache_dir = cache_dir
        self.disable_cache = disable_cache
    
    def get(self, key: str):
        """Always return None (cache miss)."""
        return None
    
    def put(self, key: str, value):
        """Do nothing - don't store anything."""
        pass
    
    def exists(self, key: str) -> bool:
        """Always return False (no cached entries)."""
        return False
    
    def clear(self):
        """Do nothing - no cache to clear."""
        pass 