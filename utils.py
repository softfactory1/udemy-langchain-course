from loguru import logger
from pyserxng import LocalSearXNGClient

from settings import SETTINGS


class SearXNGClient:
    def __init__(self):
        self.searxng_client = LocalSearXNGClient(SETTINGS.SEARX_HOST)
        # with self.searxng_client as client:
        #     if not client.test_connection():
        #         raise ConnectionError("Could not connect to SearXNG")
        if not self.searxng_client.test_connection():
            raise ConnectionError("Could not connect to SearXNG")
        self.query_history: list[str] = []

    def search(self, query: str) -> str:
        # with self.searxng_client as client:
        #     results = client.search(query)
        #     contents = [r.content for r in results.results[:SETTINGS.SEARXNG_MAX_RESULTS]]
        #     return "\n".join(contents)
        #     raise Exception("Could not connect to SearXNG")

        if query in self.query_history:
            logger.info(f"Searing query {query} repeated, skipping!")
            return ""
        self.query_history.append(query)
        results = self.searxng_client.search(query)
        contents = [r.content for r in results.results[:SETTINGS.SEARXNG_MAX_RESULTS]]
        return "\n".join(contents)

    def close(self):
        self.searxng_client.close()
