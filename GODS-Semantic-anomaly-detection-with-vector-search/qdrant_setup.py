import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()


def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")

    return QdrantClient(url=url, api_key=api_key, timeout=60.0)


def check_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        return collection_name in collection_names
    except Exception as e:
        print(f"   Error checking collections: {e}")
        return False


def check_collection_has_data(client: QdrantClient, collection_name: str) -> int:
    try:
        if not check_collection_exists(client, collection_name):
            return 0
        count = client.count(collection_name=collection_name)
        return count.count
    except Exception as e:
        print(f"   Error checking collection data: {e}")
        return 0
