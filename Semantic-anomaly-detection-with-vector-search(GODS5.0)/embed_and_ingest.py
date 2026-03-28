import os
from typing import List, Tuple
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid

load_dotenv()


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return OpenAI(api_key=api_key)


def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")
    
    return QdrantClient(url=url, api_key=api_key, timeout=120.0)


def load_balanced_dataset(
    filepath: str = "data/balanced_dataset.txt",
) -> Tuple[List[str], List[str]]:
    print(f"Loading dataset from {filepath}")

    if not os.path.exists(filepath):
        print(f"Dataset file not found: {filepath}")
        print("   Run ingest.py first to create the dataset")
        return [], []

    sequences = []
    labels = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label, sequence = parts
                labels.append(label)
                sequences.append(sequence)

    print(f"   Loaded {len(sequences)} sequences")
    print(f"      - Normal: {labels.count('normal')}")
    print(f"      - Abnormal: {labels.count('abnormal')}")

    return sequences, labels


def create_embeddings(sequences: List[str], batch_size: int = 50) -> List[List[float]]:
    """
    Create embeddings using OpenAI text-embedding-3-small model

    Single vector approach: combines block ID + sequence numbers into one 1536D vector
    Example: "blk_-7229676369905620586,5 5 22 5 11 9 11 9 11 9 26 26 26 4 3 4 2 2 2 23 23 23 21 21 21"

    Args:
        sequences: List of HDFS log sequences
        batch_size: Number of sequences to embed per API call

    Returns:
        List of embedding vectors (1536 dimensions each)
    """
    print("=" * 50)
    print("OPENAI EMBEDDING CREATION")
    print("=" * 50)
    print(f"Creating embeddings for {len(sequences)} sequences")
    print("   Model: text-embedding-3-small (1536D)")
    print("   Approach: Single vector (behavioral + contextual)")
    print()

    client = get_openai_client()
    all_embeddings = []

    # Process in batches for efficiency
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} sequences)"
        )

        try:
            # Create embeddings for batch
            response = client.embeddings.create(
                model="text-embedding-3-small", input=batch
            )

            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            print(f"   Embedded {len(batch)} sequences")

        except Exception as e:
            print(f"   Error in batch {batch_num}: {e}")
            # Continue with next batch rather than failing completely
            continue

    print("Embedding complete!")
    print(f"Total embeddings created: {len(all_embeddings)}")
    print("Dimensions per vector: 1536")

    return all_embeddings


def create_qdrant_collection(client: QdrantClient, collection_name: str, expected_count: int = 2000) -> str:
    """
    Returns: 'exists', 'created', or 'failed'
    """
    try:
        if client.collection_exists(collection_name):
            # Get collection info to check point count
            collection_info = client.get_collection(collection_name)
            data_count = collection_info.points_count
            
            print(f"   Collection '{collection_name}' exists with {data_count} points")
            
            if data_count == expected_count:
                print(f"   Collection has expected {expected_count} points - skipping recreation")
                return 'exists'
            else:
                print(f"   Expected {expected_count} points, found {data_count} - recreating...")
                print(f"   Deleting existing collection '{collection_name}'")
                client.delete_collection(collection_name=collection_name)
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
        print(f"   Created collection '{collection_name}'")
        return 'created'
        
    except Exception as e:
        print(f"   Error creating collection: {e}")
        return 'failed'
        
        
def upload_to_qdrant(client: QdrantClient, collection_name: str, sequences: List[str], labels: List[str], embeddings: List[List[float]], batch_size: int = 100) -> bool:
    print(f"Uploading {len(embeddings)} embeddings to Qdrant...")
    
    try:
        total_uploaded = 0
        total_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        for i in range(0, len(embeddings), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"   Uploading batch {batch_num}/{total_batches} ({len(batch_embeddings)} points)")
            
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "label": label, 
                        "sequence": seq,
                        "true_label": label,
                        "is_anomaly": label == "abnormal"
                    }
                )
                for seq, label, emb in zip(batch_sequences, batch_labels, batch_embeddings)
            ]
            
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            total_uploaded += len(points)
            print(f"   Batch {batch_num} uploaded successfully")
        
        print(f"   Successfully uploaded {total_uploaded} points total")
        return True
        
    except Exception as e:
        print(f"   Error uploading data: {e}")
        return False


def main():
    collection_name = "hdfs_anomaly_detection"
    
    sequences, labels = load_balanced_dataset()
    if not sequences:
        return False
    
    expected_count = len(sequences)
    print(f"Expected dataset size: {expected_count} sequences")

    print("Connecting to Qdrant Cloud...")
    try:
        client = get_qdrant_client()
        print("   Connected to Qdrant Cloud")
    except Exception as e:
        print(f"   Connection failed: {e}")
        return False

    print(f"Checking collection '{collection_name}'...")
    collection_status = create_qdrant_collection(client, collection_name, expected_count)
    
    if collection_status == 'failed':
        return False
    elif collection_status == 'exists':
        print("\n🎉 Collection already contains the expected data!")
        print("   Skipping embedding creation and upload (saving time and money)")
        print(f"   Collection: {collection_name}")
        print(f"   Points: {expected_count}")
        print("   Use detect_distance.py or detect_dbscan.py for anomaly detection")
        return True
    
    # Only create embeddings if collection was newly created or recreated
    print("\n📡 Creating embeddings (this may take a few minutes and costs OpenAI API credits)...")
    embeddings = create_embeddings(sequences)
    if len(embeddings) != len(sequences):
        print(f"Warning: Created {len(embeddings)} embeddings for {len(sequences)} sequences")

    if not upload_to_qdrant(client, collection_name, sequences, labels, embeddings):
        print("Upload failed! Exiting...")
        return False

    # Verify upload worked
    try:
        collection_info = client.get_collection(collection_name)
        actual_count = collection_info.points_count
        print(f"   Verification: Collection now contains {actual_count} points")
        
        if actual_count != len(sequences):
            print(f"   Warning: Expected {len(sequences)} points, but got {actual_count}")
            return False
    except Exception as e:
        print(f"   Error verifying upload: {e}")
        return False

    print("\nEmbedding and upload complete!")
    print(f"   Collection: {collection_name}")
    print(f"   Points: {len(sequences)}")
    print("   Use detect_distance.py or detect_dbscan.py for anomaly detection")
    return True


if __name__ == "__main__":
    main()
