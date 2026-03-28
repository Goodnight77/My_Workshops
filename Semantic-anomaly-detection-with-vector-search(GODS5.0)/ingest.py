import os
import requests
from typing import List, Tuple
import random


def download_hdfs_data() -> bool:
    print("=" * 50)
    print("HDFS DATA INGESTION")
    print("=" * 50)

    base_url = "https://raw.githubusercontent.com/ait-aecid/anomaly-detection-log-datasets/main/hdfs_loghub"

    files = {
        "train": "data/hdfs_train.txt",
        "test_normal": "data/hdfs_test_normal.txt",
        "test_abnormal": "data/hdfs_test_abnormal.txt",
    }

    url_mapping = {
        "train": "hdfs_train",
        "test_normal": "hdfs_test_normal",
        "test_abnormal": "hdfs_test_abnormal",
    }

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Download each file
    success = True
    for name, path in files.items():
        if not os.path.exists(path):
            url = f"{base_url}/{url_mapping[name]}"
            print(f"Downloading {name}...")

            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"   Saved to {path}")
                else:
                    print(f"   Error: HTTP {response.status_code}")
                    success = False
            except Exception as e:
                print(f"   Error downloading {name}: {e}")
                success = False
        else:
            print(f"{name} already exists, skipping download")

    print()
    return success


def load_sequences(filepath: str) -> List[str]:
    sequences = []

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return sequences

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq)

    return sequences


def create_balanced_dataset(
    target_size: int = 2000, normal_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    print("Creating balanced dataset...")

    normal_count = int(target_size * normal_ratio)
    abnormal_count = target_size - normal_count

    print(
        f"   Target: {target_size} total ({normal_count} normal, {abnormal_count} abnormal)"
    )

    # Load data files
    train_sequences = load_sequences("data/hdfs_train.txt")
    normal_sequences = load_sequences("data/hdfs_test_normal.txt")
    abnormal_sequences = load_sequences("data/hdfs_test_abnormal.txt")

    print(
        f"   Available: {len(train_sequences)} train, {len(normal_sequences)} normal, {len(abnormal_sequences)} abnormal"
    )

    # Combine normal sources
    all_normal = train_sequences + normal_sequences

    # Sample sequences
    random.seed(42) 

    if len(all_normal) < normal_count:
        print(
            f"   Warning: Only {len(all_normal)} normal sequences available, using all"
        )
        selected_normal = all_normal
    else:
        selected_normal = random.sample(all_normal, normal_count)

    if len(abnormal_sequences) < abnormal_count:
        print(
            f"Only {len(abnormal_sequences)} abnormal sequences available, using all"
        )
        selected_abnormal = abnormal_sequences
    else:
        selected_abnormal = random.sample(abnormal_sequences, abnormal_count)

    # Combine and create labels
    all_sequences = selected_normal + selected_abnormal
    all_labels = ["normal"] * len(selected_normal) + ["abnormal"] * len(
        selected_abnormal
    )

    # Shuffle to mix normal and abnormal
    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)

    print(f"   Created balanced dataset: {len(sequences)} sequences")
    print(f"      - Normal: {labels.count('normal')}")
    print(f"      - Abnormal: {labels.count('abnormal')}")

    return list(sequences), list(labels)


def save_balanced_dataset(
    sequences: List[str],
    labels: List[str],
    output_file: str = "data/balanced_dataset.txt",
) -> None:
    """Save balanced dataset to file with labels"""
    print(f"Saving balanced dataset to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for seq, label in zip(sequences, labels):
            f.write(f"{label}\t{seq}\n")

    print(f"   Saved {len(sequences)} sequences")


def main():
    balanced_dataset_path = "data/balanced_dataset.txt"
    expected_size = 2000
    
    if os.path.exists(balanced_dataset_path):
        try:
            with open(balanced_dataset_path, "r", encoding="utf-8") as f:
                existing_lines = sum(1 for _ in f)
            
            if existing_lines == expected_size:
                print("\n✅ Balanced dataset already exists with expected size!")
                print(f"   File: {balanced_dataset_path}")
                print(f"   Size: {existing_lines} sequences")
                print("   Skipping data ingestion (file already ready)")
                return True
            else:
                print(f"Existing dataset has {existing_lines} lines, expected {expected_size}")
                print("   Recreating balanced dataset...")
        except Exception as e:
            print(f"Error reading existing dataset: {e}")
            print("   Recreating balanced dataset...")
    
    # Download data
    if not download_hdfs_data():
        print("Data download failed!")
        return False

    # Create balanced dataset
    try:
        sequences, labels = create_balanced_dataset(target_size=expected_size, normal_ratio=0.8)
        if not sequences:
            print("Failed to create balanced dataset!")
            return False
            
        save_balanced_dataset(sequences, labels)
    except Exception as e:
        print(f"Error creating balanced dataset: {e}")
        return False

    print()
    print("Data ingestion complete!")
    print("   Use embed_and_ingest.py to create embeddings")
    return True


if __name__ == "__main__":
    main()
