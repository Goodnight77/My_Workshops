import os
import numpy as np
from typing import List, Tuple
from qdrant_client import QdrantClient
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.offline as pyo

load_dotenv()


def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")

    return QdrantClient(url=url, api_key=api_key, timeout=60.0)


def fetch_data_from_qdrant(
    collection_name: str = "hdfs_anomaly_detection",
) -> Tuple[List[List[float]], List[str], List[str]]:
    """Fetch all embeddings and metadata from Qdrant collection - OPTIMIZED"""
    print("=" * 50)
    print("DISTANCE-BASED ANOMALY DETECTION (OPTIMIZED)")
    print("=" * 50)
    print(f"Fetching data from Qdrant collection: {collection_name}")

    client = get_qdrant_client()

    # Get total count first
    try:
        count_result = client.count(collection_name=collection_name)
        total_points = count_result.count
        print(f"   Total points in collection: {total_points}")
    except Exception as e:
        print(f"   Warning: Could not get count: {e}")
        total_points = None

    # Fetch all points in one go with larger batch size
    embeddings = []
    sequences = []
    true_labels = []

    try:
        # Use larger limit for fewer API calls
        results = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Much larger batch
            with_payload=True,
            with_vectors=True,
        )

        points = results[0]
        for point in points:
            embeddings.append(point.vector)
            sequences.append(point.payload.get("sequence", ""))
            true_labels.append(point.payload.get("true_label", "unknown"))

        print(f"   Loaded {len(embeddings)} embeddings in single batch")
    except Exception as e:
        print(f"   Error fetching data: {e}")
        return [], [], []

    print(f"      - Normal: {true_labels.count('normal')}")
    print(f"      - Abnormal: {true_labels.count('abnormal')}")

    return embeddings, sequences, true_labels


def distance_based_detection(
    embeddings: List[List[float]], k: int = 5, threshold: float = 0.6
) -> Tuple[List[bool], List[float]]:
    """
    OPTIMIZED distance-based anomaly detection

    Logic: Points with no close neighbors (within threshold) are anomalies

    Optimizations:
    - Use PCA to reduce dimensions first (much faster)
    - Use euclidean distance instead of cosine (much faster)
    - Tuned parameters for better anomaly detection

    Args:
        embeddings: List of embedding vectors
        k: Number of nearest neighbors to check
        threshold: Distance threshold for considering neighbors as "close"

    Returns:
        Tuple of (is_anomaly_flags, distances_to_kth_neighbor)
    """
    print("Running OPTIMIZED distance-based detection...")

    embeddings_array = np.array(embeddings)
    print(f"   Original dimensions: {embeddings_array.shape}")

    # OPTIMIZATION 1: Reduce dimensions first for much faster distance calculation
    print("   Reducing dimensions for faster distance calculation...")
    pca_distance = PCA(
        n_components=300, random_state=42
    )  # Use more components to preserve information
    embeddings_reduced = pca_distance.fit_transform(embeddings_array)
    print(f"   Reduced to {embeddings_reduced.shape[1]}D for distance calculation")

    # OPTIMIZATION 2: Use euclidean distance and parallel processing
    nn = NearestNeighbors(
        n_neighbors=k + 1, metric="euclidean", n_jobs=-1
    )  # Much faster than cosine
    nn.fit(embeddings_reduced)

    # Find distances to k-th nearest neighbor for each point
    distances, indices = nn.kneighbors(embeddings_reduced)

    # Distance to k-th neighbor (skip [0] which is the point itself)
    kth_distances = distances[:, k]

    # AUTO-TUNE THRESHOLD if not provided
    if threshold is None:
        # Use more conservative threshold for better precision
        threshold = np.percentile(
            kth_distances, 85
        )  # Top 15% as anomalies for better precision
        print(
            f"   AUTO-TUNED threshold: {threshold:.4f} (targeting ~15% anomalies for better precision)"
        )

    print(f"   Parameters: k={k}, threshold={threshold:.4f}")

    # Points with k-th neighbor distance > threshold are anomalies
    is_anomaly = kth_distances > threshold

    anomaly_count = np.sum(is_anomaly)
    print("Detection results:")
    print(f"      - Total points: {len(embeddings)}")
    print(f"      - Detected anomalies: {anomaly_count}")
    print(f"      - Detection rate: {anomaly_count / len(embeddings) * 100:.1f}%")

    return is_anomaly.tolist(), kth_distances.tolist()


def reduce_dimensions_pca(embeddings: List[List[float]]) -> np.ndarray:
    from sklearn.manifold import TSNE

    print("Reducing dimensions for visualization...")

    embeddings_array = np.array(embeddings)

    # First reduce to reasonable dimensions with PCA if too high-dimensional
    if embeddings_array.shape[1] > 50:
        pca_pre = PCA(n_components=50, random_state=42)
        embeddings_pre = pca_pre.fit_transform(embeddings_array)
    else:
        embeddings_pre = embeddings_array

    # Use t-SNE for final 2D visualization (better for clustering visualization)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    points_2d = tsne.fit_transform(embeddings_pre)

    print("   Reduced to 2D using t-SNE for better cluster visualization")

    return points_2d


def create_distance_visualization(
    points_2d: np.ndarray,
    is_anomaly: List[bool],
    true_labels: List[str],
    distances: List[float],
    sequences: List[str],
    output_file: str = "visualization/distance_detection_results.html",
) -> None:
    """Create interactive HTML visualization of distance-based detection results"""
    os.makedirs("visualization", exist_ok=True)

    print(f"Creating visualization: {output_file}")

    # Prepare data for plotting
    normal_points = {"x": [], "y": [], "text": [], "distances": []}
    detected_anomalies = {"x": [], "y": [], "text": [], "distances": []}
    true_positives = {"x": [], "y": [], "text": [], "distances": []}
    false_positives = {"x": [], "y": [], "text": [], "distances": []}

    for i, (point, anomaly, true_label, distance, sequence) in enumerate(
        zip(points_2d, is_anomaly, true_labels, distances, sequences)
    ):
        # Create hover text
        seq_preview = sequence[:60] + "..." if len(sequence) > 60 else sequence
        hover_text = (
            f"Point {i + 1}\\n"
            f"True Label: {true_label}\\n"
            f"Detected: {'Anomaly' if anomaly else 'Normal'}\\n"
            f"Distance: {distance:.4f}\\n"
            f"Sequence: {seq_preview}"
        )

        if not anomaly:
            # Predicted normal
            normal_points["x"].append(point[0])
            normal_points["y"].append(point[1])
            normal_points["text"].append(hover_text)
            normal_points["distances"].append(distance)
        else:
            # Predicted anomaly
            detected_anomalies["x"].append(point[0])
            detected_anomalies["y"].append(point[1])
            detected_anomalies["text"].append(hover_text)
            detected_anomalies["distances"].append(distance)

            if true_label == "abnormal":
                # True positive
                true_positives["x"].append(point[0])
                true_positives["y"].append(point[1])
                true_positives["text"].append(hover_text)
                true_positives["distances"].append(distance)
            else:
                # False positive
                false_positives["x"].append(point[0])
                false_positives["y"].append(point[1])
                false_positives["text"].append(hover_text)
                false_positives["distances"].append(distance)

    # Create traces
    traces = []

    # Normal points
    if normal_points["x"]:
        traces.append(
            go.Scatter(
                x=normal_points["x"],
                y=normal_points["y"],
                mode="markers",
                name=f"Normal ({len(normal_points['x'])})",
                text=normal_points["text"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=8,
                    color="lightblue",
                    symbol="circle",
                    line=dict(width=1, color="white"),
                ),
            )
        )

    # True positives (correctly detected anomalies)
    if true_positives["x"]:
        traces.append(
            go.Scatter(
                x=true_positives["x"],
                y=true_positives["y"],
                mode="markers",
                name=f"True Positives ({len(true_positives['x'])})",
                text=true_positives["text"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=12, color="red", symbol="x", line=dict(width=2)),
            )
        )

    # False positives (incorrectly detected anomalies)
    if false_positives["x"]:
        traces.append(
            go.Scatter(
                x=false_positives["x"],
                y=false_positives["y"],
                mode="markers",
                name=f"False Positives ({len(false_positives['x'])})",
                text=false_positives["text"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=12, color="orange", symbol="x", line=dict(width=2)),
            )
        )

    # Calculate performance metrics
    true_anomalies = true_labels.count("abnormal")
    detected_anomalies_count = sum(is_anomaly)
    true_positives_count = len(true_positives["x"])
    false_positives_count = len(false_positives["x"])

    precision = (
        true_positives_count / detected_anomalies_count
        if detected_anomalies_count > 0
        else 0
    )
    recall = true_positives_count / true_anomalies if true_anomalies > 0 else 0

    # Create layout
    layout = go.Layout(
        title=dict(
            text=f"Distance-Based Detection Results<br>"
            f"<sub>Precision: {precision:.3f} | Recall: {recall:.3f} | True Positives: {true_positives_count}</sub>",
            x=0.5,
        ),
        xaxis=dict(title="t-SNE Dimension 1"),
        yaxis=dict(title="t-SNE Dimension 2"),
        hovermode="closest",
        showlegend=True,
        width=1000,
        height=700,
        margin=dict(l=50, r=50, t=100, b=50),
    )

    # Create figure and save
    fig = go.Figure(data=traces, layout=layout)
    pyo.plot(fig, filename=output_file, auto_open=False)

    print(f"   Saved visualization to: {output_file}")
    print("Performance Summary:")
    print(f"      - Precision: {precision:.3f}")
    print(f"      - Recall: {recall:.3f}")
    print(f"      - True Positives: {true_positives_count}")
    print(f"      - False Positives: {false_positives_count}")


def main():
    collection_name = "hdfs_anomaly_detection"

    # Fetch data from Qdrant
    embeddings, sequences, true_labels = fetch_data_from_qdrant(collection_name)

    if not embeddings:
        print("No data found. Run embed_and_ingest.py first.")
        return

    # Run distance-based detection with AUTO-TUNING
    is_anomaly, distances = distance_based_detection(embeddings, k=5, threshold=None)

    # Reduce dimensions for visualization
    points_2d = reduce_dimensions_pca(embeddings)

    # Create visualization
    create_distance_visualization(
        points_2d, is_anomaly, true_labels, distances, sequences
    )

    print()
    print("Distance-based detection complete!")


if __name__ == "__main__":
    main()
