import os
import numpy as np
from typing import List, Tuple
from qdrant_client import QdrantClient
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
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
    """Fetch all embeddings and metadata from Qdrant collection"""
    print("DBSCAN-BASED ANOMALY DETECTION")
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


def dbscan_clustering(
    embeddings: List[List[float]], eps: float = None, min_samples: int = 4
) -> Tuple[List[int], int, int]:
    """
    Apply DBSCAN clustering

    Logic: Points that do not belong to any cluster (labeled as -1) are anomalies

    Optimizations:
    - Use PCA to reduce dimensions first (much faster)
    - Use euclidean distance instead of cosine (much faster)
    - AUTO-TUNE eps based on k-distance analysis

    Args:
        embeddings: List of embedding vectors
        eps: Maximum distance between samples (auto-tuned if None)
        min_samples: Minimum samples in a neighborhood for DBSCAN

    Returns:
        Tuple of (cluster_labels, n_clusters, n_anomalies)
    """
    print("Running OPTIMIZED DBSCAN clustering...")

    embeddings_array = np.array(embeddings)
    print(f"   Original dimensions: {embeddings_array.shape}")

    # OPTIMIZATION 1: Reduce dimensions first for much faster clustering
    print("   Reducing dimensions for faster clustering...")
    pca_clustering = PCA(
        n_components=300, random_state=42
    )  # Use more components to preserve information
    embeddings_reduced = pca_clustering.fit_transform(embeddings_array)
    print(f"   Reduced to {embeddings_reduced.shape[1]}D for clustering")

    # AUTO-TUNE EPS if not provided
    if eps is None:
        from sklearn.neighbors import NearestNeighbors

        # Use k-distance method to find optimal eps
        k = min_samples + 1
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(embeddings_reduced)
        distances, _ = nn.kneighbors(embeddings_reduced)
        k_distances = distances[:, k - 1]  # k-th nearest neighbor distance
        k_distances = np.sort(k_distances)[::-1]  # Sort in descending order

        eps = np.percentile(k_distances, 75)
        print(
            f"   AUTO-TUNED eps: {eps:.4f} (based on k-distance analysis for better precision)"
        )

    print(f"   Parameters: eps={eps:.4f}, min_samples={min_samples}")
    # OPTIMIZATION 2: Use euclidean distance (much faster than cosine)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    cluster_labels = dbscan.fit_predict(embeddings_reduced)

    # Count clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_anomalies = list(cluster_labels).count(-1)

    print("Clustering results:")
    print(f"      - Total points: {len(embeddings)}")
    print(f"      - Clusters found: {n_clusters}")
    print(f"      - Anomalies (noise): {n_anomalies}")
    print(f"      - Detection rate: {n_anomalies / len(embeddings) * 100:.1f}%")

    return cluster_labels.tolist(), n_clusters, n_anomalies


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


def create_dbscan_visualization(
    points_2d: np.ndarray,
    cluster_labels: List[int],
    true_labels: List[str],
    sequences: List[str],
    n_clusters: int,
    n_anomalies: int,
    output_file: str = "visualization/dbscan_detection_results.html",
) -> None:
    """Create interactive HTML visualization of DBSCAN clustering results"""
    # Create visualization directory if it doesn't exist
    os.makedirs("visualization", exist_ok=True)

    print(f"Creating visualization: {output_file}")

    # Color palette for clusters
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    traces = []

    # Group points by cluster
    clusters = {}
    anomalies = {
        "x": [],
        "y": [],
        "text": [],
        "true_positives": [],
        "false_positives": [],
    }

    for i, (point, cluster_id, true_label, sequence) in enumerate(
        zip(points_2d, cluster_labels, true_labels, sequences)
    ):
        # Create hover text
        seq_preview = sequence[:60] + "..." if len(sequence) > 60 else sequence
        hover_text = (
            f"Point {i + 1}\\n"
            f"True Label: {true_label}\\n"
            f"Cluster: {cluster_id if cluster_id >= 0 else 'Anomaly'}\\n"
            f"Sequence: {seq_preview}"
        )

        if cluster_id == -1:
            # Anomaly (noise point)
            anomalies["x"].append(point[0])
            anomalies["y"].append(point[1])
            anomalies["text"].append(hover_text)

            if true_label == "abnormal":
                anomalies["true_positives"].append(len(anomalies["x"]) - 1)
            else:
                anomalies["false_positives"].append(len(anomalies["x"]) - 1)
        else:
            # Normal cluster point
            if cluster_id not in clusters:
                clusters[cluster_id] = {"x": [], "y": [], "text": []}
            clusters[cluster_id]["x"].append(point[0])
            clusters[cluster_id]["y"].append(point[1])
            clusters[cluster_id]["text"].append(hover_text)

    # Create cluster traces
    for cluster_id, cluster_data in clusters.items():
        color = colors[cluster_id % len(colors)]
        traces.append(
            go.Scatter(
                x=cluster_data["x"],
                y=cluster_data["y"],
                mode="markers",
                name=f"Cluster {cluster_id} ({len(cluster_data['x'])})",
                text=cluster_data["text"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=8,
                    color=color,
                    symbol="circle",
                    line=dict(width=1, color="white"),
                ),
            )
        )

    # Create anomaly traces
    if anomalies["x"]:
        # All anomalies
        traces.append(
            go.Scatter(
                x=anomalies["x"],
                y=anomalies["y"],
                mode="markers",
                name=f"Detected Anomalies ({len(anomalies['x'])})",
                text=anomalies["text"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=12, color="red", symbol="x", line=dict(width=2)),
            )
        )

        # Highlight true positives separately if any
        if anomalies["true_positives"]:
            tp_indices = anomalies["true_positives"]
            traces.append(
                go.Scatter(
                    x=[anomalies["x"][i] for i in tp_indices],
                    y=[anomalies["y"][i] for i in tp_indices],
                    mode="markers",
                    name=f"True Positives ({len(tp_indices)})",
                    text=[anomalies["text"][i] for i in tp_indices],
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(
                        size=14, color="darkred", symbol="x", line=dict(width=3)
                    ),
                )
            )

    # Calculate performance metrics
    true_anomalies = true_labels.count("abnormal")
    detected_anomalies = n_anomalies
    true_positives_count = len(anomalies["true_positives"])
    false_positives_count = len(anomalies["false_positives"])

    precision = (
        true_positives_count / detected_anomalies if detected_anomalies > 0 else 0
    )
    recall = true_positives_count / true_anomalies if true_anomalies > 0 else 0

    # Create layout
    layout = go.Layout(
        title=dict(
            text=f"DBSCAN Detection Results<br>"
            f"<sub>Clusters: {n_clusters} | Anomalies: {n_anomalies} | Precision: {precision:.3f} | Recall: {recall:.3f}</sub>",
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
    print(f"      - Clusters: {n_clusters}")
    print(f"      - Anomalies detected: {n_anomalies}")
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

    # Run DBSCAN clustering 
    cluster_labels, n_clusters, n_anomalies = dbscan_clustering(
        embeddings, eps=None, min_samples=4
    )

    points_2d = reduce_dimensions_pca(embeddings)

    create_dbscan_visualization(
        points_2d, cluster_labels, true_labels, sequences, n_clusters, n_anomalies
    )

    print()
    print("DBSCAN-based detection complete!")


if __name__ == "__main__":
    main()
