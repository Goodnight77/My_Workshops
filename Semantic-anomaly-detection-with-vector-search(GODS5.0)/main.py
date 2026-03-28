import os
from dotenv import load_dotenv


def run_script(script_name: str, description: str) -> bool:
    print(f"Running {description}...")

    try:
        result = None
        if script_name == "ingest":
            import ingest

            result = ingest.main()
        elif script_name == "embed_and_ingest":
            import embed_and_ingest

            result = embed_and_ingest.main()
        elif script_name == "detect_distance":
            import detect_distance

            result = detect_distance.main()
        elif script_name == "detect_dbscan":
            import detect_dbscan

            result = detect_dbscan.main()
        else:
            print(f"Unknown script: {script_name}")
            return False

        # Check if script returned False (failure)
        if result is False:
            print(f"{description} failed!")
            print()
            return False

        print(f"{description} completed successfully!")
        print()
        return True

    except Exception as e:
        print(f"{description} failed: {e}")
        print()
        return False


def check_requirements() -> bool:
    print("Checking requirements...")

    if not os.path.exists(".env"):
        print("Error: .env file not found!")
        print("Create .env with OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY")
        return False

    load_dotenv()

    required_vars = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        return False

    print("Requirements check passed!")
    print()
    return True


def main():
    print("=" * 70)
    print("HDFS ANOMALY DETECTION PIPELINE")
    print("=" * 70)
    print("Dataset: 2000 points (80% normal, 20% abnormal)")
    print("Model: OpenAI text-embedding-3-small (1536D)")

    if not check_requirements():
        return

    if not run_script("ingest", "Data Ingestion"):
        print("Pipeline failed at data ingestion step")
        return

    if not run_script("embed_and_ingest", "Embedding Creation and Qdrant Upload"):
        print("Pipeline failed at embedding step")
        return

    detection_method = os.getenv("DETECTION_METHOD", "distance").lower()

    if detection_method == "dbscan":
        if not run_script("detect_dbscan", "DBSCAN-based Anomaly Detection"):
            print("Pipeline failed at DBSCAN detection step")
            return
    else:
        if not run_script("detect_distance", "Distance-based Anomaly Detection"):
            print("Pipeline failed at distance detection step")
            return

    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Output Files:")
    if detection_method == "dbscan":
        print("   visualization/dbscan_detection_results.html")
    else:
        print("   visualization/distance_detection_results.html")
    print("   data/balanced_dataset.txt")

if __name__ == "__main__":
    main()
