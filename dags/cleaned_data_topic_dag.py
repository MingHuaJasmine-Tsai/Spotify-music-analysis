from airflow import DAG
from airflow.decorators import task
from airflow.hooks.base import BaseHook
from datetime import datetime, timedelta
import pandas as pd
from google.oauth2 import service_account

PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
INPUT_PREFIX = "cleaned/cleaned_20251020"
OUTPUT_PREFIX = "cleaned/cleaned_20251020/topics"
GCP_CONN_ID = "gcp_conn"

# Artist CSV files to process
ARTIST_FILES = [
    "comments_Doja_Cat_20251017_cleaned.csv",
    "comments_Ed_Sheeran_20251017_cleaned.csv",
    "comments_Tate_McRae_20251017_cleaned.csv"
]

def _get_gcp_creds():
    """Retrieve GCP service account credentials from Airflow Connection."""
    conn = BaseHook.get_connection(GCP_CONN_ID)
    info = conn.extra_dejson.get("extra__google_cloud_platform__keyfile_dict")
    if not info:
        raise RuntimeError("Missing keyfile_dict in GCP connection extras.")
    return service_account.Credentials.from_service_account_info(info)

default_args = {
    "owner": "ran",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="cleaned_data_topic_modeling",
    schedule=None,  # Manual trigger only
    start_date=datetime(2025, 12, 7),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["topic-modeling", "cleaned-data"],
) as dag:

    @task()
    def read_comments_from_gcs(filename: str):
        """Read comments from GCS cleaned data folder."""
        from google.cloud import storage
        from io import StringIO

        creds = _get_gcp_creds()
        client = storage.Client(credentials=creds, project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        blob_path = f"{INPUT_PREFIX}/{filename}"
        print(f"📁 Reading file: gs://{BUCKET_NAME}/{blob_path}")

        blob = bucket.blob(blob_path)
        if not blob.exists():
            raise FileNotFoundError(f"❌ File not found: gs://{BUCKET_NAME}/{blob_path}")

        df = pd.read_csv(StringIO(blob.download_as_text()))
        print(f"✅ Loaded {len(df)} comments from {filename}")

        # Keep necessary columns (adjust based on your cleaned CSV structure)
        keep = [c for c in ["video_id", "author", "text", "like_count", "published_at", "clean_text"]
                if c in df.columns]
        df = df[keep].dropna(subset=["text"] if "text" in df.columns else []).reset_index(drop=True)

        return {"filename": filename, "data": df.to_json(orient="records")}

    @task()
    def clean_and_sample(payload: dict, sample_size: int = 3000):
        """Clean text and sample if needed (may be redundant for pre-cleaned data)."""
        import json, re, string
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        import nltk

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        filename = payload["filename"]
        df = pd.DataFrame(json.loads(payload["data"]))

        # If data is already cleaned and has clean_text column, use it
        if "clean_text" in df.columns and df["clean_text"].notna().sum() > 0:
            print("✅ Using existing clean_text column")
            df = df[df["clean_text"].notna() & (df["clean_text"].str.len() > 1)].reset_index(drop=True)
        else:
            # Otherwise, clean the text column
            print("🔄 Cleaning text column")
            stop_words = set(stopwords.words("english"))
            punct = f"[{re.escape(string.punctuation)}]"

            def clean_text(t):
                t = str(t)
                t = re.sub(r"http\S+", "", t)
                t = re.sub(r"@\S+", "", t)
                t = re.sub(punct, " ", t)
                toks = [w.lower() for w in word_tokenize(t) if w.isalpha()]
                toks = [w for w in toks if w not in stop_words and len(w) > 1]
                return " ".join(toks)

            df["clean_text"] = df["text"].apply(clean_text)
            df = df[df["clean_text"].str.len() > 1].reset_index(drop=True)

        # Sample if needed
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
            print(f"✅ Sampled {sample_size} comments")
        else:
            print(f"✅ Using all {len(df)} comments")

        return {"filename": filename, "data": df.to_json(orient="records")}

    @task()
    def embed_and_umap(payload: dict):
        """Generate embeddings and reduce dimensions with UMAP."""
        import json
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import umap

        filename = payload["filename"]
        df = pd.DataFrame(json.loads(payload["data"]))

        print(f"🔄 Generating embeddings for {filename}...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            df["clean_text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False
        )

        print(f"🔄 Reducing dimensions with UMAP for {filename}...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        xy = reducer.fit_transform(embeddings)

        df["x"] = xy[:, 0]
        df["y"] = xy[:, 1]
        df["emb_norm"] = np.linalg.norm(embeddings, axis=1)

        print(f"✅ Embeddings complete for {filename}")
        return {"filename": filename, "data": df.to_json(orient="records")}

    @task()
    def topic_model(payload: dict, n_topics: int = 6):
        """Perform topic modeling using NMF."""
        import json
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
        import numpy as np

        filename = payload["filename"]
        df = pd.DataFrame(json.loads(payload["data"]))

        print(f"🔄 Running topic modeling on {filename} with {n_topics} topics...")
        vec = TfidfVectorizer(max_features=5000, min_df=5)
        X = vec.fit_transform(df["clean_text"])

        nmf = NMF(n_components=n_topics, random_state=42, init="nndsvd", max_iter=400)
        W = nmf.fit_transform(X)
        H = nmf.components_
        vocab = vec.get_feature_names_out()

        # Extract keywords for each topic
        topics = []
        for k, comp in enumerate(H):
            words = [vocab[i] for i in comp.argsort()[-10:][::-1]]
            topics.append({"topic_id": k, "keywords": ", ".join(words)})

        df["topic"] = W.argmax(axis=1)
        df["topic_score"] = W.max(axis=1)

        print(f"✅ Topic modeling complete for {filename}")
        result = {
            "filename": filename,
            "topics": topics,
            "df": df.to_json(orient="records")
        }
        return result

    @task()
    def upload_result(result: dict):
        """Upload topic modeling results to GCS."""
        import json
        from google.cloud import storage

        filename = result["filename"]
        # Extract artist name from filename (e.g., "comments_Doja_Cat_20251017_cleaned.csv" -> "Doja_Cat")
        artist_name = filename.replace("comments_", "").replace("_cleaned.csv", "").rsplit("_", 1)[0]
        date_tag = filename.replace("comments_", "").replace("_cleaned.csv", "").rsplit("_", 1)[-1]

        topics = pd.DataFrame(result["topics"])
        df = pd.DataFrame(json.loads(result["df"]))

        creds = _get_gcp_creds()
        client = storage.Client(credentials=creds, project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        # Upload topics keywords
        topics_path = f"{OUTPUT_PREFIX}/topics_{artist_name}_{date_tag}.csv"
        bucket.blob(topics_path).upload_from_string(topics.to_csv(index=False), "text/csv")
        print(f"✅ Uploaded: gs://{BUCKET_NAME}/{topics_path}")

        # Upload embeddings with topic labels
        out_cols = [c for c in ["video_id", "author", "like_count", "published_at",
                               "clean_text", "x", "y", "topic", "topic_score", "emb_norm"]
                    if c in df.columns]
        embeddings_path = f"{OUTPUT_PREFIX}/embeddings_{artist_name}_{date_tag}.csv"
        bucket.blob(embeddings_path).upload_from_string(df[out_cols].to_csv(index=False), "text/csv")
        print(f"✅ Uploaded: gs://{BUCKET_NAME}/{embeddings_path}")

        return {
            "artist": artist_name,
            "topics_path": topics_path,
            "embeddings_path": embeddings_path
        }

    # Create tasks for each artist
    for artist_file in ARTIST_FILES:
        raw = read_comments_from_gcs(artist_file)
        cleaned = clean_and_sample(raw)
        embedded = embed_and_umap(cleaned)
        modeled = topic_model(embedded)
        upload_result(modeled)
