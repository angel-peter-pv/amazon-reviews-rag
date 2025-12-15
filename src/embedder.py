"""
Generates embeddings for review chunks and stores them with metadata for FAISS-based retrieval
    Loads pre chunked data
    Computes embeddings using a SentenceTransformer model
    Stores embeddings and associated metadata for FAISS based retrieval
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import mlflow

CHUNK_FILE="data/chunks/electronics_chunks_250w_50ov.jsonl"
EMBEDDING_FILE="data/embeddings/electronics_embeddings.npy"
METADATA_FILE="data/embeddings/electronics_metadata.jsonl"


#Takes a filename - Opens the file - Reads each line - 
# Converts JSON string - Python dict - Returns a list of these dicts
def load_chunks(chunk_file):
    with open(chunk_file,"r") as f:
        return[json.loads(line) for line in f]

# Saves the embeddings matrix and chunk metadata
def save_embedding(embeddings,metadata):
    os.makedirs("data/embeddings",exist_ok=True)
    np.save(EMBEDDING_FILE,embeddings)

    with open(METADATA_FILE,"w") as f:
        for m in metadata:
            f.write(json.dumps(m)+"\n")


def main():


    # Start MLflow run
    with mlflow.start_run(run_name="embedding_generation"):
        # Log important parameters
        mlflow.log_param("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        mlflow.log_param("chunk_size", 250)
        mlflow.log_param("overlap", 50)


        print("loading chunks...")
        chunks=load_chunks(CHUNK_FILE)   #list of dicts = chunks
        print(f"Loaded {len(chunks)} chunks")
        mlflow.log_metric("num_chunks", len(chunks))
 

        print("Now loading embedding model...")
        model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        embeddings=[]
        metadata=[]

        import time
        start_time = time.time()    

        print("computing embeddings...")
        for chunk in tqdm(chunks):
            text=chunk["chunk_text"]
            emb=model.encode(text,convert_to_numpy= True)
            embeddings.append(emb)

            metadata.append({


                "chunk_id": chunk["chunk_id"],
                "asin": chunk["asin"],
                "parent_asin": chunk.get("parent_asin"),
                "product_name": chunk.get("product_name"),
                "review_title": chunk.get("review_title"),
                "rating": chunk["rating"],
                "timestamp": chunk["timestamp"],
                "helpful_vote": chunk["helpful_vote"],
                "verified_purchase": chunk["verified_purchase"],
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
                "chunk_text": chunk["chunk_text"]
            }
            )

        embeddings=np.vstack(embeddings)
        end_time = time.time()
        mlflow.log_metric("time_taken_seconds", end_time - start_time)
        mlflow.log_metric("embedding_dim", embeddings.shape[1])

        print("now saving embeddings...")
        save_embedding(embeddings,metadata)

        print("Saving complete.")
        print(f"Embeddings saved to {EMBEDDING_FILE}")
        print(F"Metadata saved to {METADATA_FILE}")

        # Log artifacts 
        mlflow.log_artifact(EMBEDDING_FILE)
        mlflow.log_artifact(METADATA_FILE)

if __name__=="__main__":
    main()
