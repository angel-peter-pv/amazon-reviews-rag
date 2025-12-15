"""
Builds and stores a FAISS index from precomputed embeddings for  semantic retrieval
    Loads embedding vectors
    Builds a FAISS IndexFlatL2 index
    Saves the index for retrieval
    Logs build metrics and artifacts using MLflow
"""
import os
import time
import faiss
import numpy as np
import mlflow

EMBEDDING_FILE="data/embeddings/electronics_embeddings.npy"
FAISS_DIR ="data/faiss"
FAISS_INDEX_FILE=os.path.join(FAISS_DIR,"electronics.index")

#  Load embeddings from a .npy file and return a contiguous float32 NumPy array.
def load_embeddings(path):
    emb=np.load(path)

    if emb.dtype != np.float32:
        emb=emb.astype(np.float32)

    if not emb.flags['C_CONTIGUOUS']:
        emb=np.ascontiguousarray(emb)

    return emb

def build_faiss_index(embeddings):

    """
    Build an IndexFlatL2 FAISS index and add embeddings.
    Returns the index object.
    """

    dim=embeddings.shape[1]
    index=faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
    
def save_faiss_index(index,path):
    """
    Save the FAISS index to disk, ensuring the directory exists.
    """

    os.makedirs(os.path.dirname(path),exist_ok=True)
    faiss.write_index(index,path)

#load embeddings - build index - save index - track everything in MLflow.
def main():
    with mlflow.start_run(run_name="faiss_index_build"):
        mlflow.log_param("faiss_index_type","Index_Flat_L2")
        mlflow.log_param("embedding_file",EMBEDDING_FILE)

        print("loading embeddings...")
        t0=time.time()
        emb=load_embeddings(EMBEDDING_FILE)
        load_time=time.time()-t0
        print(f"Loaded embeddings shape: {emb.shape}")

        num_vectors=int(emb.shape[0])
        embedding_dim=int(emb.shape[1])
        mlflow.log_metric("num_vectors", num_vectors)
        mlflow.log_metric("embedding_dim", embedding_dim)
        mlflow.log_metric("embeddings_load_seconds", load_time)

        print("Building FAISS index (IndexFlatL2)...")
        t1=time.time()
        index=build_faiss_index(emb)
        build_time=time.time()-t1
        mlflow.log_metric("index_build_seconds", build_time)

        print(f"Index trained: {num_vectors} vectors, dim={embedding_dim}")
        print("saving faiss index...")

        t2=time.time()
        save_faiss_index(index,FAISS_INDEX_FILE)
        save_time=time.time()-t2

        #compute file size
        index_size_bytes=os.path.getsize(FAISS_INDEX_FILE)
        index_size_mb=index_size_bytes/(1024*1024)
        mlflow.log_metric("index_file_size_mb", index_size_mb)
        mlflow.log_metric("faiss_save_seconds", save_time)


        # Log the index file as an MLflow artifact
        mlflow.log_artifact(FAISS_INDEX_FILE)

        total_time=time.time()-t0
        mlflow.log_metric("total_time_seconds", total_time)

        print("FAISS index saved:", FAISS_INDEX_FILE)
        print(f"Index file size: {index_size_mb:.2f} MB")
        print(f"Total time: {total_time:.2f} s")

if __name__=="__main__":
    main()




















