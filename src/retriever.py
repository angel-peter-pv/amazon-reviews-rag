"""
Uses a FAISS index to retrieve the most relevant review chunks for a user query.
    Loads FAISS index and associated metadata    
    Embeds the user query using a sentence transformer
    Retrieves the top k most similar review chunks
    Returns matched chunks with distance scores
"""
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_INDEX_FILE="data/faiss/electronics.index"
METADATA_FILE="data/embeddings/electronics_metadata.jsonl"

#LOAD FAISS INDEX
def load_faiss_index(path=FAISS_INDEX_FILE):
    return faiss.read_index(path)

#load metadata
def load_metadata(path=METADATA_FILE):
    metadata=[]
    with open(path,"r") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


#embed user query
def embed_query(query):
    model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    emb=model.encode(query,convert_to_numpy=True)
    return emb.astype(np.float32).reshape(1,-1)

#search faiss
def search_faiss(index,query_vector,k=5):
    distances, ids =index.search(query_vector,k)
    return distances[0],ids[0]

# Map FAISS IDs to metadata rows
def get_results(ids, metadata):
    results = []
    for id_ in ids:
        result = metadata[id_]   
        result["product_name"] = result.get("product_name", "Unknown Product")
        result["parent_asin"] = result.get("parent_asin", None)
        result["review_title"] = result.get("review_title", "")

        results.append(result)

    return results

#Full retrieval pipeline
def retrieve(query,k=5):
    index=load_faiss_index()
    metadata=load_metadata()
    query_vector=embed_query(query)  
    distances,ids=search_faiss(index,query_vector,k)
    results=get_results(ids,metadata)  
    return results, distances, ids


if __name__ == "__main__":
    q="which earphone is better?"
    results, distances, ids =retrieve(q,k=3)

    print("Query:", q)
    print("\nTop Results:")
    for r, d in zip(results, distances):
        print("\nDistance:", d)
        print("Product:", r["product_name"])              
        print("Chunk text:", r["chunk_text"][:200], "...")






