""" Chunking script 
        Reads cleaned dataset in jsonl format.
        Combines product name, review title, and review text
        Splits long reviews into overlapping word based chunks
        Saves chunked data with metadata
"""

import json
import uuid
from pathlib import Path

INPUT = "data/processed/electronics_50k_clean.jsonl" 
OUTPUT = "data/chunks/electronics_chunks_250w_50ov.jsonl"

CHUNK_WORD_LIMIT = 250
CHUNK_OVERLAP = 50
MIN_WORDS = 40

def words(text):
    return text.split()

def make_chunks(word_list,chunk_size,overlap):
    i = 0
    n = len(word_list)
    step = chunk_size - overlap
    if step<= 0:
        raise ValueError("chunk size must be larger than overlap")
    
    while i<n:
        start=i
        end= min(i+chunk_size,n)
        chunk_words= word_list[start:end]
        yield start,end,chunk_words
            #Yield (start_idx, end_idx, chunk_words) for text represented as words_list

        if end==n:
            break   
        i+=step


def main():
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    out_count = 0
    doc_count = 0

    with open(INPUT, "r", encoding="utf-8") as infile, \
         open(OUTPUT, "w", encoding="utf-8") as outfile:
        

        for line in infile:
            doc_count+=1
            doc= json.loads(line)

            product_name = doc.get("product_name", "") or ""
            review_title = doc.get("title", "") or ""
            review_text  = doc.get("text", "") or ""

            asin= doc.get("asin")
            parent_asin = doc.get("parent_asin")
            rating=doc.get("rating")
            timestamp=doc.get("timestamp")
            helpful=doc.get("helpful_vote")
            verified=doc.get("verified_purchase")

            # Using title + text as source for chunking 
            combined= (
                f"Product: {product_name}. "
                f"Review Title: {review_title}. "
                f"Review: {review_text}"
            ).strip()

            w=words(combined)

            #skip short documents
            if len(w)<MIN_WORDS:
                continue


            for start,end, chunk_words in make_chunks(w, CHUNK_WORD_LIMIT, CHUNK_OVERLAP):
                chunk_text =  " ".join(chunk_words).strip()

                #skip tiny chunks after joining them 
                if len(chunk_text)< MIN_WORDS:
                    continue

                chunk={
                    "chunk_id": str(uuid.uuid4()),
                    "asin": asin,
                    "parent_asin": parent_asin,         
                    "product_name": product_name,
                    "rating": rating,
                    "timestamp": timestamp,
                    "helpful_vote": helpful,
                    "verified_purchase": verified,
                    "start_word": start,
                    "end_word": end,
                    "chunk_text": chunk_text

                }

                outfile.write(json.dumps(chunk,ensure_ascii=False)+ "\n")
                out_count+=1


    print(f"Documents processed: {doc_count}")
    print(f"Chunks created: {out_count}")
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()






        



























