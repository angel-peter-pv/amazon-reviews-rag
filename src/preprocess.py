"""
Cleans  Amazon review text before chunking and embedding in the RAG pipeline.

"""
import json
import re

input_path = "data/raw/electronics_50k_with_product_name.jsonl"
output_path = "data/processed/electronics_50k_clean.jsonl"


def clean_text(text:str)->str:
    text = text or ""
    text = re.sub(r"<.*?>","", text) #Find anything that starts with < and ends with >, but stop at the first > for removing html tags
    text = text.replace("\n"," ") #remove newlines
    text = re.sub(r"\s+"," ",text) #Replace any amount of whitespace with a single space.
    return text.strip()


with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path,"w",encoding="utf-8") as outfile:
    
    for line in infile:
        review = json.loads(line)

        #extracting fields we want
        cleaned_review = {
            "asin": review.get("asin"),
            "parent_asin": review.get("parent_asin"), 
            "product_name": clean_text(review.get("product_name", "")),  
            "rating": review.get("rating"),
            "title": clean_text(review.get("title", "")),
            "text": clean_text(review.get("text", "")),
            "timestamp": review.get("timestamp"),
            "helpful_vote": review.get("helpful_vote"),
            "verified_purchase": review.get("verified_purchase"),
            
            }
        
        if len(cleaned_review["text"])<30:
            continue

        outfile.write(json.dumps(cleaned_review)+"\n")


print(f" Preprocessing complete!\nSaved cleaned data → {output_path}")        