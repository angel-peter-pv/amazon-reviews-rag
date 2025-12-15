"""
Enriches the Amazon reviews dataset by adding product names to each review
to improve traceability during retrieval
"""

import json
metadata_file = "data/raw/meta_Electronics.jsonl"            
reviews_file = "data/raw/electronics_50k.jsonl"          
output_file = "data/raw/electronics_50k_with_product_name.jsonl"  


# Build mapping: parent_asin to product title
asin_to_title = {}
with open(metadata_file, "r") as f:
    for line in f:
        item = json.loads(line.strip())
        parent_asin = item.get("parent_asin")
        title = item.get("title")
    
        if parent_asin and title:
     
            asin_to_title[parent_asin] = title


#Go through review dataset and enrich every line with title

with open(reviews_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        review = json.loads(line.strip())

        # parent_asin for matching
        parent_asin = review.get("parent_asin")
        asin = review.get("asin")

        # Find product title using parent_asin first
        product_name = None
        if parent_asin:
            product_name = asin_to_title.get(parent_asin)

        # match by asin if parent_asin missing
        if not product_name and asin:
            product_name = asin_to_title.get(asin)

        #Unknown Product if no match
        if not product_name:
            product_name = "Unknown Product"

        # Inject product_name
        review["product_name"] = product_name

        fout.write(json.dumps(review) + "\n")


