"""
Evaluates the quality of the retriever by manually measuring how relevant
the top-k retrieved chunks are for sample queries.
"""
from retriever import retrieve

# ---------------------------
# Test queries for evaluation
# ---------------------------

TEST_QUERIES = [

    "How clear is the image quality of these binoculars?",
    "Which binoculars perform better: the ones with lens coating issues or the fully coated model?",

    "Is the included tripod stable enough for photography?",
    "Which tripod is sturdier: the lightweight travel tripod or the one included with the binocular kit?",

    "Does the phone adapter stay securely attached to the phone during use?",
    "Which phone adapter feels more durable: the one with drilling issues or the older model mentioned by reviewers?",

    "How durable are the headphones after long-term daily use?",
    "Which headphones are better according to reviewers: the newer bass-heavy model or the older pair they previously owned?",

    "Is this backpack durable enough to carry camera equipment safely?",

    "Do reviewers say this product is worth the price paid?"
]


# ---------------------------
# Evaluate retriever quality
# ---------------------------
def evaluate_retriever(k=3):
    total_queries = len(TEST_QUERIES)
    total_relevance_score = 0

    print("\n======================")
    print(" RETRIEVAL EVALUATION ")
    print("======================")

    for query in TEST_QUERIES:
        print("\n------------------------------------")
        print(f"Query: {query}")

        results, distances, ids = retrieve(query, k=k)

        # Print top results
        print("\nTop Results:")
        for i, r in enumerate(results):
            print(f"\nResult {i+1}  (Distance: {distances[i]:.4f})")
            print(r["chunk_text"][:300], "...")
        
        # Ask for manual relevance input
        print("\nHow many of the above results were relevant?")
        print(f"Enter a number 0–{k}: ")

        while True:
            try:
                score = int(input("> "))
                if 0 <= score <= k:
                    break
                else:
                    print(f"Please enter a valid number between 0 and {k}.")
            except ValueError:
                print("Please enter a valid integer.")
        
        total_relevance_score += score

    # --------------------
    # Compute Precision@K
    # ---------------------
    precision_at_k = total_relevance_score / (total_queries * k)

    print("\n====================================")
    print(f" FINAL PRECISION@{k}: {precision_at_k:.2f}")
    print("====================================")


if __name__ == "__main__":
    evaluate_retriever(k=3)
