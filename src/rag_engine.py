"""
Implements the RAG answer generation logic using OpenAI (GPT-4o-mini) while keeping the retrieved context within a token limit.
"""

import re
import textwrap
import os

# from retriever import retrieve
from src.retriever import retrieve

from typing import List, Dict
import tiktoken 


#config
MAX_CONTEXT_TOKENS = 3000   

DEFAULT_K=5
SYSTEM_INSTRUCTIONS = (
    "You are an assistant that answers product questions using ONLY the provided context.\n"
    "Cite sources inline like [source:ASIN:chunk_id].\n"
    "If the answer is not supported by the context, respond: \"I don't know based on the provided information.\"\n"
    "Do NOT guess product names unless they appear in the context.\n"   
    "Keep answers short and factual. "
    "Provide a short answer and then list sources used."
)
MODEL_NAME = "gpt-4o-mini"
MAX_RESPONSE_TOKENS = 512

#count no:of tokens
tokenizer=tiktoken.encoding_for_model(MODEL_NAME)
def count_tokens(text:str)->int:
    return len(tokenizer.encode(text))


#clean the retrived text before passing to llm
def clean_text(text:str)->str:
    text = re.sub(r"\[\[(VIDEOID|ASIN)[^\]]*\]\]", "", text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

#Turn retrieved metadata dicts into a single context string.
def build_context(retrieved:List[Dict],max_tokens: int=MAX_CONTEXT_TOKENS)->str:
    ctx_parts=[]
    total_tokens=0
    for r in retrieved:
        product = r.get("product_name", "Unknown Product")     
        #to make a tag so that i can trace source : so that i know its factual
        provenance=f"[source:{r.get('asin','unknown')}:{r.get('chunk_id','unknown')}]"
        
        #clean chunk text
        txt=clean_text(r.get("chunk_text",""))

        #Combine provenance tag +product name  + cleaned text
        piece=(
            f"{provenance}\n"
            f"Product: {product}\n"             
            f"{txt}\n"
               )
        piece_tokens=count_tokens(piece)

        #tokens budget check
        if total_tokens + piece_tokens > max_tokens:
            remaining=max_tokens - total_tokens

             # allow partial chunk if meaningful
            if remaining > 50:
                truncated_tokens=tokenizer.encode(piece)[:remaining]
                truncated=tokenizer.decode(truncated_tokens) + "....\n"
                ctx_parts.append(truncated)
            break

        ctx_parts.append(piece)  
        total_tokens+=piece_tokens
    context="\n".join(ctx_parts)
    return context 
    

#Assembling the final prompt string that is passed to the LLM
def build_prompt(question:str,context:str,system_instructions:str=SYSTEM_INSTRUCTIONS)->str:
    header="Use the following retrieved review excerpts as factual context:\n\n"
    prompt=f"{system_instructions}\n\n{header}{context}\n\nQuestion: {question}\nAnswer concisely and cite sources."
    return prompt

#calling llm
def call_llm_openai(prompt:str,model=MODEL_NAME)->str:
    import openai
    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in env")
    openai.api_key=api_key
    response = openai.ChatCompletion.create(

        model=model,
        messages=[
            {"role":"system","content":"You are a helpful assistant"},
            {"role":"user","content":prompt}
                  ],
        temperature=0,
        max_tokens=MAX_RESPONSE_TOKENS
        )
    
    return response["choices"][0]["message"]["content"].strip()


#generate ans by retrieving chunks, building context and prompt, calling the LLM, 
#returning the final answer with source metadata.
def generate_answer(question:str,k=DEFAULT_K,llm_model="gpt-4o-mini")->Dict:
    retrieved, distances,ids=retrieve(question,k=k)
    if not retrieved:
        return {
            "question": question,
            "answer": "I don't know based on the provided information.",
            "sources": [],
            "prompt": ""
        }
    context=build_context(retrieved)
    prompt=build_prompt(question,context)
    answer=call_llm_openai(prompt,model=llm_model)

    enc = tiktoken.encoding_for_model(llm_model)
    print("ANSWER TOKENS:", len(enc.encode(answer)))
    sources=[
        {"asin":r.get("asin"),
         "chunk_id":r.get("chunk_id"),
         "product_name": r.get("product_name"),      
         "distance":float(distances[i])}
        for i,r in enumerate(retrieved)
    ]
    return {"question":question,"answer":answer,"sources":sources,"prompt":prompt}



def format_result(result:Dict)->str:
    lines=[]
    lines.append("ANSWER:\n"+textwrap.fill(result["answer"],400))
    lines.append("\nSOURCES:")
    for s in result["sources"]:
        lines.append(f"ASIN:{s['asin']}  chunk:{s['chunk_id']}  product:{s.get('product_name','Unknown Product')}  distance:{s['distance']:.4f}")

    return "\n".join(lines)

if __name__ == "__main__":
    q="which tripod is good and stable for photography?"
    res = generate_answer(q, k=3, llm_model="gpt-4o-mini")
    print(format_result(res))









