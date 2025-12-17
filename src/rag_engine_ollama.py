"""
Implements the RAG answer generation logic using Ollama by combining retrieval results with an LLM prompt.

"""

import re
import textwrap
import os
import requests
import mlflow
import time     

# from retriever import retrieve

from src.retriever import retrieve
from typing import List, Dict


#config
 
DEFAULT_K=5
SYSTEM_INSTRUCTIONS = (
    "You are an assistant that answers product questions using ONLY the provided context.\n"
    "Cite sources inline like [source:ASIN:chunk_id].\n"
    "If the answer is not supported by the context, respond: \"I don't know based on the provided information.\"\n"
    "Do NOT guess product names unless they appear in the context.\n"   
    "Keep answers short and factual. "
    "Provide a short answer and then list sources used."
)
OLLAMA_MODEL = "mistral"
mlflow.set_experiment("amazon-rag-latency")


#clean the retrived text before passing to llm
def clean_text(text:str)->str:
    text = re.sub(r"\[\[(VIDEOID|ASIN)[^\]]*\]\]", "", text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

#Turn retrieved metadata dicts into a single context string.
def build_context(retrieved:List[Dict])->str:
    ctx_parts=[]
    
    for r in retrieved:
        product = r.get("product_name", "Unknown Product")     
        provenance=f"[source:{r.get('asin','unknown')}:{r.get('chunk_id','unknown')}]"
        
        #clean chunk text
        txt=clean_text(r.get("chunk_text",""))

        #Combine provenance tag +product name  + cleaned text
        piece=(
            f"{provenance}\n"
            f"Product: {product}\n"             
            f"{txt}\n"
               )
        ctx_parts.append(piece)
        context="\n".join(ctx_parts)
    return context 
    

#Assembling the final prompt string that is passed to the LLM
def build_prompt(question:str,context:str,system_instructions:str=SYSTEM_INSTRUCTIONS)->str:
    header="Use the following retrieved review excerpts as factual context:\n\n"
    prompt=f"{system_instructions}\n\n{header}{context}\n\nQuestion: {question}\nAnswer concisely and cite sources."
    return prompt

#calling ollama llm
def call_llm_ollama(prompt:str,model=OLLAMA_MODEL)->str:
    response=requests.post("http://localhost:11434/api/generate",
                           json={"model": model,"prompt": prompt,"stream": False},
                           timeout=120

    )
    response.raise_for_status()
    return response.json()["response"].strip()


#generate ans by retrieving chunks, building context and prompt, calling the LLM, 
#returning the final answer with source metadata.
def generate_answer(question:str,k=DEFAULT_K)->Dict:

    total_start_time = time.time()
    with mlflow.start_run(nested=True):
        mlflow.log_param("llm_model", OLLAMA_MODEL)
        mlflow.log_param("top_k", k) 

        retrieval_start = time.time()
        retrieved, distances,ids=retrieve(question,k=k)
        retrieval_time = time.time() - retrieval_start
        mlflow.log_metric("retrieval_time_ms", retrieval_time * 1000)

        if not retrieved:
            mlflow.log_metric("total_response_time_ms",(time.time() - total_start_time) * 1000)
            return {
            "question": question,
            "answer": "I don't know based on the provided information.",
            "sources": []
            }
        context=build_context(retrieved)
        prompt=build_prompt(question,context)
    
        llm_start = time.time()

        answer=call_llm_ollama(prompt)

        llm_time = time.time() - llm_start  
        mlflow.log_metric("llm_inference_time_ms", llm_time * 1000) 
        total_time = time.time() - total_start_time  
        mlflow.log_metric("total_response_time_ms", total_time * 1000) 

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
    res = generate_answer(q, k=3)
    print(format_result(res))









