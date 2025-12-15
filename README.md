# Amazon Review RAG System



## Project Overview

This project implements an end to end Retrieval Augmented Generation (RAG) system for question answering over Amazon product reviews.
The system combines semantic search (FAISS) with a Large Language Model (LLM) to generate grounded evidence based answers.
The primary focus of this project is learning and understanding how to build a complete RAG pipeline using a real world dataset.

The project covers the full lifecycle of a RAG system, including:
- Text preprocessing and dataset preparation
- Text chunking 
- Embedding generation
- Vector indexing and similarity search
- Retrieval logic
- Prompt construction for RAG
- API deployment using FastAPI

The pipeline is implemented using OpenAI GPT4 Mini for inference and is exposed via a FastAPI service.
In addition, this project explores running the same RAG workflow using a local LLM (Mistral via Ollama).


## Problem Statement

Large Language Models on their own do not have access to external or domain specific data and may generate hallucinated answers.
This project addresses that limitation by:
- Retrieving relevant customer reviews using vector similarity search
- Providing the retrieved content as context to the LLM
- Ensuring answers are based on user generated data


## System Workflow
```
User Query
    ↓
Query Embedding
    ↓
FAISS Vector Search (Top-k Review Chunks)
    ↓
Context Assembly
    ↓
LLM Answer Generation
    ↓
Answer Returned via API
```


## Dataset

This project uses the **Amazon Reviews 2023 dataset** published by the McAuley Lab: https://amazon-reviews-2023.github.io/
For experimentation and faster iteration, a subset of the dataset is extracted and used in this project.


## How to Run the Project

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Set required environment variables
```export OPENAI_API_KEY="your_api_key"```
5. Start the FastAPI server
```uvicorn src.api:app --reload```

   
## Using API
Once the FastAPI server is running, open the Swagger UI:
```http://127.0.0.1:8000/docs```
  
### Ask a Question

- Endpoint:
`POST /ask`
- Request body:

```json
{
  "question": "Is this tripod stable for outdoor photography?"
}


   





      
