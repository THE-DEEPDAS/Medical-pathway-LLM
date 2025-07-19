# Remove medical imports and add financial ones
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
import os
import json
import sys
from typing import Optional, Dict, Any

# Remove: sys.path.insert(0, r"F:\Wearables\Medical-RAG-LLM\Data")

# Remove: from insurance_data import insurance_data
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize LLM config
config = {
    'max_new_tokens': 1024,
    'context_length': 1024,
    'temperature': 0.1,
    'top_p': 0.9,
    'stream': False,
    'threads': min(4, int(os.cpu_count() / 2)),
}

# Initialize the financial query prompt template
FINANCIAL_QUERY_PROMPT = """
You are a financial advisor assistant specializing in insurance and banking. Use the following information to answer the query.

Context: {context}
Query: {query}

Please provide a clear and detailed response focusing on the financial information requested.
If comparing products or policies, highlight key differences in features, costs, and benefits.
If the query is not related to available financial information, politely indicate that.

Response:
"""

# Update model path
MODEL_PATH = "C:\Medicine Recommendation System\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Initialize components
try:
    # Use local model directly
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config=config
    )
    print("Successfully loaded local model from:", MODEL_PATH)
    
    # Initialize embeddings with specific kwargs
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    from qdrant_client.http.models import VectorParams  # add this import

    # Initialize Qdrant client
    client = QdrantClient("http://localhost:6333")

    # Simply connect to the collection, don't try to create it
    try:
        # Create vector store for financial documents
        db = Qdrant(
            client=client, 
            embeddings=embeddings,
            collection_name="financial_docs"
        )
        print("Connected to 'financial_docs' collection")
        
        retriever = db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error connecting to collection: {e}")
        raise
    
except Exception as e:
    print(f"Initialization error: {e}")
    print(f"Make sure the model exists at: {MODEL_PATH}")
    raise

# New main endpoint to handle financial queries (used by the frontend)
@app.post("/query_new")
async def process_query_new(request: QueryRequest):
    """Handle financial queries using a simplified and robust logic"""
    try:
        query = request.query.strip()
        if not query: 
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        docs = retriever.invoke(query)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        context_parts = []
        for doc in docs:
            content = getattr(doc, "page_content", None) or str(doc)
            context_parts.append(content)
        context = "\n".join(context_parts)
        
        # Removed insurance_data handling; use context only.
        combined_context = context[:config['context_length']]
        
        prompt = FINANCIAL_QUERY_PROMPT.format(context=combined_context, query=query)
        response = llm.invoke(prompt)
        if not response:
            raise HTTPException(status_code=500, detail="LLM returned no response")
            
        return JSONResponse(content={
            "query": query,
            "response": response.strip()
        })
    except Exception as e:
        print(f"Error in /query_new: {e}")
        # Fixed the missing closing quote and parenthesis below
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# New alias endpoint to support legacy POST requests to "/query"
@app.post("/query")
async def query_alias(request: QueryRequest):
    return await process_query_new(request)

# Add health-check endpoint
@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Add a route to handle favicon.ico requests
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Add helper function to search financial info
def search_financial_info(query: str) -> dict:
    """Search through financial documents"""
    results = []
    query = query.lower()
    
    # Removed insurance_data search block since the file does not exist.
    
    # Add results from vector store
    docs = retriever.invoke(query)
    for doc in docs:
        results.append({
            "type": "document",
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown")
        })
    
    return results

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)