# RUNNING IN TENSORFLOW ENV 
# coher, or reranking, long-context-reorder method to abound missing in the middle 
# uvicorn rag:app
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings  # Changed from SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import os
import json
from health_simulator import HealthMetricsSimulator
from pydantic import BaseModel, EmailStr
from typing import Optional
from ingest import ingest_docs  # Add this at the top with other imports
from db.user_store import UserStore
from models.user import UserProfile

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"  # Change to a publicly available model
model_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"   # Specific GGUF file

# Add model path check
def get_model_path():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "models", model_file)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Attempting to download model automatically...")
        try:
            from download_model import download_file
            model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            download_file(model_url, model_file)
        except Exception as e:
            print(f"Auto-download failed: {e}")
            print("\nPlease follow these steps:")
            print("1. Create a 'models' directory in your project")
            print("2. Download the model from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
            print("3. Place the downloaded file in the models directory")
            raise FileNotFoundError(f"Please download the model file {model_file}")
    
    return model_path

# Add after the get_model_path function
def ensure_model_directory():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_path, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

config = {
    'max_new_tokens': 2048,
    'context_length': 2048,
    'repetition_penalty': 1.1,
    'temperature': 0.2,
    'top_k': 50,
    'top_p': 1,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

# Call this before initializing the model
ensure_model_directory()

try:
    model_path = get_model_path()
    llm = CTransformers(
        model=model_path,
        model_type="mistral",
        lib="avx2",
        **config
    )
    print("LLM Initialized successfully...")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("\nPlease follow these steps:")
    print("1. Create a 'models' directory in your project")
    print("2. Download the model from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
    print("3. Place the downloaded file in the models directory")
    raise

print("LLM Initialized....")

def format_health_analysis(metrics):
    """Format health metrics analysis"""
    status = []
    
    # Heart rate analysis
    hr = int(metrics['heart_rate'])
    if hr < 60: status.append("Heart rate is below normal range")
    elif hr > 100: status.append("Heart rate is above normal range")
    
    # Blood pressure analysis
    bp = metrics['blood_pressure'].split('/')
    systolic, diastolic = int(bp[0]), int(bp[1])
    if systolic > 130 or diastolic > 80:
        status.append("Blood pressure is elevated")
    
    # Cholesterol analysis
    if metrics['cholesterol'] > 200:
        status.append("Cholesterol levels are elevated")
    
    return " | ".join(status) if status else "All metrics within normal range"

prompt_template = """
Use the following pieces of medical information and current health metrics to provide accurate responses to the user's questions.

Context: {context}
Question: {question}

Current Health Metrics:
- Heart Rate: {heart_rate} bpm
- Blood Pressure: {blood_pressure}
- Sleep Duration: {sleep_hours} hours
- Sleep Quality: {sleep_quality}
- Calories Burned: {calories_burned} kcal
- Cholesterol: {cholesterol} mg/dL

Patient Details:
- Age: {age}
- Gender: {gender}

Health Analysis: {health_analysis}

Please provide a comprehensive answer that:
1. Addresses the specific question
2. Considers the patient's current health metrics
3. Provides relevant medical advice based on both the context and health status
4. Suggests any necessary lifestyle modifications or precautions

Helpful answer:
"""

# Replace the embeddings initialization with:
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embeddings model loaded successfully")
except Exception as e:
    print(f"Error loading embeddings model: {e}")
    raise

url = "http://localhost:6333"

# Fix client initialization with proper error handling
try:
    client = QdrantClient(url=url, prefer_grpc=False)
    # Test connection
    client.get_collections()
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    print("Make sure Qdrant is running with: docker run -p 6333:6333 qdrant/qdrant")
    raise

# Add after client initialization:
def check_collection_exists():
    try:
        collections = client.get_collections()
        return any(col.name == "medical_db" for col in collections.collections)
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False

# Replace the collection check block with:
def initialize_collection():
    if not check_collection_exists():
        print("Warning: medical_db collection not found!")
        print("Please run ingest.py first to create the collection")
        if not os.path.exists('data'):
            os.makedirs('data')
            print("Created data directory. Please add PDF files and run again.")
            return False
        if ingest_docs():
            print("Successfully created medical_db collection")
            return True
        return False
    return True

# Call this before creating the db instance
if not initialize_collection():
    raise RuntimeError("Failed to initialize medical_db collection")

db = Qdrant(client=client, embeddings=embeddings, collection_name="medical_db")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

class QueryRequest(BaseModel):
    query: str
    age: Optional[int] = None
    gender: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

user_store = UserStore()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/guide", response_class=HTMLResponse)
async def show_guide(request: Request):
    return templates.TemplateResponse("guide.html", {"request": request})

@app.post("/login")
async def login(request: LoginRequest):
    user_id = user_store.verify_login(request.email, request.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user_id}

@app.post("/register_user")
async def register_user(user: UserProfile):
    try:
        user_id = user_store.save_user(user)
        return JSONResponse(
            content={"status": "success", "user_id": user_id},
            status_code=201
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=400
        )

@app.post("/get_response")
async def get_response(
    request: Request,
    query: str = Form(None),
    user_id: str = Form(None)
):
    # Validate required fields
    errors = []
    if not query:
        errors.append("Question is required")
    if not user_id:
        errors.append("User ID is required")
    
    if errors:
        return JSONResponse(
            content={
                "error": "Validation Error",
                "details": errors,
                "help": "Please visit /guide for usage instructions"
            },
            status_code=422
        )

    try:
        user = user_store.get_user(user_id)
        if not user:
            return JSONResponse(
                content={
                    "error": "User not found",
                    "help": "Please register first at /guide"
                },
                status_code=404
            )
        
        # Generate health metrics using profile
        health_metrics = HealthMetricsSimulator.generate_metrics(user)
        
        # Create the question input with combined static and real-time data
        question_input = {
            "question": query,
            "context": "",
            **health_metrics["static"],
            **health_metrics["real_time"]
        }
        
        # Get response from QA chain
        response = qa(question_input)
        
        return JSONResponse(content={
            "answer": response['result'],
            "source_document": response['source_documents'][0].page_content if response['source_documents'] else "",
            "health_metrics": health_metrics,
            "user_profile": user.dict()
        })
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )