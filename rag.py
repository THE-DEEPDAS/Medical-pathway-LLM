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
from typing import Optional, List, Dict
from ingest import ingest_docs  # Add this at the top with other imports
from db.user_store import UserStore
from models.user import UserProfile
from datetime import datetime
import random
import numpy as np
from custom_pathway import PathwayAnalyzer  # Add this import
from langchain.prompts import PromptTemplate  # Fix import warning

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

# Replace the prompt_template with this version that includes 'context' and 'question'
prompt_template = """
Context: {context}
Query: {query}

Patient Profile:
- Age: {age} years
- Gender: {gender}
- BMI: {bmi} ({bmi_category})

Current Metrics:
{real_time_metrics}

Analysis:
{threshold_analysis}

Based on this information, please provide:
1. Overall health assessment
2. Key concerns
3. Recommendations
4. Required monitoring

Response:
"""

# Call this before creating the db instance
if not initialize_collection():
    raise RuntimeError("Failed to initialize medical_db collection")

# Create db and retriever before using them
db = Qdrant(client=client, embeddings=embeddings, collection_name="medical_db")
retriever = db.as_retriever(search_kwargs={"k": 3})

# Now create the prompt template
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'query', 'age', 'gender', 'bmi', 'bmi_category', 'real_time_metrics', 'threshold_analysis']
)

# Create QA chain with the defined retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"
    }
)

# Add root route handler with proper template
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class HealthAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.thresholds = {
            'heart_rate': {'low': 60, 'high': 100},
            'blood_pressure': {'systolic': {'low': 90, 'high': 140},
                             'diastolic': {'low': 60, 'high': 90}},
            'blood_sugar': {'low': 70, 'high': 140},
            'spo2': {'low': 95, 'high': 100},
            'respiratory_rate': {'low': 12, 'high': 20},
            'body_temperature': {'low': 36.5, 'high': 37.5}
        }

    def analyze(self, user_data: dict) -> dict:
        try:
            # Get simulated metrics
            metrics = HealthMetricsSimulator.generate_metrics(user_data)
            
            # Calculate health indicators
            bmi = self.calculate_bmi(user_data['weight'], user_data['height'])
            bmi_category = self.get_bmi_category(bmi)
            
            # Analyze metrics against thresholds
            threshold_analysis = self.analyze_thresholds(metrics['real_time'])
            
            # Format metrics for prompt
            formatted_metrics = self.format_metrics(metrics['real_time'])
            
            # Create context from user data and metrics
            context = self.create_analysis_context(user_data, metrics, bmi, bmi_category, threshold_analysis)
            
            # Get analysis from LLM
            analysis = self.get_llm_analysis(context)

            return {
                "metrics": metrics['real_time'],
                "static_data": {
                    "age": user_data['age'],
                    "gender": user_data['gender'],
                    "height": user_data['height'],
                    "weight": user_data['weight'],
                    "bmi": round(bmi, 2),
                    "bmi_category": bmi_category
                },
                "analysis": analysis,
                "threshold_violations": threshold_analysis,
                "recommendations": self.extract_recommendations(analysis),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return self._get_safe_response()

    def analyze_thresholds(self, metrics: dict) -> List[str]:
        violations = []
        
        for metric, value in metrics.items():
            if metric == 'blood_pressure':
                try:
                    systolic, diastolic = map(int, value.split('/'))
                    if systolic > self.thresholds['blood_pressure']['systolic']['high']:
                        violations.append(f"High systolic blood pressure: {systolic} mmHg")
                    elif systolic < self.thresholds['blood_pressure']['systolic']['low']:
                        violations.append(f"Low systolic blood pressure: {systolic} mmHg")
                    
                    if diastolic > self.thresholds['blood_pressure']['diastolic']['high']:
                        violations.append(f"High diastolic blood pressure: {diastolic} mmHg")
                    elif diastolic < self.thresholds['blood_pressure']['diastolic']['low']:
                        violations.append(f"Low diastolic blood pressure: {diastolic} mmHg")
                except:
                    violations.append("Invalid blood pressure format")
                continue
                
            if metric in self.thresholds:
                try:
                    if isinstance(value, (int, float)):
                        if value < self.thresholds[metric]['low']:
                            violations.append(f"Low {metric.replace('_', ' ')}: {value}")
                        elif value > self.thresholds[metric]['high']:
                            violations.append(f"High {metric.replace('_', ' ')}: {value}")
                except:
                    continue

        return violations

    def create_analysis_context(self, user_data: dict, metrics: dict, bmi: float, bmi_category: str, violations: List[str]) -> str:
        return f"""
        Patient Profile:
        - Age: {user_data['age']} years
        - Gender: {user_data['gender']}
        - BMI: {bmi:.1f} ({bmi_category})
        - Lifestyle: {user_data.get('lifestyle', 'moderate')}

        Current Health Metrics:
        {self.format_metrics(metrics['real_time'])}

        Health Concerns:
        {'; '.join(violations) if violations else 'No immediate concerns'}

        Medical History:
        {user_data.get('medical_conditions', ['No known conditions'])}

        Current Medications:
        {user_data.get('medications', ['None reported'])}
        """

    def get_llm_analysis(self, context: str) -> str:
        prompt = f"""
        As a healthcare AI assistant, analyze the following patient data and provide:
        1. Overall health assessment
        2. Key concerns and risks
        3. Specific recommendations for improvement
        4. Monitoring requirements

        {context}

        Please provide a detailed but concise analysis:
        """
        
        try:
            return self.llm(prompt)
        except Exception as e:
            print(f"LLM Analysis error: {str(e)}")
            return "Error generating analysis"

    def extract_recommendations(self, analysis: str) -> List[str]:
        try:
            # Split by newlines and look for recommendation markers
            lines = analysis.split('\n')
            recommendations = []
            capturing = False
            
            for line in lines:
                if 'recommendation' in line.lower() or 'advise' in line.lower():
                    capturing = True
                elif capturing and line.strip():
                    if line.startswith('-') or line.startswith('•'):
                        recommendations.append(line.strip('- •'))
                elif capturing and not line.strip():
                    capturing = False
                    
            return recommendations if recommendations else ["No specific recommendations generated"]
        except:
            return ["Error extracting recommendations"]

    # ...existing code (keep the remaining helper methods)...

    @staticmethod
    def calculate_bmi(weight: float, height: float) -> float:
        return weight / ((height/100) ** 2)

    @staticmethod
    def get_bmi_category(bmi: float) -> str:
        if bmi < 18.5: return "Underweight"
        elif bmi < 25: return "Normal weight"
        elif bmi < 30: return "Overweight"
        else: return "Obese"

    @staticmethod
    def format_metrics(metrics: dict) -> str:
        return "\n".join([f"- {k.replace('_', ' ').title()}: {v}" 
                         for k, v in metrics.items()])

    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from the analysis text"""
        # Placeholder implementation
        return analysis.split('\n')

    def _get_safe_response(self) -> dict:
        """Return a safe response in case of errors"""
        return {
            "metrics": {},
            "static_data": {},
            "analysis": "Error in analysis",
            "threshold_violations": [],
            "bmi_data": {"value": 0, "category": "Unknown"},
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }

# Add these new route handlers before the existing /analyze_health endpoint
@app.get("/analyze_health")
async def get_health_form(request: Request):
    """Return the health analysis form"""
    return templates.TemplateResponse("health_form.html", {"request": request})

@app.get("/health_analyze")
async def get_health_analyze(
    age: int,
    gender: str,
    weight: float,
    height: float,
    lifestyle: str = "moderate"
):
    """Handle GET requests for health analysis"""
    try:
        data = {
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "lifestyle": lifestyle
        }
        result = analyzer.analyze(data)
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing health analysis: {str(e)}"
        )

# Modify the existing POST endpoint
@app.post("/analyze_health")
async def analyze_health(data: dict):
    """Handle POST requests for health analysis"""
    try:
        required_fields = ["age", "gender", "weight", "height"]
        if not all(k in data for k in required_fields):
            raise ValueError(f"Missing required fields. Please provide: {', '.join(required_fields)}")
        
        # Convert numeric fields
        data["age"] = int(data["age"])
        data["weight"] = float(data["weight"])
        data["height"] = float(data["height"])
        
        # Set default lifestyle if not provided
        if "lifestyle" not in data:
            data["lifestyle"] = "moderate"
        
        result = analyzer.analyze(data)
        return JSONResponse(content=jsonable_encoder(result))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing health analysis: {str(e)}")

# Initialize analyzer with just LLM
analyzer = HealthAnalyzer(llm)

# Initialize with Fastapi mount for static files
app.mount("/static", StaticFiles(directory="static"), name="static")