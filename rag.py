# RUNNING IN TENSORFLOW ENV 
# coher, or reranking, long-context-reorder method to abound missing in the middle 
# uvicorn rag:app
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
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
from ingest import ingest_docs
from db.user_store import UserStore
from models.user import UserProfile
from datetime import datetime
import random
import numpy as np
from custom_pathway import PathwayAnalyzer
from langchain.prompts import PromptTemplate

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

# Update the config settings
config = {
    'max_new_tokens': 1024,  # Reduced from 2048
    'context_length': 1024,  # Reduced from 2048
    'repetition_penalty': 1.1,
    'temperature': 0.1,      # Reduced from 0.2 for more consistent outputs
    'top_k': 40,            # Reduced from 50
    'top_p': 0.9,           # Added more focused sampling
    'stream': False,        # Changed to False to prevent streaming issues
    'threads': min(4, int(os.cpu_count() / 2)),  # Limit thread count
    'batch_size': 1,        # Added batch size
    'n_ctx': 1024          # Added context window size
}

# Call this before initializing the model
ensure_model_directory()

# Add improved error handling and retries for LLM
class LLMWrapper:
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.retries = 3
        self.llm = None
        self.initialize_llm()

    def initialize_llm(self):
        for attempt in range(self.retries):
            try:
                self.llm = CTransformers(
                    model=self.model_path,
                    model_type="mistral",
                    lib="avx2",
                    **self.config
                )
                return
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.retries - 1:
                    raise

    def __call__(self, *args, **kwargs):
        for attempt in range(self.retries):
            try:
                return self.llm(*args, **kwargs)
            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.retries - 1:
                    raise

# Add after the config definition
llm = LLMWrapper(get_model_path(), config)

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

class HealthTrend:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.metrics_history = []
        
    def add_metrics(self, metrics):
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
            
    def analyze_trends(self):
        if len(self.metrics_history) < 2:
            return "Insufficient data for trend analysis"
            
        trends = {}
        for key in ['heart_rate', 'blood_sugar', 'spo2']:
            values = [m.get(key) for m in self.metrics_history if key in m]
            if len(values) >= 2:
                trend = self._calculate_trend(values)
                trends[key] = trend
                
        return self._format_trends(trends)
    
    def _calculate_trend(self, values):
        try:
            values = [float(v) for v in values]
            first, last = values[0], values[-1]
            change = ((last - first) / first) * 100
            if abs(change) < 2:
                return "stable"
            return "increasing" if change > 0 else "decreasing"
        except:
            return "unknown"
            
    def _format_trends(self, trends):
        return "\n".join([f"- {k.replace('_', ' ').title()}: {v}" 
                         for k, v in trends.items()])

# Modify HealthAnalyzer class to include trends
class HealthAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.simulation_time = 0
        self.previous_metrics = {}  # Initialize as empty dict
        self.cache = {}  # Initialize cache
        self.current_metrics = None  # Track current metrics
        self.thresholds = {
            'heart_rate': {'low': 60, 'high': 100},
            'blood_pressure': {'systolic': {'low': 90, 'high': 140},
                             'diastolic': {'low': 60, 'high': 90}},
            'blood_sugar': {'low': 70, 'high': 140},
            'spo2': {'low': 95, 'high': 100},
            'respiratory_rate': {'low': 12, 'high': 20},
            'body_temperature': {'low': 36.5, 'high': 37.5}
        }
        self.trends = HealthTrend()
        self.alert_history = []
        self.risk_scores = {}
        
    def analyze(self, user_data: dict) -> dict:
        try:
            metrics = HealthMetricsSimulator.generate_metrics(user_data)
            self.trends.add_metrics(metrics['real_time'])
            
            # Calculate risk scores
            risk_score = self._calculate_risk_score(metrics['real_time'])
            self.risk_scores[datetime.now()] = risk_score
            
            # Generate enhanced analysis
            analysis = self._generate_enhanced_analysis(
                metrics['real_time'],
                self.previous_metrics,
                risk_score
            )
            
            self.previous_metrics = metrics['real_time'].copy()
            
            return {
                "metrics": metrics['real_time'],
                "static_data": metrics['static'],
                "analysis": analysis,
                "threshold_violations": self.analyze_thresholds(metrics['real_time']),
                "recommendations": self._generate_detailed_recommendations(
                    metrics['real_time'],
                    risk_score
                ),
                "risk_score": risk_score,
                "trends": self.trends.analyze_trends(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._get_safe_response()

    def _calculate_risk_score(self, metrics: dict) -> float:
        try:
            score = 0.0
            violations = self.analyze_thresholds(metrics)
            
            # Base score from violations
            score += len(violations) * 10
            
            # Check specific metrics
            if 'heart_rate' in metrics:
                hr = float(metrics['heart_rate'])
                if hr > 120 or hr < 50:
                    score += 20
                elif hr > 100 or hr < 60:
                    score += 10
                    
            if 'blood_pressure' in metrics:
                try:
                    sys, dia = map(int, metrics['blood_pressure'].split('/'))
                    if sys > 180 or dia > 120:
                        score += 30
                    elif sys > 140 or dia > 90:
                        score += 15
                except:
                    pass
                    
            if 'spo2' in metrics:
                spo2 = float(metrics['spo2'])
                if spo2 < 90:
                    score += 40
                elif spo2 < 95:
                    score += 20
                    
            return min(100, score)
        except:
            return 0.0

    def _generate_detailed_recommendations(self, metrics: dict, risk_score: float) -> str:
        basic_recs = self._generate_recommendations(metrics)
        
        # Add risk-based recommendations
        risk_recs = []
        if risk_score >= 70:
            risk_recs.extend([
                "- Immediate medical attention recommended",
                "- Contact healthcare provider urgently",
                "- Prepare for possible emergency response"
            ])
        elif risk_score >= 40:
            risk_recs.extend([
                "- Schedule urgent medical consultation",
                "- Increase monitoring frequency",
                "- Review medication compliance"
            ])
        elif risk_score >= 20:
            risk_recs.extend([
                "- Schedule routine check-up",
                "- Monitor trends closely",
                "- Review lifestyle factors"
            ])
            
        lifestyle_recs = [
            "- Maintain regular sleep schedule",
            "- Stay hydrated",
            "- Practice stress management",
            "- Ensure regular physical activity as appropriate",
            "- Follow balanced nutrition plan"
        ]
            
        return f"""Basic Recommendations:
{basic_recs}

Risk-Based Actions:
{chr(10).join(risk_recs)}

Lifestyle Recommendations:
{chr(10).join(lifestyle_recs)}
"""

    def analyze_thresholds(self, metrics: dict) -> List[str]:
        violations = []
        
        try:
            # Handle blood pressure separately
            if 'blood_pressure' in metrics:
                bp = metrics['blood_pressure']
                if isinstance(bp, str) and '/' in bp:
                    try:
                        systolic, diastolic = map(int, bp.split('/'))
                        if systolic > self.thresholds['blood_pressure']['systolic']['high']:
                            violations.append(f"High systolic blood pressure: {systolic} mmHg")
                        elif systolic < self.thresholds['blood_pressure']['systolic']['low']:
                            violations.append(f"Low systolic blood pressure: {systolic} mmHg")
                        
                        if diastolic > self.thresholds['blood_pressure']['diastolic']['high']:
                            violations.append(f"High diastolic blood pressure: {diastolic} mmHg")
                        elif diastolic < self.thresholds['blood_pressure']['diastolic']['low']:
                            violations.append(f"Low diastolic blood pressure: {diastolic} mmHg")
                    except ValueError:
                        violations.append("Invalid blood pressure format")

            # Handle other metrics
            for metric, value in metrics.items():
                if metric == 'blood_pressure' or metric == 'timestamp':
                    continue
                    
                if metric in self.thresholds:
                    try:
                        value = float(value)
                        if value < self.thresholds[metric]['low']:
                            violations.append(f"Low {metric.replace('_', ' ')}: {value}")
                        elif value > self.thresholds[metric]['high']:
                            violations.append(f"High {metric.replace('_', ' ')}: {value}")
                    except (ValueError, TypeError):
                        continue

        except Exception as e:
            print(f"Error in analyze_thresholds: {str(e)}")
            violations.append("Error analyzing health metrics")

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
        try:
            if context in self.cache:
                return self.cache[context]

            prompt = self._create_analysis_prompt(context)
            
            # Add safety checks
            if not self.llm:
                return self._get_default_analysis()
                
            try:
                response = self.llm(prompt, max_tokens=256)
                if not response or not isinstance(response, str):
                    return self._get_default_analysis()
                    
                cleaned_response = response.strip()
                self.cache[context] = cleaned_response  # Cache the response
                return cleaned_response
            except Exception as e:
                print(f"LLM error: {str(e)}")
                return self._get_default_analysis()
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._get_default_analysis()

    def _create_analysis_prompt(self, context: str) -> str:
        return f"""You are a medical AI assistant monitoring patient vitals in real-time. Time elapsed: {self.simulation_time} seconds.

{context}

Based on the current metrics and their trends, provide a brief natural language analysis:

1. Current Status: Summarize the patient's current state
2. Changes: Note any significant changes from previous readings
3. Recommendations: Suggest immediate actions if needed
4. Next Steps: What to monitor closely in the next 30 seconds

Keep the response conversational and focused on real-time changes."""

    def _verify_response_structure(self, response: str) -> bool:
        required_sections = [
            "OVERALL HEALTH ASSESSMENT:",
            "KEY CONCERNS:",
            "RECOMMENDATIONS:",
            "MONITORING REQUIREMENTS:"
        ]
        return all(section in response for section in required_sections)

    def extract_recommendations(self, analysis: str) -> List[str]:
        try:
            recommendations = []
            in_recommendations = False
            
            for line in analysis.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                    
                if "RECOMMENDATIONS:" in line:
                    in_recommendations = True
                    continue
                elif "MONITORING REQUIREMENTS:" in line:
                    break
                    
                if in_recommendations and (line.startswith('-') or line.startswith('•')):
                    recommendation = line.strip('- •').strip()
                    if recommendation:
                        recommendations.append(recommendation)
            
            return recommendations if recommendations else ["No specific recommendations generated"]
        except Exception as e:
            print(f"Error extracting recommendations: {str(e)}")
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

    def _get_default_analysis(self) -> str:
        """Provide a detailed default analysis when LLM fails"""
        metrics = self.previous_metrics or {}
        violations = self.analyze_thresholds(metrics)
        
        status = "All vital signs are within normal ranges" if not violations else \
                "Some metrics require attention"
                
        return f"""
Time elapsed: {self.simulation_time} seconds

Current Status:
- {status}
{self._format_violations(violations)}

Changes from Previous Reading:
- Continuing to monitor vital signs
- No significant changes detected

Recommendations:
- Continue regular monitoring
- Maintain current activity level
- Stay hydrated and well-rested

Next Steps:
- Monitor vital signs for next 30 seconds
- Watch for any sudden changes
- Record any symptoms or concerns
"""

    def _format_violations(self, violations: List[str]) -> str:
        if not violations:
            return "- No health concerns detected"
        return "\n".join(f"- {v}" for v in violations)

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

# Add a background task to refresh the analysis every minute
from fastapi_utils.tasks import repeat_every

@app.on_event("startup")
@repeat_every(seconds=30)  # Changed from 60 to 30 seconds
async def refresh_analysis():
    try:
        print("\nStarting new health simulation cycle...")
        analyzer.simulation_time += 30  # Increment simulation time
        
        # Simulate changing health metrics
        user_data = {
            "age": 35,
            "gender": "male",
            "weight": 75.5,
            "height": 175,
            "lifestyle": "moderate",
            "simulation_time": analyzer.simulation_time
        }
        
        result = analyzer.analyze(user_data)
        
        # Print real-time analysis
        print("\nReal-time Health Analysis:")
        print("-------------------------")
        print(f"Time elapsed: {analyzer.simulation_time} seconds")
        print("Current Metrics:")
        for key, value in result['metrics'].items():
            if key != 'timestamp':
                print(f"  {key}: {value}")
        print("\nAnalysis:")
        print(result['analysis'])
        print("-------------------------\n")
        
    except Exception as e:
        print(f"Error in simulation cycle: {str(e)}")

# ...existing code...

class HealthAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.simulation_time = 0  # Add simulation time counter
        self.previous_metrics = None  # Store previous metrics for comparison
        self.thresholds = {
            # ...existing thresholds...
        }

    def analyze(self, user_data: dict) -> dict:
        try:
            # Add simulation time to user data
            user_data['simulation_time'] = self.simulation_time
            
            # Generate metrics
            metrics = HealthMetricsSimulator.generate_metrics(user_data)
            
            # Compare with previous metrics
            changes = self.compare_metrics(self.previous_metrics, metrics['real_time'])
            self.previous_metrics = metrics['real_time']

            # Create analysis message
            analysis = f"""
Time elapsed: {self.simulation_time} seconds

Current Status:
- {self.summarize_current_status(metrics['real_time'])}

Changes from Previous Reading:
{changes}

Recommendations:
{self.get_recommendations(metrics['real_time'], changes)}
"""

            return {
                "metrics": metrics['real_time'],
                "static_data": {
                    "age": user_data['age'],
                    "gender": user_data['gender'],
                    "height": user_data['height'],
                    "weight": user_data['weight'],
                    "simulation_time": self.simulation_time
                },
                "analysis": analysis,
                "threshold_violations": self.analyze_thresholds(metrics['real_time']),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._get_safe_response()

    def compare_metrics(self, prev_metrics: dict, current_metrics: dict) -> str:
        if not prev_metrics:
            return "Initial reading - no changes to report."
        
        changes = []
        for key in ['heart_rate', 'blood_pressure', 'blood_sugar', 'spo2']:
            if key in prev_metrics and key in current_metrics:
                prev_val = prev_metrics[key]
                curr_val = current_metrics[key]
                if prev_val != curr_val:
                    changes.append(f"- {key.replace('_', ' ').title()}: {prev_val} → {curr_val}")
        
        return "\n".join(changes) if changes else "No significant changes"

    def summarize_current_status(self, metrics: dict) -> str:
        violations = self.analyze_thresholds(metrics)
        if not violations:
            return "All vital signs are within normal ranges"
        return "Attention needed: " + "; ".join(violations)

    def get_recommendations(self, metrics: dict, changes: str) -> str:
        violations = self.analyze_thresholds(metrics)
        if not violations:
            return "Continue monitoring - all parameters are stable"
        
        recs = []
        for violation in violations:
            if "blood pressure" in violation.lower():
                recs.append("Monitor blood pressure closely")
            elif "heart rate" in violation.lower():
                recs.append("Check physical activity and stress levels")
            # Add more specific recommendations...
        
        return "\n".join(recs) if recs else "No specific recommendations at this time"

# ...existing code...

@app.on_event("startup")
@repeat_every(seconds=30)
async def refresh_analysis():
    try:
        print("\n=== Health Simulation Update ===")
        analyzer.simulation_time += 30
        
        user_data = {
            "age": 35,
            "gender": "male",
            "weight": 75.5,
            "height": 175,
            "lifestyle": "moderate"
        }
        
        result = analyzer.analyze(user_data)
        
        print(f"\nTime Elapsed: {analyzer.simulation_time} seconds")
        print("\nCurrent Metrics:")
        for key, value in result['metrics'].items():
            if key != 'timestamp':
                print(f"  {key}: {value}")
        
        print("\nAnalysis:")
        print(result['analysis'])
        print("\n" + "="*30 + "\n")
        
    except Exception as e:
        print(f"Simulation Error: {str(e)}")

# ...existing code...

class HealthAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.simulation_time = 0
        self.previous_metrics = None
        self.cache = {}  # Add cache for storing previous analyses

    def analyze(self, user_data: dict) -> dict:
        try:
            metrics = HealthMetricsSimulator.generate_metrics(user_data)
            
            # Generate simple analysis without LLM
            analysis = self.generate_simple_analysis(metrics['real_time'], self.previous_metrics)
            self.previous_metrics = metrics['real_time']

            return {
                "metrics": metrics['real_time'],
                "static_data": metrics['static'],
                "analysis": analysis,
                "threshold_violations": self.analyze_thresholds(metrics['real_time']),
                "recommendations": self.generate_recommendations(metrics['real_time']),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._get_safe_response()

    def generate_simple_analysis(self, current_metrics: dict, previous_metrics: dict) -> str:
        changes = []
        status_msg = "All vital signs are within normal limits."
        
        violations = self.analyze_thresholds(current_metrics)
        if violations:
            status_msg = "Attention needed: " + "; ".join(violations)

        if previous_metrics:
            for key in ['heart_rate', 'blood_pressure', 'blood_sugar', 'spo2']:
                if key in previous_metrics and key in current_metrics:
                    prev = previous_metrics[key]
                    curr = current_metrics[key]
                    if prev != curr:
                        changes.append(f"{key.replace('_', ' ').title()}: {prev} → {curr}")

        return f"""
Time elapsed: {self.simulation_time} seconds

Current Status:
{status_msg}

Changes from Previous Reading:
{chr(10).join(changes) if changes else "No significant changes"}

Recommendations:
{self.generate_recommendations(current_metrics)}
"""

    def generate_recommendations(self, metrics: dict) -> str:
        violations = self.analyze_thresholds(metrics)
        recs = []
        
        if not violations:
            return "- Continue monitoring - all parameters are stable"
            
        for v in violations:
            if "blood pressure" in v.lower():
                recs.append("- Check blood pressure again in 5 minutes")
            elif "heart rate" in v.lower():
                recs.append("- Monitor heart rate closely")
            elif "spo2" in v.lower():
                recs.append("- Check oxygen saturation")
                
        if not recs:
            recs.append("- Continue regular monitoring")
            
        return "\n".join(recs)

# ...existing code...

@app.on_event("startup")
@repeat_every(seconds=30)
async def refresh_analysis():
    try:
        print("\n=== Health Simulation Update ===")
        analyzer.simulation_time += 30
        
        user_data = {
            "age": 35,
            "gender": "male",
            "weight": 75.5,
            "height": 175,
            "lifestyle": "moderate",
            "simulation_time": analyzer.simulation_time
        }
        
        result = analyzer.analyze(user_data)
        
        # Print formatted output
        print(f"\nTime Elapsed: {analyzer.simulation_time} seconds")
        print("\nCurrent Metrics:")
        for key, value in result['metrics'].items():
            if key != 'timestamp':
                print(f"  {key}: {value}")
        
        print("\nAnalysis:")
        print(result['analysis'])
        print("\n" + "="*30)
        
    except Exception as e:
        print(f"Simulation Error: {str(e)}")

# ...existing code...

class HealthAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.simulation_time = 0
        self.previous_metrics = None
        self.thresholds = {
            # ...existing thresholds...
        }

    def analyze(self, user_data: dict) -> dict:
        try:
            # Generate metrics
            metrics = HealthMetricsSimulator.generate_metrics(user_data)
            
            # Generate analysis without LLM
            analysis = self._generate_analysis(metrics['real_time'], self.previous_metrics)
            self.previous_metrics = metrics['real_time'].copy()

            return {
                "metrics": metrics['real_time'],
                "static_data": metrics['static'],
                "analysis": analysis,
                "threshold_violations": self.analyze_thresholds(metrics['real_time']),
                "recommendations": self._generate_recommendations(metrics['real_time']),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._get_safe_response()

    def _generate_analysis(self, current_metrics: dict, previous_metrics: dict) -> str:
        changes = []
        
        # Get violations and status
        violations = self.analyze_thresholds(current_metrics)
        status = "All vital signs are within normal ranges" if not violations else \
                "Attention needed: " + "; ".join(violations)

        # Compare with previous readings
        if previous_metrics:
            for key in ['heart_rate', 'blood_pressure', 'blood_sugar', 'spo2']:
                if key in previous_metrics and key in current_metrics:
                    prev = previous_metrics[key]
                    curr = current_metrics[key]
                    if prev != curr:
                        changes.append(f"- {key.replace('_', ' ').title()}: {prev} → {curr}")

        return f"""
Time elapsed: {self.simulation_time} seconds

Current Status:
- {status}

Changes from Previous Reading:
{chr(10).join(changes) if changes else "- No significant changes"}

Recommendations:
{self._generate_recommendations(current_metrics)}
"""

    def _generate_recommendations(self, metrics: dict) -> str:
        violations = self.analyze_thresholds(metrics)
        if not violations:
            return "- Continue monitoring - all parameters are stable"

        recs = []
        for v in violations:
            if "blood pressure" in v.lower():
                recs.append("- Monitor blood pressure every 5 minutes")
            elif "heart rate" in v.lower():
                recs.append("- Check activity level and stress")
            elif "spo2" in v.lower():
                recs.append("- Verify oxygen saturation reading")
            else:
                recs.append(f"- Monitor {v.split(':')[0].lower()}")

        return "\n".join(recs) if recs else "- Continue regular monitoring"

# ...existing code...

@app.on_event("startup")
@repeat_every(seconds=30)
async def refresh_analysis():
    try:
        analyzer.simulation_time += 30
        
        user_data = {
            "age": 35,
            "gender": "male",
            "weight": 75.5,
            "height": 175,
            "lifestyle": "moderate",
            "simulation_time": analyzer.simulation_time
        }
        
        result = analyzer.analyze(user_data)
        
        # Print formatted output
        print("\n=== Health Simulation Update ===")
        print(f"Time Elapsed: {analyzer.simulation_time} seconds")
        print("\nCurrent Metrics:")
        for key, value in result['metrics'].items():
            if key != 'timestamp':
                print(f"  {key}: {value}")
        print("\nAnalysis:")
        print(result['analysis'])
        print("=" * 50)
        
    except Exception as e:
        print(f"Simulation Error: {str(e)}")

# Add enhanced error handling for API endpoints
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = str(exc)
    print(f"Error: {error_msg}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": error_msg}
    )

# Add health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test LLM
        llm("Test prompt", max_tokens=10)
        
        # Test vector store
        db.similarity_search("test", k=1)
        
        return {"status": "healthy", "components": {
            "llm": "operational",
            "vector_store": "operational"
        }}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

# ...existing code...