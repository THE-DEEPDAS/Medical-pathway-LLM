import pathway as pw
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from datetime import datetime
import os
from typing import Dict, List, Optional
import json
from health_simulator import HealthMetricsSimulator
import numpy as np

class MedicalPredictionPipeline:
    def __init__(self):
        # Initialize local embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize local LLM
        self.llm = CTransformers(
            model="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            model_type="mistral",
            config={
                'max_new_tokens': 1024,
                'context_length': 1024,
                'temperature': 0.1
            }
        )

        # Load medical knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # Initialize thresholds for metrics
        self.thresholds = {
            'heart_rate': {'low': 60, 'high': 100},
            'blood_pressure': {'systolic': {'low': 90, 'high': 140},
                             'diastolic': {'low': 60, 'high': 90}},
            'blood_sugar': {'low': 70, 'high': 140},
            'spo2': {'low': 95, 'high': 100}
        }

    def _load_knowledge_base(self) -> List[Dict]:
        """Load medical knowledge base from PDF files"""
        kb_path = os.path.join(os.path.dirname(__file__), 'Data')
        documents = []
        
        for file in os.listdir(kb_path):
            if file.endswith('.pdf'):
                # Process PDF and extract text
                text = self._process_pdf(os.path.join(kb_path, file))
                documents.append({
                    "source": file,
                    "content": text
                })
        
        return documents

    def create_pipeline(self):
        # Define input stream for real-time metrics
        metrics_stream = pw.io.csv.read(
            "real_time_metrics.csv",
            schema={
                "timestamp": str,
                "heart_rate": float,
                "blood_pressure": str,
                "blood_sugar": float,
                "spo2": float
            },
            mode="streaming"
        )

        # Embed the metrics data
        embedded_metrics = metrics_stream + pw.apply(
            self.embedder.embed_text,
            pw.this.to_json()
        )

        # Join with knowledge base
        relevant_knowledge = embedded_metrics.join(
            self.knowledge_base,
            pw.this.embedding,
            k=3  # Get top 3 relevant conditions
        )

        # Generate predictions using LLM
        predictions = relevant_knowledge + pw.apply(
            self._generate_prediction,
            metrics=pw.this.metrics,
            knowledge=pw.this.knowledge,
            context=pw.this.context
        )

        return predictions

    def _generate_prediction(self, metrics: Dict, knowledge: Dict, context: str) -> Dict:
        """Generate medical predictions using LLM"""
        prompt = self._create_prediction_prompt(metrics, knowledge, context)
        
        response = self.llm.generate(prompt)
        
        # Parse and structure the response
        try:
            prediction = {
                "timestamp": datetime.now().isoformat(),
                "metrics_used": metrics,
                "predictions": self._parse_llm_response(response),
                "knowledge_sources": knowledge,
                "confidence_score": self._calculate_confidence(metrics, knowledge)
            }
            return prediction
        except Exception as e:
            print(f"Error generating prediction: {e}")
            return self._get_safe_prediction()

    def _create_prediction_prompt(self, metrics: Dict, knowledge: Dict, context: str) -> str:
        return f"""
Based on the following patient data and medical knowledge, provide a detailed health analysis and predictions:

Current Vital Signs:
{json.dumps(metrics, indent=2)}

Relevant Medical Knowledge:
{json.dumps(knowledge, indent=2)}

Additional Context:
{context}

Please provide:
1. Current health status assessment
2. Potential risks and concerns
3. Short-term predictions (next 24 hours)
4. Recommended monitoring parameters
5. Confidence level in predictions (0-100%)

Use specific data points to justify each conclusion.
"""

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse and structure the LLM response"""
        try:
            # Add your parsing logic here
            sections = response.split("\n\n")
            parsed = {
                "status": sections[0] if len(sections) > 0 else "No status available",
                "risks": sections[1] if len(sections) > 1 else "No risks identified",
                "predictions": sections[2] if len(sections) > 2 else "No predictions available",
                "monitoring": sections[3] if len(sections) > 3 else "Standard monitoring recommended",
                "confidence": sections[4] if len(sections) > 4 else "Confidence level unknown"
            }
            return parsed
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                "status": "Error in prediction",
                "risks": "Unable to assess",
                "predictions": "Prediction failed",
                "monitoring": "Standard monitoring recommended",
                "confidence": "0"
            }

    def _calculate_confidence(self, metrics: Dict, knowledge: Dict) -> float:
        """Calculate confidence score for predictions"""
        try:
            # Add your confidence calculation logic here
            # Example: Based on data completeness and threshold violations
            completeness = sum(1 for v in metrics.values() if v is not None) / len(metrics)
            violations = self._check_thresholds(metrics)
            confidence = completeness * (1 - len(violations)/10)  # Reduce confidence for each violation
            return round(max(0, min(1, confidence)) * 100, 2)
        except:
            return 0.0

    def _check_thresholds(self, metrics: Dict) -> List[str]:
        """Check if metrics are within normal ranges"""
        violations = []
        
        try:
            for metric, value in metrics.items():
                if metric in self.thresholds:
                    threshold = self.thresholds[metric]
                    if isinstance(value, (int, float)):
                        if value < threshold['low']:
                            violations.append(f"Low {metric}: {value}")
                        elif value > threshold['high']:
                            violations.append(f"High {metric}: {value}")
                    elif metric == 'blood_pressure' and isinstance(value, str):
                        try:
                            systolic, diastolic = map(int, value.split('/'))
                            if systolic < threshold['systolic']['low']:
                                violations.append(f"Low systolic BP: {systolic}")
                            elif systolic > threshold['systolic']['high']:
                                violations.append(f"High systolic BP: {systolic}")
                            if diastolic < threshold['diastolic']['low']:
                                violations.append(f"Low diastolic BP: {diastolic}")
                            elif diastolic > threshold['diastolic']['high']:
                                violations.append(f"High diastolic BP: {diastolic}")
                        except:
                            violations.append("Invalid blood pressure format")
        except Exception as e:
            print(f"Error checking thresholds: {e}")
            
        return violations

    def _get_safe_prediction(self) -> Dict:
        """Return a safe prediction in case of errors"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics_used": {},
            "predictions": {
                "status": "Unable to generate prediction",
                "risks": "Unknown",
                "predictions": "System error - using default monitoring",
                "monitoring": "Standard protocols",
                "confidence": 0
            },
            "knowledge_sources": [],
            "confidence_score": 0
        }

# Usage example
if __name__ == "__main__":
    # Set up environment variables
    os.environ["HUGGINGFACE_API_KEY"] = "your_huggingface_api_key"
    
    try:
        # Initialize pipeline
        pipeline = MedicalPredictionPipeline()
        
        # Example static data
        static_data = {
            "age": 45,
            "gender": "female",
            "height": 165,
            "weight": 70,
            "medical_history": ["hypertension", "type 2 diabetes"]
        }
        
        # Generate simulated metrics
        simulator = HealthMetricsSimulator()
        real_time_data = simulator.generate_metrics(static_data)
        
        # Create and run prediction pipeline
        predictions = pipeline.create_pipeline()
        pw.run(predictions)
        
        print("Pipeline running successfully!")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
