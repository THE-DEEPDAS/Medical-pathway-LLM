from pydantic import BaseModel, EmailStr
from typing import Dict, Optional
import uuid

class UserProfile(BaseModel):
    user_id: str = str(uuid.uuid4())
    email: EmailStr
    password: str
    name: str
    age: int
    gender: str
    height: float
    weight: float
    medical_conditions: list[str] = []
    medications: list[str] = []
    lifestyle: Dict = {
        "exercise_frequency": "moderate",
        "diet_type": "balanced",
        "smoking": False,
        "alcohol": "none"
    }
    emergency_contact: Optional[Dict] = None
