import json
import os
from typing import Dict, Optional

class UserProfile:
    def __init__(self):
        self.profile_file = "user_profiles.json"
        self.profiles = self._load_profiles()

    def _load_profiles(self) -> Dict:
        if os.path.exists(self.profile_file):
            with open(self.profile_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_profiles(self):
        with open(self.profile_file, 'w') as f:
            json.dump(self.profiles, f, indent=4)

    def create_profile(self, user_id: str, data: Dict):
        """Create or update user profile with baseline health data"""
        self.profiles[user_id] = {
            "baseline": {
                "age": data.get("age", 30),
                "gender": data.get("gender", "unknown"),
                "height": data.get("height", 170),  # cm
                "weight": data.get("weight", 70),   # kg
                "medical_conditions": data.get("medical_conditions", []),
                "medications": data.get("medications", []),
                "allergies": data.get("allergies", []),
                "lifestyle": {
                    "exercise_frequency": data.get("exercise_frequency", "moderate"),
                    "diet_type": data.get("diet_type", "balanced"),
                    "smoking": data.get("smoking", False),
                    "alcohol": data.get("alcohol", "none")
                }
            }
        }
        self._save_profiles()

    def get_profile(self, user_id: str) -> Optional[Dict]:
        return self.profiles.get(user_id)
