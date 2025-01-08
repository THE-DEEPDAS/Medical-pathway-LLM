import json
import os

class UserDataStore:
    def __init__(self):
        self.data_file = "user_static_data.json"
        self._ensure_data_file()

    def _ensure_data_file(self):
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump({}, f)

    def save_user_data(self, user_id: str, data: dict):
        """Save static user data"""
        static_data = {
            "age": data.get("age"),
            "gender": data.get("gender"),
            "height": data.get("height"),
            "weight": data.get("weight"),
            "medical_conditions": data.get("medical_conditions", []),
            "medications": data.get("medications", []),
            "allergies": data.get("allergies", [])
        }
        
        with open(self.data_file, 'r+') as f:
            data = json.load(f)
            data[user_id] = static_data
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

    def get_user_data(self, user_id: str) -> dict:
        with open(self.data_file, 'r') as f:
            data = json.load(f)
            return data.get(user_id, {})
