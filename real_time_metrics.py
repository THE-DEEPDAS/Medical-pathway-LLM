import random
from datetime import datetime

class RealTimeMetrics:
    @staticmethod
    def generate(user_data: dict):
        """Generate real-time health metrics based on user's static data"""
        age = user_data.get("age", 30)
        weight = user_data.get("weight", 70)

        return {
            "heart_rate": random.randint(60 + (age//20), 100 + (age//20)),
            "blood_pressure": f"{random.randint(110, 140)}/{random.randint(70, 90)}",
            "blood_sugar": random.randint(80, 140),
            "spo2": random.randint(95, 100),
            "body_temperature": round(random.uniform(36.5, 37.5), 1),
            "respiratory_rate": random.randint(12, 20),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
