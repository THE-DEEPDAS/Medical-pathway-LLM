import random
from datetime import datetime
import math

class HealthMetricsSimulator:
    @staticmethod
    def generate_metrics(user_data: dict) -> dict:
        try:
            simulation_time = user_data.get('simulation_time', 0)
            time_factor = simulation_time / 60  # Time in minutes
            
            # Base values with periodic variations
            heart_rate = 75 + int(5 * math.sin(time_factor))
            systolic = 120 + int(3 * math.sin(time_factor/2))
            diastolic = 80 + int(2 * math.sin(time_factor/2))
            blood_sugar = 100 + int(5 * math.sin(time_factor/3))
            
            metrics = {
                "heart_rate": max(60, min(100, heart_rate)),
                "blood_pressure": f"{systolic}/{diastolic}",
                "blood_sugar": max(70, min(140, blood_sugar)),
                "spo2": min(100, 98 + random.randint(-1, 1)),
                "respiratory_rate": max(12, min(20, 16 + random.randint(-2, 2))),
                "body_temperature": round(37.0 + 0.1 * math.sin(time_factor/4), 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return {
                "static": user_data,
                "real_time": metrics
            }

        except Exception as e:
            print(f"Error generating metrics: {str(e)}")
            return HealthMetricsSimulator._get_safe_defaults()

    @staticmethod
    def _get_safe_defaults() -> dict:
        return {
            "static": {"height": 170, "weight": 70, "age": 30},
            "real_time": {
                "heart_rate": 75,
                "blood_pressure": "120/80",
                "blood_sugar": 100,
                "spo2": 98,
                "respiratory_rate": 16,
                "body_temperature": 37.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
