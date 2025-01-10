import random
from datetime import datetime

class HealthMetricsSimulator:
    @staticmethod
    def generate_metrics(user_data: dict) -> dict:
        """Generate real-time health metrics"""
        try:
            age = int(user_data.get("age", 30))
            weight = float(user_data.get("weight", 70))
            lifestyle = str(user_data.get("lifestyle", "moderate"))
            medical_conditions = user_data.get("medical_conditions", [])

            # Adjust base metrics based on age, lifestyle and conditions
            activity_factor = {"sedentary": 1.2, "moderate": 1.0, "active": 0.8}.get(lifestyle, 1.0)
            condition_factor = 1.0 + (len(medical_conditions) * 0.1)  # Increase variability with conditions

            # Generate base metrics with realistic variations
            base_hr = 75 - (age // 20)
            base_systolic = 110 + (age // 5)
            base_diastolic = 70 + (age // 10)

            metrics = {
                "heart_rate": max(60, min(100, int(base_hr * activity_factor * condition_factor + random.randint(-5, 5)))),
                "blood_pressure": f"{base_systolic}/{base_diastolic}",
                "blood_sugar": random.randint(80, 140),
                "spo2": random.randint(95, 100),
                "respiratory_rate": random.randint(12, 20),
                "body_temperature": round(37 + random.uniform(-0.5, 0.5), 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return {
                "static": {
                    "height": user_data.get("height"),
                    "weight": weight,
                    "age": age,
                    "lifestyle": lifestyle
                },
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
                "stress_level": 5,
                "hydration": 75,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
