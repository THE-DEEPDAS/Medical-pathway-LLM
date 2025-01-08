import random
from datetime import datetime
from typing import Dict, Union

class HealthMetricsSimulator:
    @staticmethod
    def generate_metrics(age_or_profile: Union[int, Dict], gender: str = None):
        """Generate health metrics based on either age/gender or full profile"""
        # If first argument is a dictionary, use profile-based generation
        if isinstance(age_or_profile, dict):
            return HealthMetricsSimulator._generate_from_profile(age_or_profile)
        
        # Create a basic profile for simple age/gender generation
        profile = {
            "baseline": {
                "age": age_or_profile,
                "gender": gender or "unknown",
                "height": 170,
                "weight": 70,
                "lifestyle": {
                    "exercise_frequency": "moderate",
                    "diet_type": "balanced",
                    "smoking": False,
                    "alcohol": "none"
                }
            }
        }
        return HealthMetricsSimulator._generate_from_profile(profile)

    @staticmethod
    def _generate_from_profile(profile: Dict):
        """Generate metrics based on detailed user profile"""
        baseline = profile["baseline"]
        age = baseline["age"]
        gender = baseline["gender"]
        weight = baseline.get("weight", 70)
        lifestyle = baseline.get("lifestyle", {})

        heart_rate_base = 70 if age < 60 else 75
        heart_rate_variance = 15 if lifestyle.get("exercise_frequency") == "high" else 20
        
        metrics = {
            "heart_rate": random.randint(
                heart_rate_base - heart_rate_variance,
                heart_rate_base + heart_rate_variance
            ),
            "blood_pressure": f"{random.randint(110, 140)}/{random.randint(70, 90)}",
            "sleep_hours": round(random.uniform(5.0, 9.0), 1),
            "calories_burned": HealthMetricsSimulator._calculate_calories(
                weight, 
                lifestyle.get("exercise_frequency", "moderate")
            ),
            "cholesterol": HealthMetricsSimulator._calculate_cholesterol(
                age, 
                lifestyle
            ),
            "sleep_quality": random.choice(["Poor", "Fair", "Good", "Excellent"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bmi": round(weight / ((baseline["height"]/100) ** 2), 1)
        }
        
        return metrics

    @staticmethod
    def _calculate_calories(weight: float, exercise_frequency: str) -> int:
        base_calories = weight * 30
        exercise_multiplier = {
            "low": 1.0,
            "moderate": 1.2,
            "high": 1.4
        }.get(exercise_frequency, 1.0)
        return int(base_calories * exercise_multiplier)

    @staticmethod
    def _calculate_cholesterol(age: int, lifestyle: Dict) -> int:
        base_cholesterol = 150
        if age > 40:
            base_cholesterol += (age - 40) * 1.5
        if lifestyle["diet_type"] == "unhealthy":
            base_cholesterol += 30
        if lifestyle["smoking"]:
            base_cholesterol += 20
        return int(base_cholesterol)
