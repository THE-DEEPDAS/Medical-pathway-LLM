import random
from datetime import datetime
import math

class HealthMetricsSimulator:
    @staticmethod
    def generate_metrics(user_data: dict) -> dict:
        try:
            simulation_time = user_data.get('simulation_time', 0)
            time_factor = simulation_time / 60  # Time in minutes
            
            # Add randomization to make metrics more dynamic
            variation = random.uniform(-0.5, 0.5)
            
            # Base values with more pronounced variations
            heart_rate = 75 + int(10 * math.sin(time_factor + variation))
            systolic = 120 + int(8 * math.sin(time_factor/2 + variation))
            diastolic = 80 + int(5 * math.sin(time_factor/2 + variation))
            blood_sugar = 100 + int(15 * math.sin(time_factor/3 + variation))
            spo2 = min(100, 98 + int(3 * math.sin(time_factor/4 + variation)))
            respiratory_rate = 16 + int(4 * math.sin(time_factor/3 + variation))
            body_temp = 37.0 + 0.5 * math.sin(time_factor/4 + variation)
            
            # Occasionally introduce more significant changes
            if random.random() < 0.3:  # 30% chance of significant variation
                modifier = random.choice([
                    ('heart_rate', 15),
                    ('systolic', 20),
                    ('diastolic', 15),
                    ('blood_sugar', 30),
                    ('spo2', -4),
                    ('respiratory_rate', 5),
                    ('body_temp', 0.8)
                ])
                if modifier[0] == 'heart_rate':
                    heart_rate += modifier[1] * random.choice([-1, 1])
                elif modifier[0] == 'systolic':
                    systolic += modifier[1] * random.choice([-1, 1])
                elif modifier[0] == 'diastolic':
                    diastolic += modifier[1] * random.choice([-1, 1])
                elif modifier[0] == 'blood_sugar':
                    blood_sugar += modifier[1] * random.choice([-1, 1])
                elif modifier[0] == 'spo2':
                    spo2 += modifier[1] * random.choice([-1, 1])
                elif modifier[0] == 'respiratory_rate':
                    respiratory_rate += modifier[1] * random.choice([-1, 1])
                elif modifier[0] == 'body_temp':
                    body_temp += modifier[1] * random.choice([-1, 1])

            metrics = {
                "heart_rate": max(50, min(120, heart_rate)),
                "blood_pressure": f"{max(90, min(160, systolic))}/{max(50, min(100, diastolic))}",
                "blood_sugar": max(60, min(200, blood_sugar)),
                "spo2": max(90, min(100, spo2)),
                "respiratory_rate": max(8, min(25, respiratory_rate)),
                "body_temperature": round(max(35.5, min(38.5, body_temp)), 1),
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