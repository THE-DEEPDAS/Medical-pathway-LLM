import pathway as pw
from typing import Dict, List
import json
from datetime import datetime

class HealthMetricsPathway:
    def __init__(self):
        self.input_schema = {
            "timestamp": pw.Column(pw.Timestamp),
            "heart_rate": pw.Column(pw.Float64),
            "blood_pressure": pw.Column(pw.String),
            "blood_sugar": pw.Column(pw.Float64),
            "spo2": pw.Column(pw.Float64),
            "respiratory_rate": pw.Column(pw.Float64),
            "body_temperature": pw.Column(pw.Float64)
        }

    def create_pipeline(self) -> pw.Table:
        # Input stream from health metrics
        metrics_stream = pw.io.csv.read(
            "metrics_stream",
            schema=self.input_schema,
            mode="streaming"
        )

        # Apply transformations
        processed = metrics_stream + pw.apply(
            self._process_metrics,
            pw.this.as_dict()
        )

        # Time window aggregation
        windowed = processed.windowby(
            pw.this.timestamp,
            pw.temporal.fixed_windows(pw.temporal.duration.seconds(30))
        ).aggregate(
            stats=pw.reducers.custom(self._compute_statistics)
        )

        # Join with knowledge base
        enriched = windowed.join(
            self._load_knowledge_base(),
            pw.this.stats.context_key == pw.this.condition_key
        )

        return enriched

    @staticmethod
    def _process_metrics(data: Dict) -> Dict:
        """Process individual metric records"""
        try:
            # Extract systolic/diastolic from blood pressure
            if 'blood_pressure' in data:
                sys, dia = map(int, data['blood_pressure'].split('/'))
                data['systolic'] = sys
                data['diastolic'] = dia

            # Add derived metrics
            data['mean_arterial_pressure'] = (
                (data['systolic'] + 2 * data['diastolic']) / 3
            )

            # Add timestamp-based features
            ts = datetime.fromisoformat(data['timestamp'])
            data['hour_of_day'] = ts.hour
            data['is_night'] = 22 <= ts.hour or ts.hour <= 6

            return data
        except Exception as e:
            print(f"Error processing metrics: {e}")
            return data

    @staticmethod
    def _compute_statistics(values: List[Dict]) -> Dict:
        """Compute statistics over time windows"""
        try:
            stats = {}
            for metric in ['heart_rate', 'systolic', 'diastolic', 'blood_sugar', 'spo2']:
                metric_values = [v.get(metric) for v in values if v.get(metric)]
                if metric_values:
                    stats[f"{metric}_mean"] = sum(metric_values) / len(metric_values)
                    stats[f"{metric}_min"] = min(metric_values)
                    stats[f"{metric}_max"] = max(metric_values)
                    
            return stats
        except Exception as e:
            print(f"Error computing statistics: {e}")
            return {}

    @staticmethod
    def _load_knowledge_base() -> pw.Table:
        """Load medical knowledge base into Pathway"""
        return pw.io.json.read(
            "knowledge_base",
            schema={
                "condition_key": pw.Column(pw.String),
                "description": pw.Column(pw.String),
                "thresholds": pw.Column(pw.JSON)
            }
        )

    def run(self):
        """Run the Pathway pipeline"""
        pipeline = self.create_pipeline()
        pw.run(pipeline)
