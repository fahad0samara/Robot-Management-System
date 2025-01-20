import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import joblib

class MaintenancePredictor:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.training_data = []
        self.failure_history = []
        self.last_training_time = datetime.now()  # Initialize with current time
        self.training_interval = timedelta(hours=1)
        
    def collect_robot_data(self, robot_id: int, metrics: Dict):
        """Collect robot performance data for analysis"""
        timestamp = datetime.now()
        data = {
            'timestamp': timestamp,
            'robot_id': robot_id,
            'battery_voltage': metrics.get('battery_voltage', 0),
            'motor_temperature': metrics.get('motor_temperature', 0),
            'vibration_level': metrics.get('vibration_level', 0),
            'noise_level': metrics.get('noise_level', 0),
            'movement_accuracy': metrics.get('movement_accuracy', 1.0),
            'battery_cycles': metrics.get('battery_cycles', 0),
            'distance_traveled': metrics.get('distance_traveled', 0),
            'load_weight_avg': metrics.get('load_weight_avg', 0),
            'error_count': metrics.get('error_count', 0),
            'maintenance_needed': metrics.get('maintenance_needed', False)
        }
        self.training_data.append(data)
        
    def record_failure(self, robot_id: int, failure_type: str, metrics: Dict):
        """Record actual failure events"""
        self.failure_history.append({
            'timestamp': datetime.now(),
            'robot_id': robot_id,
            'failure_type': failure_type,
            'metrics': metrics
        })
        
    def train_models(self):
        """Train anomaly detection and failure prediction models"""
        if not self.training_data or datetime.now() - self.last_training_time < self.training_interval:
            return
            
        df = pd.DataFrame(self.training_data)
        feature_columns = [
            'battery_voltage', 'motor_temperature', 'vibration_level',
            'noise_level', 'movement_accuracy', 'battery_cycles',
            'distance_traveled', 'load_weight_avg', 'error_count'
        ]
        
        # Prepare data
        X = df[feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Train failure predictor if we have failure history
        if self.failure_history:
            failure_df = pd.DataFrame(self.failure_history)
            df['failure'] = df['timestamp'].apply(
                lambda x: any((x - f['timestamp']).total_seconds() < 3600
                            for f in self.failure_history)
            )
            y = df['failure']
            self.failure_predictor.fit(X_scaled, y)
            
        self.last_training_time = datetime.now()
        
    def predict_maintenance_needs(self, robot_id: int, current_metrics: Dict) -> Dict:
        """Predict maintenance needs for a robot"""
        if not self.training_data:
            return self._basic_maintenance_check(current_metrics)
            
        # Prepare input data
        feature_columns = [
            'battery_voltage', 'motor_temperature', 'vibration_level',
            'noise_level', 'movement_accuracy', 'battery_cycles',
            'distance_traveled', 'load_weight_avg', 'error_count'
        ]
        
        X = np.array([[current_metrics.get(col, 0) for col in feature_columns]])
        
        try:
            X_scaled = self.scaler.transform(X)
            # Get anomaly score
            anomaly_score = self.anomaly_detector.score_samples(X_scaled)[0]
            
            # Predict failure probability
            failure_prob = 0.0
            if len(self.failure_history) > 0:
                failure_prob = self.failure_predictor.predict_proba(X_scaled)[0][1]
        except (NotFittedError, ValueError):
            # If models aren't fitted yet, do basic check
            return self._basic_maintenance_check(current_metrics)
            
        # Calculate maintenance urgency
        urgency = self._calculate_maintenance_urgency(
            anomaly_score, failure_prob, current_metrics)
            
        return {
            'needs_maintenance': urgency > 0.7,
            'maintenance_urgency': urgency,
            'anomaly_score': anomaly_score,
            'failure_probability': failure_prob,
            'recommended_actions': self._get_recommended_actions(
                current_metrics, anomaly_score, failure_prob),
            'next_maintenance_window': self._suggest_maintenance_window(urgency)
        }
        
    def _basic_maintenance_check(self, metrics: Dict) -> Dict:
        """Perform basic maintenance check when ML models aren't ready"""
        # Define thresholds for basic checks
        thresholds = {
            'battery_voltage': 11.5,  # Volts
            'motor_temperature': 70,  # Celsius
            'vibration_level': 0.8,  # normalized
            'error_count': 5,  # errors per hour
            'battery_cycles': 500  # number of charge cycles
        }
        
        # Calculate how many metrics are above threshold
        issues = []
        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                issues.append(f"High {metric.replace('_', ' ')}")
                
        # Calculate basic urgency score
        urgency = len(issues) / len(thresholds)
        
        return {
            'needs_maintenance': urgency > 0.4,
            'maintenance_urgency': urgency,
            'anomaly_score': -1,  # Indicates no ML prediction
            'failure_probability': -1,  # Indicates no ML prediction
            'recommended_actions': issues if issues else ["Routine maintenance check"],
            'next_maintenance_window': datetime.now() + timedelta(hours=24 if urgency < 0.3 else 12)
        }
        
    def _calculate_maintenance_urgency(self, anomaly_score: float,
                                    failure_prob: float, metrics: Dict) -> float:
        """Calculate maintenance urgency score"""
        # Convert anomaly score to 0-1 scale (higher is more urgent)
        anomaly_urgency = 1 / (1 + np.exp(anomaly_score))
        
        # Combine different factors
        urgency = (
            0.3 * anomaly_urgency +
            0.3 * failure_prob +
            0.2 * (1 - metrics.get('movement_accuracy', 1.0)) +
            0.1 * (metrics.get('error_count', 0) / 10) +
            0.1 * (metrics.get('distance_traveled', 0) / 1000)
        )
        
        return min(1.0, urgency)
        
    def _get_recommended_actions(self, metrics: Dict, anomaly_score: float,
                               failure_prob: float) -> List[str]:
        """Get list of recommended maintenance actions"""
        actions = []
        
        if metrics.get('battery_voltage', 100) < 80:
            actions.append("Battery inspection and possible replacement")
        if metrics.get('motor_temperature', 0) > 50:
            actions.append("Motor cooling system check")
        if metrics.get('vibration_level', 0) > 0.7:
            actions.append("Mechanical components inspection")
        if metrics.get('movement_accuracy', 1.0) < 0.8:
            actions.append("Movement system calibration")
        if metrics.get('error_count', 0) > 5:
            actions.append("Software diagnostics")
        if metrics.get('distance_traveled', 0) > 1000:
            actions.append("Routine maintenance check")
            
        if anomaly_score < -0.5:
            actions.append("General system diagnostic")
        if failure_prob > 0.3:
            actions.append("Preventive maintenance recommended")
            
        return actions
        
    def _suggest_maintenance_window(self, urgency: float) -> datetime:
        """Suggest next maintenance window based on urgency"""
        if urgency > 0.9:
            return datetime.now()  # Immediate maintenance
        elif urgency > 0.7:
            return datetime.now() + timedelta(hours=24)
        elif urgency > 0.5:
            return datetime.now() + timedelta(days=3)
        else:
            return datetime.now() + timedelta(weeks=1)
            
    def save_models(self, path: str):
        """Save trained models to disk"""
        joblib.dump({
            'anomaly_detector': self.anomaly_detector,
            'failure_predictor': self.failure_predictor,
            'scaler': self.scaler,
            'last_training_time': self.last_training_time
        }, path)
        
    def load_models(self, path: str):
        """Load trained models from disk"""
        try:
            models = joblib.load(path)
            self.anomaly_detector = models['anomaly_detector']
            self.failure_predictor = models['failure_predictor']
            self.scaler = models['scaler']
            self.last_training_time = models['last_training_time']
            return True
        except:
            return False
