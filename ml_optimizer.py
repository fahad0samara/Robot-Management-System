import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import joblib
from datetime import datetime, timedelta
import pandas as pd

class PathOptimizer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.training_data = []
        self.last_training_time = None
        
    def collect_path_data(self, path: List[Tuple[int, int]], completion_time: float,
                         obstacles: List[Tuple[int, int]], congestion_level: float,
                         success: bool):
        """Collect data about path execution for training"""
        path_length = len(path)
        turns = sum(1 for i in range(1, len(path)-1) if
                   path[i][0] != path[i-1][0] and path[i][1] != path[i-1][1])
        
        obstacle_proximity = min(
            ((p[0] - o[0])**2 + (p[1] - o[1])**2)**0.5
            for p in path
            for o in obstacles
        ) if obstacles else 999
        
        self.training_data.append({
            'path_length': path_length,
            'turns': turns,
            'obstacle_proximity': obstacle_proximity,
            'congestion_level': congestion_level,
            'completion_time': completion_time,
            'success': success
        })
        
    def train_model(self):
        """Train the path optimization model"""
        if not self.training_data:
            return
            
        df = pd.DataFrame(self.training_data)
        X = df[['path_length', 'turns', 'obstacle_proximity', 'congestion_level']]
        y = df['completion_time']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.last_training_time = datetime.now()
        
    def predict_path_time(self, path: List[Tuple[int, int]], 
                         obstacles: List[Tuple[int, int]], 
                         congestion_level: float) -> float:
        """Predict completion time for a given path"""
        if not self.last_training_time:
            return len(path) * 1.0  # Simple estimation if model not trained
            
        path_length = len(path)
        turns = sum(1 for i in range(1, len(path)-1) if
                   path[i][0] != path[i-1][0] and path[i][1] != path[i-1][1])
        
        obstacle_proximity = min(
            ((p[0] - o[0])**2 + (p[1] - o[1])**2)**0.5
            for p in path
            for o in obstacles
        ) if obstacles else 999
        
        X = np.array([[path_length, turns, obstacle_proximity, congestion_level]])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]

class TaskPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.training_data = []
        
    def collect_task_data(self, task_features: Dict, robot_features: Dict,
                         performance_metrics: Dict):
        """Collect data about task execution for training"""
        self.training_data.append({
            # Task features
            'priority': task_features['priority'].value,
            'weight': task_features['weight'],
            'distance': task_features['distance'],
            'deadline_margin': task_features.get('deadline_margin', 999999),
            
            # Robot features
            'battery_level': robot_features['battery_level'],
            'current_load': robot_features['current_load'],
            'efficiency': robot_features['efficiency'],
            
            # Performance metrics
            'completion_time': performance_metrics['completion_time'],
            'energy_used': performance_metrics['energy_used'],
            'success': performance_metrics['success']
        })
        
    def train_model(self):
        """Train the task prediction model"""
        if not self.training_data:
            return
            
        df = pd.DataFrame(self.training_data)
        
        # Prepare features and targets
        X = df[['priority', 'weight', 'distance', 'deadline_margin',
               'battery_level', 'current_load', 'efficiency']]
        y_time = df['completion_time']
        y_energy = df['energy_used']
        
        # Train completion time predictor
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_time)
        
    def predict_task_metrics(self, task_features: Dict, robot_features: Dict) -> Dict:
        """Predict metrics for a task-robot combination"""
        if not self.training_data:
            return {
                'estimated_completion_time': task_features['distance'] * 2,
                'estimated_energy_usage': task_features['distance'] * 0.1,
                'success_probability': 0.9
            }
            
        X = np.array([[
            task_features['priority'].value,
            task_features['weight'],
            task_features['distance'],
            task_features.get('deadline_margin', 999999),
            robot_features['battery_level'],
            robot_features['current_load'],
            robot_features['efficiency']
        ]])
        
        X_scaled = self.scaler.transform(X)
        completion_time = self.model.predict(X_scaled)[0]
        
        # Calculate success probability based on various factors
        success_prob = self._calculate_success_probability(
            task_features, robot_features, completion_time)
        
        return {
            'estimated_completion_time': completion_time,
            'estimated_energy_usage': completion_time * 0.1,
            'success_probability': success_prob
        }
        
    def _calculate_success_probability(self, task_features: Dict,
                                    robot_features: Dict,
                                    estimated_time: float) -> float:
        """Calculate probability of successful task completion"""
        # Battery sufficiency
        battery_factor = min(1.0, robot_features['battery_level'] /
                           (estimated_time * 0.1 * 2))  # Include safety margin
        
        # Load capacity
        load_factor = 1.0 if task_features['weight'] <= robot_features['efficiency'] * 100 else 0.5
        
        # Deadline feasibility
        deadline_factor = 1.0
        if 'deadline_margin' in task_features:
            deadline_factor = min(1.0, task_features['deadline_margin'] / estimated_time)
            
        return min(0.99, battery_factor * load_factor * deadline_factor)

class MLOptimizer:
    def __init__(self):
        self.path_optimizer = PathOptimizer()
        self.task_predictor = TaskPredictor()
        self.training_interval = timedelta(minutes=30)
        self.last_training = datetime.now()
        
    def update(self):
        """Periodic update and training of models"""
        current_time = datetime.now()
        if current_time - self.last_training >= self.training_interval:
            self.path_optimizer.train_model()
            self.task_predictor.train_model()
            self.last_training = current_time
            
    def optimize_path(self, paths: List[List[Tuple[int, int]]], 
                     obstacles: List[Tuple[int, int]], 
                     congestion_level: float) -> List[Tuple[int, int]]:
        """Choose the optimal path based on predictions"""
        if not paths:
            return []
            
        path_times = [
            self.path_optimizer.predict_path_time(path, obstacles, congestion_level)
            for path in paths
        ]
        
        return paths[np.argmin(path_times)]
        
    def predict_task_assignment(self, task_features: Dict,
                              available_robots: List[Dict]) -> Tuple[int, Dict]:
        """Predict the best robot for a task"""
        best_robot_id = None
        best_metrics = None
        best_score = float('-inf')
        
        for robot in available_robots:
            metrics = self.task_predictor.predict_task_metrics(task_features, robot)
            
            # Calculate overall score
            score = (metrics['success_probability'] * 100 -
                    metrics['estimated_completion_time'] * 0.1 -
                    metrics['estimated_energy_usage'] * 0.2)
                    
            if score > best_score:
                best_score = score
                best_robot_id = robot['id']
                best_metrics = metrics
                
        return best_robot_id, best_metrics
        
    def save_models(self, path: str):
        """Save trained models to disk"""
        joblib.dump({
            'path_optimizer_model': self.path_optimizer.model,
            'path_optimizer_scaler': self.path_optimizer.scaler,
            'task_predictor_model': self.task_predictor.model,
            'task_predictor_scaler': self.task_predictor.scaler
        }, path)
        
    def load_models(self, path: str):
        """Load trained models from disk"""
        try:
            models = joblib.load(path)
            self.path_optimizer.model = models['path_optimizer_model']
            self.path_optimizer.scaler = models['path_optimizer_scaler']
            self.task_predictor.model = models['task_predictor_model']
            self.task_predictor.scaler = models['task_predictor_scaler']
            return True
        except:
            return False
