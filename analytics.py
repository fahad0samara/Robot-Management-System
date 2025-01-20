from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime, timedelta
import pygame

@dataclass
class RobotMetrics:
    robot_id: int
    total_distance: float = 0.0
    tasks_completed: int = 0
    battery_usage: float = 0.0
    idle_time: float = 0.0
    working_time: float = 0.0
    collision_avoided: int = 0
    charging_cycles: int = 0
    avg_task_completion_time: float = 0.0
    total_weight_carried: float = 0.0

class WarehouseAnalytics:
    def __init__(self):
        self.metrics = {
            'tasks_completed': 0,
            'total_distance': 0.0,
            'collisions_avoided': 0,
            'battery_efficiency': 0.0,
            'robot_utilization': 0.0
        }
        self.robot_metrics: Dict[int, RobotMetrics] = {}
        self.warehouse_efficiency = 0.0
        self.heatmap_data = None
        self.warehouse_heatmap = None
        self.start_time = datetime.now()
        self.performance_history: List[Dict] = []
        
    def initialize_warehouse_heatmap(self, dimensions: Tuple[int, int]):
        """Initialize heatmap for tracking warehouse activity"""
        self.warehouse_heatmap = np.zeros(dimensions)
        
    def update_heatmap(self, position: Tuple[int, int]):
        """Update heatmap with new robot position"""
        if self.warehouse_heatmap is not None:
            self.warehouse_heatmap[position] += 1
            
    def register_robot(self, robot_id: int):
        """Register a new robot for tracking"""
        if robot_id not in self.robot_metrics:
            self.robot_metrics[robot_id] = RobotMetrics(robot_id=robot_id)
            
    def update_robot_metrics(self, robot_id: int, **metrics):
        """Update metrics for a specific robot"""
        if robot_id in self.robot_metrics:
            for key, value in metrics.items():
                if hasattr(self.robot_metrics[robot_id], key):
                    setattr(self.robot_metrics[robot_id], key, value)
                    
    def calculate_warehouse_efficiency(self) -> Dict:
        """Calculate overall warehouse efficiency metrics"""
        total_robots = len(self.robot_metrics)
        if total_robots == 0:
            return {}
            
        total_tasks = sum(r.tasks_completed for r in self.robot_metrics.values())
        total_distance = sum(r.total_distance for r in self.robot_metrics.values())
        total_working_time = sum(r.working_time for r in self.robot_metrics.values())
        total_idle_time = sum(r.idle_time for r in self.robot_metrics.values())
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "tasks_per_hour": (total_tasks / uptime) * 3600 if uptime > 0 else 0,
            "avg_robot_utilization": (total_working_time / (total_working_time + total_idle_time)) * 100 if (total_working_time + total_idle_time) > 0 else 0,
            "total_distance_covered": total_distance,
            "avg_tasks_per_robot": total_tasks / total_robots,
            "collision_avoidance_count": sum(r.collision_avoided for r in self.robot_metrics.values()),
            "total_weight_handled": sum(r.total_weight_carried for r in self.robot_metrics.values())
        }
        
    def get_robot_efficiency(self, robot_id: int) -> Dict:
        """Get efficiency metrics for a specific robot"""
        if robot_id not in self.robot_metrics:
            return {}
            
        metrics = self.robot_metrics[robot_id]
        total_time = metrics.working_time + metrics.idle_time
        
        return {
            "utilization_rate": (metrics.working_time / total_time * 100) if total_time > 0 else 0,
            "tasks_per_hour": (metrics.tasks_completed / metrics.working_time * 3600) if metrics.working_time > 0 else 0,
            "avg_task_time": metrics.avg_task_completion_time,
            "battery_efficiency": metrics.battery_usage / metrics.total_distance if metrics.total_distance > 0 else 0,
            "collision_avoidance_rate": metrics.collision_avoided,
            "charging_frequency": metrics.charging_cycles
        }
        
    def get_heatmap(self) -> np.ndarray:
        """Get the current warehouse activity heatmap"""
        return self.warehouse_heatmap if self.warehouse_heatmap is not None else np.array([])
        
    def update_metrics(self, robots, warehouse):
        """Update analytics metrics based on current state"""
        total_battery = 0
        total_tasks = 0
        total_distance = 0
        total_collisions = 0
        
        for robot in robots:
            total_battery += robot.current_battery / robot.specs.battery_capacity
            total_tasks += robot.performance_metrics['tasks_completed']
            total_distance += robot.performance_metrics['distance_traveled']
            total_collisions += robot.collision_count
            
        num_robots = len(robots)
        if num_robots > 0:
            self.metrics['battery_efficiency'] = total_battery / num_robots * 100
            self.metrics['tasks_completed'] = total_tasks
            self.metrics['total_distance'] = total_distance
            self.metrics['collisions_avoided'] = total_collisions
            self.metrics['robot_utilization'] = self._calculate_utilization(robots)
            
        # Update warehouse efficiency
        self.warehouse_efficiency = self._calculate_warehouse_efficiency(warehouse)
        
        # Update heatmap data
        self.heatmap_data = warehouse.generate_heatmap()
        
    def _calculate_utilization(self, robots):
        """Calculate robot utilization percentage"""
        active_robots = sum(1 for robot in robots if robot.status != "idle")
        return (active_robots / len(robots)) * 100 if robots else 0
        
    def _calculate_warehouse_efficiency(self, warehouse):
        """Calculate overall warehouse efficiency"""
        # Consider factors like:
        # - Storage utilization
        # - Robot utilization
        # - Task completion rate
        # - Battery efficiency
        weights = {
            'storage': 0.3,
            'robots': 0.3,
            'tasks': 0.2,
            'battery': 0.2
        }
        
        storage_util = self._calculate_storage_utilization(warehouse)
        robot_util = self.metrics['robot_utilization']
        task_efficiency = self.metrics['tasks_completed'] / max(1, len(warehouse.robots))
        battery_efficiency = self.metrics['battery_efficiency']
        
        efficiency = (
            weights['storage'] * storage_util +
            weights['robots'] * robot_util +
            weights['tasks'] * task_efficiency +
            weights['battery'] * battery_efficiency
        )
        
        return min(100.0, efficiency)
        
    def _calculate_storage_utilization(self, warehouse):
        """Calculate storage space utilization"""
        total_storage = sum(1 for y in range(warehouse.dimensions[1])
                          for x in range(warehouse.dimensions[0])
                          if warehouse.grid[y, x] == 1)
        used_storage = sum(1 for loc in warehouse.storage_locations.values()
                         if loc.current_load > 0)
        return (used_storage / total_storage * 100) if total_storage > 0 else 0
        
    def calculate_warehouse_efficiency(self):
        """Get the current warehouse efficiency"""
        return self.warehouse_efficiency
        
    def get_heatmap(self):
        """Get the current heatmap data"""
        return self.heatmap_data
        
    def draw_heatmap(self, screen, warehouse_size, cell_size):
        """Draw the heatmap on the given surface"""
        if self.heatmap_data is None:
            return
            
        # Normalize heatmap data
        max_value = np.max(self.heatmap_data)
        if max_value > 0:
            normalized_data = self.heatmap_data / max_value
        else:
            normalized_data = self.heatmap_data
            
        # Draw heatmap cells
        for y in range(warehouse_size[1]):
            for x in range(warehouse_size[0]):
                value = normalized_data[y, x]
                color = self._get_heatmap_color(value)
                rect = pygame.Rect(
                    x * cell_size,
                    y * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(screen, color, rect)
                
    def _get_heatmap_color(self, value):
        """Get color for heatmap based on value (0-1)"""
        # Blue (cold) to Red (hot) gradient
        r = int(255 * value)
        b = int(255 * (1 - value))
        return (r, 0, b, 128)  # Semi-transparent
