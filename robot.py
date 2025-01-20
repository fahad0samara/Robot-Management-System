from enum import Enum
from typing import Tuple, List, Optional, Dict
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import random
import time
import asyncio

class RobotStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    PICKING = "picking"
    PLACING = "placing"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    LOW_BATTERY = "low_battery"

class TaskType(Enum):
    DELIVERY = "delivery"
    PICKUP = "pickup"
    TRANSFER = "transfer"

@dataclass
class RobotSpecifications:
    length: float = 1.2  # meters
    width: float = 0.8   # meters
    height: float = 1.5  # meters
    max_payload: float = 500.0  # kg
    max_speed: float = 2.0  # m/s
    battery_capacity: float = 100.0  # kWh
    charging_rate: float = 10.0  # kWh/hour
    min_battery_level: float = 15.0  # percentage
    speed_efficiency: float = 1.0  # Multiplier for speed tasks
    lifting_efficiency: float = 1.0  # Multiplier for heavy lifting
    battery_efficiency: float = 1.0  # Multiplier for battery consumption

class Robot:
    def __init__(self, id: int, position: Tuple[float, float] = (0, 0), specs: RobotSpecifications = RobotSpecifications()):
        self.id = id
        self.position = position
        self.specs = specs
        self.target = None
        self.status = "idle"
        self.current_task = None
        self.distance_traveled = 0.0
        self.tasks_completed = 0
        self.specialty = None  # Robot's specialization
        self.battery_level = 100.0
        self.nearby_robots = []
        self.collision_count = 0
        self.performance_metrics = {
            'tasks_completed': 0,
            'distance_traveled': 0.0,
            'collisions': 0,
            'collisions_avoided': 0
        }
        self.warehouse = None
        
    async def update(self, warehouse, time_step=0.016):
        """Update robot state"""
        self.warehouse = warehouse
        
        # Only drain battery when moving
        if self.status != "idle":
            # Calculate battery drain based on movement and task
            drain = 0.1  # Base drain
            if self.current_task:
                if self.current_task.task_type == "transfer":
                    drain *= 1.5  # Transfer tasks drain more battery
                elif self.current_task.task_type == "pickup":
                    drain *= 1.2  # Pickup tasks drain more battery
            
            # Specialty affects battery efficiency
            if self.specialty == self.current_task.task_type:
                drain *= 0.8  # More efficient at specialized tasks
            
            self.battery_level = max(0, self.battery_level - drain * time_step)
        else:
            # Minimal battery drain when idle
            self.battery_level = max(0, self.battery_level - 0.01 * time_step)
        
        # Handle low battery
        if self.battery_level < 20.0 and self.status not in ["charging", "need_charge"]:
            self.status = "need_charge"
            if self.current_task:
                self.current_task.assigned = False  # Release current task
                self.current_task = None
            nearest_station = warehouse.find_nearest_charging_station(self.position)
            if nearest_station:
                await self.go_to_charge(nearest_station)
                return
        
        # If in need_charge state, try to find charging station
        if self.status == "need_charge":
            nearest_station = warehouse.find_nearest_charging_station(self.position)
            if nearest_station:
                await self.go_to_charge(nearest_station)
                return
        
        # Execute current task if any and battery is sufficient
        if self.current_task and self.battery_level >= 20.0:
            task_complete = await self.execute_task(self.current_task, warehouse)
            if task_complete:
                self.warehouse.complete_task(self.current_task)  # Notify warehouse
                await self.complete_current_task()
        elif self.status == "charging":
            await self.charge(time_step)
            
    async def execute_task(self, task, warehouse):
        """Execute current task"""
        if not task or task.completed:
            self.status = "idle"
            self.current_task = None
            return True
            
        # Calculate distance to target
        if self.status == "moving_to_start":
            target = task.start
        elif self.status == "moving_to_end":
            target = task.end
        else:
            self.status = "moving_to_start"
            target = task.start
            
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        distance = (dx**2 + dy**2)**0.5
        
        # Check if we've reached the target
        if distance < 0.5:  # Reduced threshold for arrival
            if self.status == "moving_to_start":
                self.position = task.start  # Snap to exact position
                self.status = "moving_to_end"
                return False
            elif self.status == "moving_to_end":
                self.position = task.end  # Snap to exact position
                task.completed = True
                self.tasks_completed += 1
                self.performance_metrics['tasks_completed'] += 1
                self.current_task = None
                self.status = "idle"
                return True
                
        # Move towards target
        speed = self.specs.max_speed * 0.032  # Base speed
        if self.specialty == "speed":
            speed *= 2.0
            
        # Normalize direction
        if distance > 0:
            dx = dx / distance
            dy = dy / distance
            
            # Calculate new position
            new_x = self.position[0] + dx * speed
            new_y = self.position[1] + dy * speed
            
            # Update position and metrics
            old_pos = self.position
            self.position = (new_x, new_y)
            moved = ((new_x - old_pos[0])**2 + (new_y - old_pos[1])**2)**0.5
            if moved > 0:
                self.distance_traveled += moved
                self.performance_metrics['distance_traveled'] += moved
        
        return False
        
    async def seek_charging_station(self, warehouse):
        """Find and move to nearest charging station"""
        self.status = "seeking_charge"
        nearest_station = warehouse.get_nearest_charging_station(self.position)
        
        if nearest_station:
            arrived = await self.move_to(nearest_station, warehouse)
            if arrived:
                self.status = "charging"
                self.battery_level = min(100.0, self.battery_level + 15.0)  # Faster charging
                
        if self.battery_level >= 95.0:  # Full charge
            self.status = "idle"
            
    def update_nearby_robots(self, robots):
        """Update list of nearby robots for collision avoidance"""
        self.nearby_robots = []
        for robot in robots:
            if robot.id == self.id:
                continue
            dx = robot.position[0] - self.position[0]
            dy = robot.position[1] - self.position[1]
            distance = (dx**2 + dy**2)**0.5
            if distance < 3.0:  # Detection radius
                self.nearby_robots.append(robot)
                
    def _avoid_collisions(self, next_pos):
        """Adjust movement to avoid collisions with nearby robots"""
        if not self.nearby_robots:
            return next_pos
            
        adjusted_x, adjusted_y = next_pos[0], next_pos[1]
        
        for robot in self.nearby_robots:
            dx = next_pos[0] - robot.position[0]
            dy = next_pos[1] - robot.position[1]
            distance = (dx**2 + dy**2)**0.5
            
            if distance < 1.0:  # Collision avoidance radius
                # Calculate repulsion
                repulsion = 1.0 - min(1.0, distance)
                adjusted_x += dx * repulsion
                adjusted_y += dy * repulsion
                self.performance_metrics['collisions_avoided'] += 1
                
        return (adjusted_x, adjusted_y)
        
    def move_towards_target(self):
        """Move towards current target"""
        if not self.target:
            return
            
        # Calculate direction to target
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        distance = (dx**2 + dy**2)**0.5
        
        if distance < 0.1:  # Already at target
            return
            
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Calculate movement
        speed = self.specs.max_speed * 0.032  # Doubled speed
        if self.specialty == "speed":
            speed *= 2.0  # Increased speed multiplier
            
        # Update position
        new_x = self.position[0] + dx * speed
        new_y = self.position[1] + dy * speed
        
        # Update distance traveled
        moved_distance = ((new_x - self.position[0])**2 + (new_y - self.position[1])**2)**0.5
        self.distance_traveled += moved_distance
        
        # Set new position
        self.position = (new_x, new_y)
        
    async def move_to(self, target, warehouse):
        """Move towards target position"""
        if not target:
            return True
            
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        distance = (dx**2 + dy**2)**0.5
        
        if distance < 0.1:  # Close enough
            self.position = target  # Snap to exact position
            return True
            
        # Calculate movement
        speed = self.specs.max_speed * 0.032  # Doubled speed
        if self.specialty == "speed":
            speed *= 2.0  # Increased speed multiplier
            
        # Normalize and scale movement
        dx = (dx / distance) * speed
        dy = (dy / distance) * speed
        
        # Calculate new position
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        # Check for collisions with larger margin
        for robot in warehouse.robots:
            if robot != self and robot.position:
                rx = robot.position[0] - new_x
                ry = robot.position[1] - new_y
                dist = (rx**2 + ry**2)**0.5
                if dist < 0.8:  # Reduced collision margin
                    return False
        
        # Update metrics
        old_pos = self.position
        self.position = (new_x, new_y)
        moved = ((new_x - old_pos[0])**2 + (new_y - old_pos[1])**2)**0.5
        self.performance_metrics['distance_traveled'] += moved
        
        return False
        
    async def complete_current_task(self):
        """Complete current task and update metrics"""
        if self.current_task:
            self.tasks_completed += 1
            self.current_task = None
            self.status = "idle"
            self.target = None
            
    def assign_task(self, task):
        """Assign a task to this robot"""
        if task.assign(self.id):  # Only assign if task is unassigned
            self.current_task = task
            self.status = "moving_to_start"
            self.target = task.start
            return True
        return False
        
    def specialize(self, task_type):
        """Specialize the robot for a specific task type"""
        self.specialty = task_type
        
    def get_battery_level(self):
        """Get current battery level"""
        return self.battery_level
        
    def get_position(self):
        """Get current position"""
        return self.position
        
    def get_current_task(self):
        """Get the robot's current task"""
        return self.current_task
        
    def can_accept_task(self, task) -> bool:
        """Check if robot can accept a new task"""
        # If task is None, just check if robot is available
        if task is None:
            return (self.status == "idle" and 
                   self.current_task is None and 
                   self.battery_level > self.specs.min_battery_level)
        
        # Check basic conditions
        if (self.status != "idle" or 
            self.current_task is not None or 
            self.battery_level <= self.specs.min_battery_level):
            return False
            
        # Check weight capacity
        if task.weight > self.specs.max_payload:
            return False
            
        # Calculate total distance for task
        total_distance = (
            self._calculate_distance(self.position, task.start) +  # Distance to start
            self._calculate_distance(task.start, task.end)         # Distance to end
        )
        
        # Estimate battery usage (15% per unit distance + 30% safety margin)
        estimated_battery_usage = total_distance * 0.15 + 30
        if self.battery_level < estimated_battery_usage:
            return False
            
        # Check if robot is specialized and task matches specialty
        if self.specialty and task.task_type != self.specialty:
            return False
            
        return True
        
    def get_estimated_completion_time(self, task):
        """Estimate time to complete task based on distance and speed"""
        if not task:
            return float('inf')
            
        # Calculate total distance
        if self.status == "idle":
            d1 = self._calculate_distance(self.position, task.start)
            d2 = self._calculate_distance(task.start, task.end)
            total_distance = d1 + d2
        else:
            total_distance = self._calculate_distance(self.position, task.end)
            
        # Account for specialization
        speed = self.specs.max_speed
        if self.specialty == "speed":
            speed *= 1.5
            
        return total_distance / speed
        
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    async def go_to_charge(self, charging_station):
        """Go to charging station"""
        if charging_station:
            # Calculate distance to charging station
            dx = charging_station[0] - self.position[0]
            dy = charging_station[1] - self.position[1]
            distance = (dx**2 + dy**2)**0.5
            
            # If close enough to charging station, start charging
            if distance < 0.5:
                self.status = "charging"
                self.position = charging_station  # Snap to charging station
                self.target = None
            else:
                # Move towards charging station
                self.status = "need_charge"
                await self.move_to(charging_station, self.warehouse)
            
    async def charge(self, time_step=0.016):
        """Charge the robot's battery"""
        if self.status == "charging":
            charge_amount = 50.0 * time_step  # 50% per second
            self.battery_level = min(100.0, self.battery_level + charge_amount)
            
            # If fully charged, return to idle
            if self.battery_level >= 95.0:
                self.status = "idle"
                self.target = None
                return True
        return False
