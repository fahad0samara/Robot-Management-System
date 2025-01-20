from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum, auto
import time

class TaskType(Enum):
    PICKUP = "pickup"
    DELIVERY = "delivery"
    TRANSFER = "transfer"

class Task:
    """Represents a task in the warehouse"""
    def __init__(self, task_type, start, end, priority=1.0):
        self.task_type = task_type
        self.start = start
        self.end = end
        self.priority = priority
        self.assigned_to = None
        self.completed = False
        self.creation_time = time.time()
        
    def assign(self, robot_id):
        """Assign task to a robot"""
        if not self.assigned_to:
            self.assigned_to = robot_id
            return True
        return False
        
    def complete(self):
        """Mark task as completed"""
        self.completed = True
        self.assigned_to = None
        
    def get_progress(self) -> float:
        """Get task progress (0-1)"""
        if self.completed:
            return 1.0
        elif not self.assigned_to:
            return 0.0
        return 0.5  # In progress
        
    def get_status(self) -> str:
        """Get task status string"""
        if self.completed:
            return "completed"
        elif self.assigned_to is not None:
            return "in_progress"
        return "waiting"
