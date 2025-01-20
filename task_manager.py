from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
from datetime import datetime
import heapq

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: int
    priority: TaskPriority
    start_location: Tuple[int, int]
    end_location: Tuple[int, int]
    creation_time: datetime
    deadline: Optional[datetime] = None
    weight: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: Optional[int] = None
    completion_time: Optional[datetime] = None
    
    def __lt__(self, other):
        return (self.priority.value, self.creation_time) < (other.priority.value, other.creation_time)

class TaskManager:
    def __init__(self):
        self.task_queue = []  # Priority queue for tasks
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.task_counter = 0
        
    def create_task(self, priority: TaskPriority, start_loc: Tuple[int, int], 
                   end_loc: Tuple[int, int], weight: float = 0.0,
                   deadline: Optional[datetime] = None) -> Task:
        """Create a new task with given parameters"""
        task = Task(
            id=self.task_counter,
            priority=priority,
            start_location=start_loc,
            end_location=end_loc,
            creation_time=datetime.now(),
            deadline=deadline,
            weight=weight
        )
        self.task_counter += 1
        heapq.heappush(self.task_queue, task)
        return task
        
    def get_next_task(self, robot_id: int, robot_position: Tuple[int, int],
                      robot_capacity: float) -> Optional[Task]:
        """Get the next suitable task for a robot"""
        if not self.task_queue:
            return None
            
        # Create a copy of the queue for searching
        temp_queue = self.task_queue.copy()
        best_task = None
        best_score = float('-inf')
        
        while temp_queue:
            task = heapq.heappop(temp_queue)
            if task.status != TaskStatus.PENDING or task.weight > robot_capacity:
                continue
                
            # Calculate task score based on multiple factors
            score = self._calculate_task_score(task, robot_position)
            if score > best_score:
                best_score = score
                best_task = task
                
        if best_task:
            self.task_queue.remove(best_task)
            best_task.status = TaskStatus.ASSIGNED
            best_task.assigned_robot = robot_id
            
        return best_task
        
    def _calculate_task_score(self, task: Task, robot_position: Tuple[int, int]) -> float:
        """Calculate a score for task suitability"""
        # Priority factor
        priority_score = task.priority.value * 10
        
        # Distance factor
        distance_to_start = ((robot_position[0] - task.start_location[0])**2 + 
                           (robot_position[1] - task.start_location[1])**2)**0.5
        distance_score = 100 / (1 + distance_to_start)  # Inverse distance
        
        # Urgency factor
        urgency_score = 0
        if task.deadline:
            time_until_deadline = (task.deadline - datetime.now()).total_seconds()
            urgency_score = 100 if time_until_deadline < 60 else 50 / (1 + time_until_deadline/3600)
            
        return priority_score + distance_score + urgency_score
        
    def complete_task(self, task: Task):
        """Mark a task as completed"""
        task.status = TaskStatus.COMPLETED
        task.completion_time = datetime.now()
        self.completed_tasks.append(task)
        
    def fail_task(self, task: Task):
        """Mark a task as failed"""
        task.status = TaskStatus.FAILED
        self.failed_tasks.append(task)
        
    def get_task_metrics(self) -> dict:
        """Get metrics about task completion and performance"""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks) + len(self.task_queue)
        if total_tasks == 0:
            return {}
            
        completed_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        
        # Calculate average completion time
        avg_completion_time = 0
        if completed_count > 0:
            completion_times = [
                (task.completion_time - task.creation_time).total_seconds()
                for task in self.completed_tasks
            ]
            avg_completion_time = sum(completion_times) / len(completion_times)
            
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "pending_tasks": len(self.task_queue),
            "completion_rate": (completed_count / total_tasks) * 100,
            "average_completion_time": avg_completion_time,
            "tasks_by_priority": {
                priority.name: sum(1 for t in self.completed_tasks if t.priority == priority)
                for priority in TaskPriority
            }
        }
