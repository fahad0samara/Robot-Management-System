import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import random
from task import TaskType  # Import TaskType enum
import threading
import time

@dataclass
class StorageLocation:
    def __init__(self, position, id=None, capacity=1000.0, current_load=0.0, item_type="general"):
        self.id = str(id) if id is not None else f"storage_{position[0]}_{position[1]}"
        self.position = position
        self.capacity = capacity
        self.current_load = current_load
        self.item_type = item_type
        self.occupied = False
        self.status = "available"
        
    def can_store(self, item_weight):
        """Check if location can store additional weight"""
        return self.current_load + item_weight <= self.capacity
        
    def add_item(self, weight):
        """Add item weight to storage"""
        if self.can_store(weight):
            self.current_load += weight
            return True
        return False
        
    def remove_item(self, weight):
        """Remove item weight from storage"""
        if self.current_load >= weight:
            self.current_load -= weight
            return True
        return False
        
    def get_available_capacity(self):
        """Get remaining storage capacity"""
        return self.capacity - self.current_load
        
    def is_full(self):
        """Check if storage is at capacity"""
        return self.current_load >= self.capacity
        
    def is_empty(self):
        """Check if storage is empty"""
        return self.current_load == 0.0

class Task:
    """A task that needs to be completed by a robot"""
    _next_id = 0  # Class variable for generating unique IDs
    _task_set = set()  # Set of all active tasks
    _task_lock = threading.Lock()  # Lock for task operations
    _assignment_lock = threading.Lock()  # Lock for assignment operations
    _assigned_tasks = {}  # Track task assignments globally
    _assigned_coords = set()  # Track assigned coordinates
    
    def __init__(self, task_type: TaskType, start: Tuple[float, float], 
                 end: Tuple[float, float], priority: float = 1.0, 
                 weight: float = 10.0, creation_time: float = None):
        with Task._task_lock:
            Task._next_id += 1
            self.task_type = task_type
            self.start = start
            self.end = end
            self.priority = priority
            self.weight = weight
            self.id = Task._next_id  # Unique identifier for each task
            self._assigned_to = None  # Use private variable for assignment
            self._locked = False  # Lock to prevent race conditions
            self.completed = False
            self.creation_time = creation_time
            Task._task_set.add(self)  # Add task to global set
        
    @property
    def assigned_to(self):
        """Get assignment status"""
        with Task._assignment_lock:
            return self.id in Task._assigned_tasks
            
    def is_available(self):
        """Check if task is available for assignment"""
        with Task._assignment_lock:
            if self.id in Task._assigned_tasks or self.completed:
                return False
                
            # Check if coordinates are in use
            if self.start in Task._assigned_coords or self.end in Task._assigned_coords:
                return False
                
            # Check if task type + coordinates exist
            for t_id, r_id in Task._assigned_tasks.items():
                assigned_task = next((t for t in Task._task_set if t.id == t_id), None)
                if assigned_task:
                    if (assigned_task.task_type == self.task_type and
                        assigned_task.start == self.start and
                        assigned_task.end == self.end):
                        return False
                        
            return True
        
    def lock(self) -> bool:
        """Try to lock the task for assignment. Returns True if successful."""
        with Task._assignment_lock:
            if self.is_available():
                with Task._task_lock:
                    if not self._locked and self in Task._task_set:
                        self._locked = True
                        return True
            return False
        
    def unlock(self):
        """Unlock the task"""
        with Task._task_lock:
            self._locked = False
        
    def assign(self, robot_id) -> bool:
        """Assign task to a robot. Returns True if successful."""
        with Task._assignment_lock:
            if self.is_available():
                with Task._task_lock:
                    if self._locked and self in Task._task_set:
                        Task._assigned_tasks[self.id] = robot_id
                        Task._assigned_coords.add(self.start)
                        Task._assigned_coords.add(self.end)
                        self._assigned_to = robot_id
                        self._locked = False  # Release lock after assignment
                        return True
            return False
            
    def complete(self):
        """Mark task as completed and remove from active set"""
        with Task._assignment_lock:
            if self.id in Task._assigned_tasks:
                del Task._assigned_tasks[self.id]
                Task._assigned_coords.discard(self.start)
                Task._assigned_coords.discard(self.end)
                with Task._task_lock:
                    self.completed = True
                    Task._task_set.discard(self)
        
    def __eq__(self, other):
        """Tasks are equal if they have same ID"""
        if not isinstance(other, Task):
            return False
        return self.id == other.id
                
    def __hash__(self):
        """Hash based on unique ID"""
        return hash(self.id)

class Warehouse:
    def __init__(self, dimensions=(20, 20)):
        """Initialize warehouse with given dimensions"""
        self.width, self.height = dimensions
        self.robots = []
        self.tasks = []
        self.waiting_tasks = []  # Track unassigned tasks
        self.completed_tasks = []
        self.storage_locations = {}
        self.charging_stations = [
            (0, 0),  # Corner charging stations
            (self.width-1, 0),
            (0, self.height-1),
            (self.width-1, self.height-1),
            (self.width//2, self.height//2)  # Center charging station
        ]
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.visit_counts = np.zeros(dimensions[::-1], dtype=int)
        self.cell_visits = defaultdict(int)
        self._task_lock = threading.Lock()  # Lock for task assignment
        self._assigned_tasks = {}  # Track task assignments by robot ID
        self._task_queue = []  # Queue for task assignment
        
        self.initialize_layout()
    
    def initialize_layout(self):
        """Initialize warehouse layout with charging stations and storage areas"""
        # Add charging stations along the edges
        for pos in self.charging_stations:
            self.add_charging_station(pos)
            
        # Create storage areas
        # Main storage area (middle section)
        for x in range(5, 15):
            for y in range(5, 10):
                self.add_storage_location((x, y))
                
        # Processing area (right section)
        for x in range(15, 18):
            for y in range(3, 8):
                self.add_storage_location((x, y))
                
        # Delivery area (top section)
        for x in range(5, 15):
            for y in range(15, 18):
                self.add_storage_location((x, y))
                
    def add_charging_station(self, position):
        """Add a charging station at the specified position"""
        if self.is_valid_position(position):
            self.grid[position[1]][position[0]] = 2  # 2 represents charging station
            
    def add_storage_location(self, position):
        """Add a storage location at the specified position"""
        if self.is_valid_position(position):
            storage_id = len(self.storage_locations)
            self.storage_locations[position] = StorageLocation(
                position=position,
                id=storage_id,
                capacity=1000.0
            )
            self.grid[position[1]][position[0]] = 1  # 1 represents storage
            
    def is_valid_position(self, position):
        """Check if a position is within warehouse bounds"""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height
        
    def is_storage_location(self, position):
        """Check if a position is a storage location"""
        return position in self.storage_locations
        
    def is_charging_station(self, position):
        """Check if a position is a charging station"""
        return position in self.charging_stations
        
    def find_nearest_charging_station(self, position):
        """Find nearest available charging station"""
        if not self.charging_stations:
            return None
            
        # Find nearest station
        nearest = min(self.charging_stations,
                     key=lambda station: ((station[0] - position[0])**2 +
                                        (station[1] - position[1])**2))
        return nearest
        
    def get_random_storage_location(self):
        """Get a random storage location"""
        if not self.storage_locations:
            return None
        return random.choice(list(self.storage_locations.keys()))
        
    def add_robot(self, robot):
        """Add a robot to the warehouse"""
        self.robots.append(robot)
    
    def remove_robot(self, robot):
        """Remove a robot from the warehouse"""
        if robot in self.robots:
            self.robots.remove(robot)
    
    def add_task(self, task):
        """Add a new task to the warehouse"""
        self.waiting_tasks.append(task)
        
    def get_available_tasks(self):
        """Get list of available (unassigned) tasks"""
        return [t for t in self.waiting_tasks if not t.assigned_to]
        
    def complete_task(self, task):
        """Mark a task as completed"""
        if task in self.waiting_tasks:
            self.waiting_tasks.remove(task)
        self.completed_tasks.append(task)
            
    def get_tasks(self):
        """Get all tasks in the warehouse"""
        return self.waiting_tasks + self.completed_tasks
        
    def get_completed_tasks(self):
        """Get completed tasks"""
        return self.completed_tasks
        
    def get_efficiency(self):
        """Calculate warehouse efficiency"""
        total_tasks = len(self.completed_tasks) + len(self.tasks)
        if total_tasks == 0:
            return 0.0
        return (len(self.completed_tasks) / total_tasks) * 100
        
    def find_nearest_charging_station(self, position: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find nearest charging station to given position"""
        nearest = None
        min_distance = float('inf')
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 2:  # Charging station
                    distance = ((position[0] - x)**2 + (position[1] - y)**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        nearest = (x, y)
                        
        return nearest
    
    def get_nearest_charging_station(self, position: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Find nearest charging station to given position"""
        x, y = position
        charging_stations = [(i, j) for i in range(self.width) for j in range(self.height) if self.grid[j][i] == 2]
        if len(charging_stations) == 0:
            return None
        
        # Find nearest station
        nearest = min(charging_stations, key=lambda s: ((s[0] - x) ** 2 + (s[1] - y) ** 2) ** 0.5)
        return (nearest[0], nearest[1])  # Return as (x, y)
    
    def get_cell_visits(self, position: Tuple[int, int]) -> int:
        """Get the number of times a cell has been visited"""
        return self.cell_visits[position]
    
    def increment_cell_visits(self, position: Tuple[int, int]):
        """Increment visit count for cell"""
        x, y = position
        if self.is_valid_position((x, y)):
            self.cell_visits[position] += 1
    
    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Check if position is within warehouse bounds and not blocked"""
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.grid[int(y)][int(x)] != 1  # Not a storage location
        
    def check_robot_position_valid(self, position: Tuple[float, float]) -> bool:
        """Check if position is valid for a robot"""
        # First check if position is valid
        if not self.is_valid_position(position):
            return False
            
        # Check if position is too close to other robots
        MIN_ROBOT_DISTANCE = 1.0
        for robot in self.robots:
            if robot.position is not None:
                dist = np.sqrt((position[0] - robot.position[0])**2 + 
                             (position[1] - robot.position[1])**2)
                if dist < MIN_ROBOT_DISTANCE:
                    return False
                    
        return True
    
    def is_occupied(self, position: Tuple[float, float]) -> bool:
        """Check if position is occupied by storage location"""
        if not self.is_valid_position(position):
            return True
        x, y = int(position[0]), int(position[1])
        return self.grid[y][x] == 1  # Only storage locations block movement
    
    def get_valid_moves(self, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Get list of valid moves from current position"""
        valid_moves = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:  # Four directions
            new_pos = (position[0] + dx, position[1] + dy)
            if self.is_valid_position(new_pos) and not self.is_occupied(new_pos):
                valid_moves.append(new_pos)
        return valid_moves

    def manage_charging(self):
        """Manage robot charging"""
        charging_spots = set()  # Track occupied charging spots
        
        # First, handle robots already charging or moving to charge
        for robot in self.robots:
            if robot.status in ["charging", "moving_to_charge"]:
                if robot.status == "charging":
                    charging_spots.add(tuple(robot.position))
                elif robot.path:  # Moving to charge
                    charging_spots.add(tuple(robot.path[-1]))
        
        # Then handle robots that need charging
        for robot in self.robots:
            if robot.status == "need_charge":
                # Find best available charging station
                best_station = None
                min_cost = float('inf')
                
                for station in self.charging_stations:
                    if tuple(station) in charging_spots:
                        continue
                        
                    # Calculate distance to station
                    distance = ((robot.position[0] - station[0])**2 + 
                              (robot.position[1] - station[1])**2)**0.5
                    
                    # Calculate energy needed to reach station
                    energy_needed = distance * 0.5
                    
                    # Skip if robot doesn't have enough battery to reach
                    if energy_needed >= robot.current_battery:
                        continue
                    
                    # Cost function considers both distance and current occupancy
                    cost = distance
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_station = station
                
                if best_station:
                    robot.status = "moving_to_charge"
                    robot.path = robot._plan_path(robot.position, best_station)
                    charging_spots.add(tuple(best_station))
                    
    def _validate_task_assignment(self, task, robot_id):
        """Validate that a task can be assigned to a robot"""
        with self._task_lock:
            # Check if task is already assigned
            if task.assigned_to:
                return False
                
            # Check if task coordinates are already in use
            for r in self.robots:
                if r.current_task and r.id != robot_id:
                    if (r.current_task.start == task.start and 
                        r.current_task.end == task.end):
                        return False
                        
            # Check if task type + coordinates combination exists
            task_signature = (task.task_type, task.start, task.end)
            for r in self.robots:
                if r.current_task and r.id != robot_id:
                    if (r.current_task.task_type == task.task_type and
                        r.current_task.start == task.start and
                        r.current_task.end == task.end):
                        return False
                        
            # Check if task ID is already assigned
            if task.id in [r.current_task.id for r in self.robots if r.current_task]:
                return False
                
            return True
            
    def _calculate_task_score(self, task, robot, current_time):
        """Calculate score for task-robot assignment"""
        score = 0
        
        # Distance penalty (negative because closer is better)
        distance_to_start = robot._calculate_distance(robot.position, task.start)
        distance_penalty = -distance_to_start * 10
        score += distance_penalty
        
        # Specialty bonus
        if robot.specialty == task.task_type:
            score += 200  # High bonus for specialized tasks
        
        # Workload balancing
        tasks_behind = max(0, sum(r.tasks_completed for r in self.robots) / len(self.robots) - robot.tasks_completed)
        score += tasks_behind * 50  # Bonus for robots with fewer completed tasks
        
        # Battery efficiency bonus
        battery_margin = robot.battery_level - (robot._calculate_distance(robot.position, task.start) + 
                                              robot._calculate_distance(task.start, task.end)) * 0.15 - 30
        if battery_margin > 0:
            score += battery_margin * 1.2  # Reward for having extra battery
        
        # Waiting time bonus
        wait_time = current_time - task.creation_time
        score += min(wait_time * 15, 150)  # Cap at 150 points
        
        # Priority bonus
        score += task.priority * 100  # Scale priority to be significant
        
        # Weight efficiency bonus
        weight_capacity_used = task.weight / robot.specs.max_payload
        if weight_capacity_used <= 0.8:  # Prefer tasks that don't max out capacity
            score += (0.8 - weight_capacity_used) * 50
        
        # Proximity bonus for nearby tasks
        if distance_to_start < 5:
            score += 75
        
        # Idle time bonus
        idle_time = current_time - getattr(robot, 'idle_start_time', current_time)
        score += min(idle_time * 20, 200)  # Cap at 200 points
        
        return score
        
    def assign_tasks(self):
        """Assign available tasks to idle robots"""
        current_time = time.time()
        
        with Task._assignment_lock:  # Use global task assignment lock
            # Track all task assignments
            task_assignments = {}  # Map of task ID to (robot_id, task)
            
            # Get current task assignments
            for robot in self.robots:
                if robot.current_task:
                    task = robot.current_task
                    task_assignments[task.id] = (robot.id, task)
                    
            # Get list of available tasks
            available_tasks = []
            for task in self.tasks:
                # Skip if task is completed or assigned
                if task.completed or task.id in Task._assigned_tasks:
                    continue
                    
                # Skip if task coordinates are in use
                coordinates_in_use = False
                for other_task in self.tasks:
                    if other_task.id in task_assignments:
                        if (other_task.start == task.start or 
                            other_task.end == task.end or
                            (other_task.task_type == task.task_type and
                             other_task.start == task.start and
                             other_task.end == task.end)):
                            coordinates_in_use = True
                            break
                            
                if not coordinates_in_use:
                    available_tasks.append(task)
            
            # Get available robots
            idle_robots = [r for r in self.robots if r.status == "idle"]
            idle_robots.sort(key=lambda r: (
                getattr(r, 'idle_start_time', 0),
                -r.tasks_completed
            ))
            
            # Process each idle robot
            for robot in idle_robots:
                if not robot.can_accept_task(None):
                    continue
                    
                best_score = float('-inf')
                best_task = None
                
                # Score each available task
                for task in available_tasks[:]:
                    # Skip if task is no longer available
                    if task.id in Task._assigned_tasks:
                        continue
                        
                    # Skip if robot can't handle task
                    if not robot.can_accept_task(task):
                        continue
                        
                    # Calculate task score
                    score = self._calculate_task_score(task, robot, current_time)
                    
                    if score > best_score:
                        best_score = score
                        best_task = task
                
                # Try to assign best task
                if best_task and best_task.id not in Task._assigned_tasks:
                    # Double check task is still available
                    if best_task.id not in Task._assigned_tasks:
                        if best_task.assign(robot.id):
                            robot.current_task = best_task
                            robot.status = "moving_to_start"
                            robot.path = robot._plan_path(robot.position, best_task.start)
                            robot.idle_start_time = 0
                            available_tasks.remove(best_task)
                            task_assignments[best_task.id] = (robot.id, best_task)
                
    def _force_task_assignment(self, robot):
        """Force task assignment for robots that have been idle too long"""
        if robot.status != "idle":
            return
            
        # Track assigned tasks and coordinates
        assigned_tasks = set()
        assigned_coords = set()
        task_types = set()
        
        for r in self.robots:
            if r.current_task:
                assigned_tasks.add(r.current_task)
                assigned_coords.add(r.current_task.start)
                assigned_coords.add(r.current_task.end)
                task_types.add((r.current_task.task_type, 
                              r.current_task.start, 
                              r.current_task.end))
        
        # Get valid unassigned tasks
        unassigned_tasks = [
            t for t in self.tasks 
            if not t.assigned_to and not t.completed and
            t not in assigned_tasks and
            t.start not in assigned_coords and
            t.end not in assigned_coords and
            (t.task_type, t.start, t.end) not in task_types
        ]
        
        if not unassigned_tasks:
            return
            
        # Find best task considering distance and waiting time
        current_time = time.time()
        best_task = min(
            unassigned_tasks,
            key=lambda t: (
                ((robot.position[0] - t.start[0])**2 + 
                 (robot.position[1] - t.start[1])**2)**0.5 * 0.3 -  # Distance penalty
                (current_time - t.creation_time) * 0.7  # Waiting time bonus
            )
        )
        
        best_task.assign_to_robot(robot.id)
        robot.current_task = best_task
        robot.status = "moving_to_start"
        robot.path = robot._plan_path(robot.position, best_task.start)
        robot.idle_start_time = 0

    def generate_tasks(self):
        """Generate new tasks based on warehouse state"""
        current_time = time.time()
        
        # Calculate optimal number of tasks based on warehouse state
        available_robots = len([r for r in self.robots if r.status not in ["charging", "moving_to_charge", "need_charge"]])
        active_tasks = len([t for t in self.tasks if not t.completed])
        waiting_tasks = len([t for t in self.tasks if not t.assigned_to and not t.completed])
        
        # Adjust task generation based on waiting tasks
        if waiting_tasks >= available_robots * 3:  # Cap waiting tasks
            return
            
        optimal_tasks = max(2, available_robots * 3)  # Keep 3 tasks per available robot, minimum 2
        
        # Generate new tasks if needed
        if active_tasks < optimal_tasks:
            tasks_to_generate = min(optimal_tasks - active_tasks, 5)  # Generate at most 5 at once
            
            for _ in range(tasks_to_generate):
                # Track assigned coordinates
                assigned_coords = set()
                for task in self.tasks:
                    if not task.completed:
                        assigned_coords.add(task.start)
                        assigned_coords.add(task.end)
                
                # Determine task type based on current distribution
                task_types = [TaskType.PICKUP, TaskType.DELIVERY, TaskType.TRANSFER]
                type_counts = {t: len([task for task in self.tasks 
                                     if task.task_type == t and not task.completed])
                             for t in task_types}
                
                # Prefer underrepresented task types
                min_count = min(type_counts.values()) if type_counts else 0
                preferred_types = [t for t, c in type_counts.items() 
                                 if c == min_count]
                task_type = random.choice(preferred_types or task_types)
                
                # Generate task locations based on type
                for attempt in range(10):  # Try up to 10 times to find valid locations
                    if task_type == TaskType.PICKUP:
                        start = self._get_random_storage_location()
                        end = self._get_random_processing_location()
                        weight = random.uniform(10.0, 80.0)  # Light items
                        priority = random.uniform(0.5, 1.0)
                    elif task_type == TaskType.DELIVERY:
                        start = self._get_random_processing_location()
                        end = self._get_random_delivery_location()
                        weight = random.uniform(50.0, 150.0)  # Medium items
                        priority = random.uniform(0.7, 1.2)
                    else:  # transfer
                        start = self._get_random_storage_location()
                        end = self._get_random_storage_location()
                        while end == start:  # Ensure different locations
                            end = self._get_random_storage_location()
                        weight = random.uniform(100.0, 300.0)  # Heavy items
                        priority = random.uniform(0.8, 1.5)
                    
                    # Check if locations are valid and not already assigned
                    if (start not in assigned_coords and 
                        end not in assigned_coords):
                        break
                else:
                    continue  # Skip if no valid locations found
                
                # Create and add task
                task = Task(
                    task_type=task_type,
                    start=start,
                    end=end,
                    priority=priority,
                    weight=weight,
                    creation_time=current_time
                )
                self.add_task(task)
                
    def _get_random_storage_location(self) -> Tuple[int, int]:
        """Get a random storage location"""
        storage_areas = [
            (x, y) for x in range(5, 15)
            for y in range(5, 10)
        ]
        return random.choice(storage_areas)
        
    def _get_random_processing_location(self) -> Tuple[int, int]:
        """Get a random processing location"""
        processing_areas = [
            (x, y) for x in range(15, 20)
            for y in range(3, 8)
        ]
        return random.choice(processing_areas)
        
    def _get_random_delivery_location(self) -> Tuple[int, int]:
        """Get a random delivery location"""
        delivery_areas = [
            (x, y) for x in range(5, 15)
            for y in range(15, 18)
        ]
        return random.choice(delivery_areas)

    async def optimize_layout(self, inventory_data: Dict):
        prompt = f"""
        Optimize warehouse layout for efficiency:
        Current dimensions: {self.width}x{self.height}
        Current storage locations: {len(self.storage_locations)}
        Inventory data: {inventory_data}
        Optimize for:
        - Pick efficiency
        - Storage density
        - Robot movement patterns
        - Charging station placement
        """
        response = await self.model.generate_content(prompt)
        self._apply_layout_optimization(response.text)

    def _apply_layout_optimization(self, optimization_plan: str):
        pass

    def update_robot_position(self, robot, new_position):
        """Update robot position if valid"""
        if robot in self.robots:
            if self.is_valid_position(new_position) and not self.is_occupied(new_position):
                old_pos = robot.get_position_tuple()
                robot.position = np.array(new_position)
                new_pos = robot.get_position_tuple()
                self.cell_visits[new_pos] += 1
                return True
        return False

    def find_nearest_charging_station(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Find nearest charging station to given position"""
        if not self.charging_stations:
            return None
        distances = [np.linalg.norm(np.array(position) - np.array(station))
                    for station in self.charging_stations]
        return self.charging_stations[np.argmin(distances)]

    def get_path_to_location(self, start: Tuple[float, float], 
                           end: Tuple[float, float]) -> List[Tuple[float, float]]:
        return [start, end]  

    def update_storage_location(self, location_id, weight, item_type):
        """Update a storage location with new item"""
        for location in self.storage_locations.values():
            if location.id == str(location_id):
                location.add_item(weight)
                location.item_type = item_type
                return True
        return False

    def get_storage_utilization(self) -> Dict:
        total_capacity = sum(loc.capacity for loc in self.storage_locations.values())
        total_used = sum(loc.current_load for loc in self.storage_locations.values())
        return {
            'total_capacity': total_capacity,
            'total_used': total_used,
            'utilization_percentage': (total_used / total_capacity) * 100 if total_capacity > 0 else 0,
            'available_locations': sum(1 for loc in self.storage_locations.values() 
                                    if loc.status == "available"),
            'total_locations': len(self.storage_locations)
        }

    def generate_heatmap(self) -> np.ndarray:
        """Generate a heatmap of warehouse activity"""
        heatmap = np.zeros((self.height, self.width), dtype=float)  # height x width
        
        # Add robot positions to heatmap
        for robot in self.robots:
            pos = robot.get_position_tuple()
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                heatmap[y, x] += 1
                
                # Add influence to surrounding cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.width and 
                            0 <= ny < self.height):
                            heatmap[ny, nx] += 0.5 / (abs(dx) + abs(dy) + 1)
        
        # Add storage locations with items
        for loc in self.storage_locations.values():
            if loc.current_load > 0:
                x, y = int(loc.position[0]), int(loc.position[1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    heatmap[y, x] += 0.5
        
        # Add charging stations
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 2:  # Charging station
                    heatmap[y, x] += 0.3
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap

    def get_random_empty_position(self) -> Tuple[float, float]:
        """Get a random empty position in the warehouse"""
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random position
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            # Check if position is empty (not storage or charging)
            if self.grid[y][x] == 0:
                # Check if no robot is at this position
                position_occupied = False
                for robot in self.robots:
                    robot_pos = robot.get_position_tuple()
                    if abs(robot_pos[0] - x) < 1 and abs(robot_pos[1] - y) < 1:
                        position_occupied = True
                        break
                
                if not position_occupied:
                    return (float(x), float(y))
        
        raise RuntimeError("Could not find empty position after maximum attempts")

    def check_position_empty(self, position: Tuple[float, float]) -> bool:
        """Check if a position is empty (no storage, charging station, or robot)"""
        x, y = int(position[0]), int(position[1])
        
        # Check grid bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
            
        # Check if position is empty in grid
        if self.grid[y][x] != 0:
            return False
            
        # Check for robots
        for robot in self.robots:
            robot_pos = robot.get_position_tuple()
            if abs(robot_pos[0] - position[0]) < 1 and abs(robot_pos[1] - position[1]) < 1:
                return False
                
        return True

    def _complete_task(self, robot):
        """Complete the current task for a robot"""
        if robot.current_task:
            robot.current_task.complete()  # Remove from active task set
            self.completed_tasks.append(robot.current_task)
            robot.tasks_completed += 1
            robot.current_task = None
            robot.status = "idle"
            robot.idle_start_time = time.time()
