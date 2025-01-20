import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
from dataclasses import dataclass
from enum import Enum
import random
import time
import threading
import math
import heapq
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Robot status enum
class RobotStatus(Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    CHARGING = "CHARGING"
    LOW_BATTERY = "LOW_BATTERY"
    WAITING = "WAITING"
    ERROR = "ERROR"
    PICKING = "PICKING"
    DELIVERING = "DELIVERING"

# Task types
class TaskType(Enum):
    MOVE = "MOVE"
    PICKUP = "PICKUP"
    DELIVER = "DELIVER"
    CHARGE = "CHARGE"
    WAIT = "WAIT"

class Task:
    def __init__(self, type: TaskType, target_pos: Tuple[int, int], item: str = None):
        self.type = type
        self.target_pos = target_pos
        self.item = item
        self.completed = False
        self.start_time = time.time()

class PathFinder:
    def __init__(self, warehouse):
        self.warehouse = warehouse

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        if start == goal:
            return [start]
            
        # A* pathfinding
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            # Check all neighbors (including diagonals)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                        
                    next_pos = (current[0] + dx, current[1] + dy)
                    
                    # Check if position is valid
                    if not self.warehouse.is_valid_position(next_pos):
                        continue
                        
                    # Calculate new cost (diagonal movement costs more)
                    new_cost = cost_so_far[current]
                    if dx != 0 and dy != 0:
                        new_cost += 1.4  # sqrt(2)
                    else:
                        new_cost += 1
                        
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + math.sqrt((goal[0] - next_pos[0])**2 + (goal[1] - next_pos[1])**2)
                        heapq.heappush(frontier, (priority, next_pos))
                        came_from[next_pos] = current
        
        # Reconstruct path
        if goal not in came_from:
            return []
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path

class Robot:
    def __init__(self, id: int, position: Tuple[float, float], speed: float):
        self.id = id
        self.position = position
        self.battery = 100.0
        self.status = RobotStatus.IDLE
        self.path = []
        self.speed = speed
        self.carrying_item = None
        self.tasks: List[Task] = []
        self.error_count = 0
        self.total_distance = 0
        self.task_history = []
        
    def add_task(self, task: Task):
        self.tasks.append(task)
        if self.status == RobotStatus.IDLE:
            self.status = RobotStatus.WAITING
            
    def update(self, warehouse):
        # Battery management
        if self.status in [RobotStatus.MOVING, RobotStatus.PICKING, RobotStatus.DELIVERING]:
            self.battery = max(0, self.battery - 0.2)
            self.total_distance += self.speed
        elif self.status == RobotStatus.CHARGING:
            self.battery = min(100, self.battery + 1.0)
            if self.battery >= 100:
                self.status = RobotStatus.IDLE
                self.last_charge = time.time()
        
        # Error simulation
        if random.random() < 0.001:  # 0.1% chance of error
            self.error_count += 1
            self.status = RobotStatus.ERROR
            return
        
        # Low battery handling
        if self.battery <= 20 and self.status not in [RobotStatus.CHARGING, RobotStatus.ERROR]:
            self.status = RobotStatus.LOW_BATTERY
            charging_stations = warehouse.get_charging_stations()
            if charging_stations:
                nearest = min(charging_stations, key=lambda pos: 
                    ((pos[0] - self.position[0])**2 + (pos[1] - self.position[1])**2)**0.5)
                path = PathFinder(warehouse).find_path(
                    (int(self.position[0]), int(self.position[1])), 
                    nearest
                )
                if path:
                    self.path = path
                    self.tasks = [Task(TaskType.CHARGE, nearest)]
        
        # Task execution
        if self.tasks and self.battery > 20 and self.status != RobotStatus.ERROR:
            current_task = self.tasks[0]
            
            # If we don't have a path, calculate one
            if not self.path:
                path = PathFinder(warehouse).find_path(
                    (int(self.position[0]), int(self.position[1])), 
                    current_task.target_pos
                )
                if path:
                    self.path = path
            
            # Check if we've reached the target
            if self.at_position(current_task.target_pos):
                if current_task.type == TaskType.PICKUP:
                    self.status = RobotStatus.PICKING
                    self.carrying_item = current_task.item
                elif current_task.type == TaskType.DELIVER:
                    self.status = RobotStatus.DELIVERING
                    self.carrying_item = None
                elif current_task.type == TaskType.WAIT:
                    self.status = RobotStatus.WAITING
                    if time.time() - current_task.start_time >= 2:  # Wait for 2 seconds
                        current_task.completed = True
                
                if current_task.type not in [TaskType.WAIT]:
                    current_task.completed = True
                
                if current_task.completed:
                    self.task_history.append(current_task)
                    self.tasks.pop(0)
                    self.path = []
                    if not self.tasks:
                        self.status = RobotStatus.IDLE
        
        # Movement
        if self.path and self.battery > 0 and self.status not in [RobotStatus.ERROR, RobotStatus.PICKING, RobotStatus.DELIVERING, RobotStatus.WAITING]:
            target = self.path[0]
            dx = target[0] - self.position[0]
            dy = target[1] - self.position[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < self.speed:
                self.position = target
                self.path.pop(0)
                if not self.path:
                    if self.tasks and self.tasks[0].type == TaskType.CHARGE:
                        self.status = RobotStatus.CHARGING
                    else:
                        self.status = RobotStatus.IDLE
            else:
                self.status = RobotStatus.MOVING
                self.position = (
                    self.position[0] + (dx/dist) * self.speed,
                    self.position[1] + (dy/dist) * self.speed
                )
    
    def at_position(self, pos: Tuple[int, int]) -> bool:
        dx = pos[0] - self.position[0]
        dy = pos[1] - self.position[1]
        return math.sqrt(dx*dx + dy*dy) < 0.1

class Warehouse:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.items = {}  # Dictionary to store items and their positions
        self.setup_layout()
    
    def setup_layout(self):
        # Add storage locations
        for x in range(1, self.width-1, 2):
            self.grid[:, x] = 1
        
        # Add charging stations in corners
        charging_positions = [(0,0), (self.width-1,0), (0,self.height-1), (self.width-1,self.height-1)]
        for x, y in charging_positions:
            self.grid[y, x] = 2
        
        # Add some random items
        storage_locations = [(x, y) for x in range(self.width) for y in range(self.height) 
                           if self.grid[y, x] == 1]
        for i in range(min(10, len(storage_locations))):
            pos = random.choice(storage_locations)
            storage_locations.remove(pos)
            self.items[pos] = f"Item-{i+1}"
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = int(pos[0]), int(pos[1])
        return (0 <= x < self.width and 0 <= y < self.height and 
                self.grid[y, x] != 1)
    
    def is_charging_station(self, pos: Tuple[int, int]) -> bool:
        x, y = int(pos[0]), int(pos[1])
        return (0 <= x < self.width and 0 <= y < self.height and 
                self.grid[y, x] == 2)
    
    def get_charging_stations(self) -> List[Tuple[int, int]]:
        return [(x, y) for y in range(self.height) for x in range(self.width) 
                if self.grid[y, x] == 2]
    
    def get_storage_locations(self) -> List[Tuple[int, int]]:
        return [(x, y) for y in range(self.height) for x in range(self.width) 
                if self.grid[y, x] == 1]
    
    def get_item_at(self, pos: Tuple[int, int]) -> Optional[str]:
        return self.items.get((int(pos[0]), int(pos[1])))
    
    def add_item(self, pos: Tuple[int, int], item: str):
        self.items[pos] = item
    
    def remove_item(self, pos: Tuple[int, int]) -> Optional[str]:
        return self.items.pop((int(pos[0]), int(pos[1])), None)

# Flask app setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize warehouse and robots
warehouse = Warehouse(12, 8)  # Slightly wider warehouse
robots = [
    Robot(0, (0, 0), speed=0.2),
    Robot(1, (11, 0), speed=0.2),
    Robot(2, (0, 7), speed=0.2),
    Robot(3, (11, 7), speed=0.2)
]

# Simulation state
simulation_state = {
    'paused': False,
    'speed': 1.0,
    'auto_mode': False,
    'selected_robot': None,
    'task_queue': []
}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit_state()

@socketio.on('click')
def handle_click(data):
    logger.info(f'Click event received: {data}')
    if simulation_state['paused']:
        return
        
    x, y = data['x'], data['y']
    pos = (x, y)
    
    # Check if clicked on a robot
    clicked_robot = None
    for robot in robots:
        if (int(robot.position[0]), int(robot.position[1])) == pos:
            clicked_robot = robot
            break
    
    if clicked_robot:
        simulation_state['selected_robot'] = clicked_robot.id
        logger.info(f'Selected robot {clicked_robot.id}')
    elif simulation_state['selected_robot'] is not None:
        # Get selected robot
        robot = next((r for r in robots if r.id == simulation_state['selected_robot']), None)
        if robot and robot.status != RobotStatus.ERROR:
            # Check what was clicked
            if warehouse.is_valid_position(pos):
                logger.info(f'Moving robot {robot.id} to position {pos}')
                # If clicked on storage location with item
                item = warehouse.get_item_at(pos)
                if item and not robot.carrying_item:
                    # Pickup task
                    robot.tasks = [
                        Task(TaskType.MOVE, pos),
                        Task(TaskType.PICKUP, pos, item),
                        Task(TaskType.WAIT, pos)
                    ]
                    logger.info(f'Robot {robot.id} picking up {item}')
                elif robot.carrying_item and warehouse.grid[y, x] == 1:
                    # Delivery task
                    robot.tasks = [
                        Task(TaskType.MOVE, pos),
                        Task(TaskType.DELIVER, pos, robot.carrying_item),
                        Task(TaskType.WAIT, pos)
                    ]
                    warehouse.add_item(pos, robot.carrying_item)
                    logger.info(f'Robot {robot.id} delivering {robot.carrying_item}')
                else:
                    # Simple move task
                    robot.tasks = [Task(TaskType.MOVE, pos)]
                    logger.info(f'Robot {robot.id} moving to {pos}')
                
                # Calculate path
                path = PathFinder(warehouse).find_path(
                    (int(robot.position[0]), int(robot.position[1])), 
                    pos
                )
                if path:
                    robot.path = path
                    logger.info(f'Path found: {path}')
                else:
                    logger.warning('No path found')
        
        simulation_state['selected_robot'] = None
    
    emit_state()

@socketio.on('pause')
def handle_pause():
    logger.info('Pause event received')
    simulation_state['paused'] = not simulation_state['paused']
    emit_state()

@socketio.on('speed')
def handle_speed(data):
    logger.info(f'Speed event received: {data}')
    simulation_state['speed'] = float(data['speed'])
    for robot in robots:
        robot.speed = 0.2 * simulation_state['speed']
    emit_state()

@socketio.on('auto')
def handle_auto():
    logger.info('Auto mode event received')
    simulation_state['auto_mode'] = not simulation_state['auto_mode']
    emit_state()

@socketio.on('reset')
def handle_reset():
    logger.info('Reset event received')
    global warehouse
    warehouse = Warehouse(12, 8)
    
    for robot in robots:
        robot.status = RobotStatus.IDLE
        robot.battery = 100.0
        robot.tasks = []
        robot.path = []
        robot.error_count = 0
        robot.carrying_item = None
        robot.total_distance = 0
        robot.task_history = []
        
        # Return to starting position
        if robot.id == 0:
            robot.position = (0, 0)
        elif robot.id == 1:
            robot.position = (11, 0)
        elif robot.id == 2:
            robot.position = (0, 7)
        else:
            robot.position = (11, 7)
    
    simulation_state['paused'] = False
    simulation_state['selected_robot'] = None
    simulation_state['auto_mode'] = False
    simulation_state['task_queue'] = []
    
    emit_state()

def emit_state():
    state = {
        'warehouse': {
            'width': warehouse.width,
            'height': warehouse.height,
            'grid': warehouse.grid.tolist(),
            'items': [(pos, item) for pos, item in warehouse.items.items()]
        },
        'robots': [{
            'id': robot.id,
            'position': [float(robot.position[0]), float(robot.position[1])],
            'battery': robot.battery,
            'status': robot.status.value,
            'path': robot.path,
            'carrying_item': robot.carrying_item,
            'error_count': robot.error_count,
            'total_distance': round(robot.total_distance, 2),
            'tasks': [{'type': t.type.value, 'target': t.target_pos, 'item': t.item} 
                     for t in robot.tasks]
        } for robot in robots],
        'simulation': simulation_state
    }
    socketio.emit('state', state)
    logger.debug('State emitted')

def generate_random_task():
    # Find available items and robots
    items = [(pos, item) for pos, item in warehouse.items.items()]
    available_robots = [r for r in robots if r.status == RobotStatus.IDLE and not r.tasks]
    empty_storage = [(x, y) for x, y in warehouse.get_storage_locations() 
                    if (x, y) not in warehouse.items]
    
    if items and available_robots and empty_storage:
        robot = random.choice(available_robots)
        source_pos, item = random.choice(items)
        target_pos = random.choice(empty_storage)
        
        # Create pickup and delivery tasks
        robot.tasks = [
            Task(TaskType.MOVE, source_pos),
            Task(TaskType.PICKUP, source_pos, item),
            Task(TaskType.WAIT, source_pos),
            Task(TaskType.MOVE, target_pos),
            Task(TaskType.DELIVER, target_pos, item),
            Task(TaskType.WAIT, target_pos)
        ]
        
        # Calculate initial path
        path = PathFinder(warehouse).find_path(
            (int(robot.position[0]), int(robot.position[1])), 
            source_pos
        )
        if path:
            robot.path = path
            warehouse.remove_item(source_pos)

def update_simulation():
    last_update = time.time()
    
    while True:
        if not simulation_state['paused']:
            current_time = time.time()
            dt = current_time - last_update
            last_update = current_time
            
            # Update each robot
            for robot in robots:
                if robot.status != RobotStatus.ERROR:
                    # Update battery
                    battery_drain = 0.1 * dt * simulation_state['speed']
                    if robot.carrying_item:
                        battery_drain *= 1.5
                    robot.battery = max(0, robot.battery - battery_drain)
                    
                    # Check battery status
                    if robot.battery <= 20 and robot.status != RobotStatus.CHARGING:
                        robot.status = RobotStatus.LOW_BATTERY
                        # Find nearest charging station
                        charging_stations = [(x, y) for y in range(warehouse.height) 
                                          for x in range(warehouse.width) 
                                          if warehouse.grid[y, x] == 2]
                        if charging_stations:
                            nearest = min(charging_stations, 
                                       key=lambda pos: ((pos[0]-robot.position[0])**2 + 
                                                      (pos[1]-robot.position[1])**2)**0.5)
                            robot.tasks = [Task(TaskType.MOVE, nearest)]
                            robot.path = PathFinder(warehouse).find_path(
                                (int(robot.position[0]), int(robot.position[1])), 
                                nearest
                            )
                    elif robot.battery <= 0:
                        robot.status = RobotStatus.ERROR
                        robot.error_count += 1
                        continue
                    
                    # Handle charging
                    if warehouse.grid[int(robot.position[1]), int(robot.position[0])] == 2:
                        robot.status = RobotStatus.CHARGING
                        robot.battery = min(100, robot.battery + 20 * dt * simulation_state['speed'])
                        if robot.battery >= 90:  # Stop charging at 90%
                            robot.status = RobotStatus.IDLE
                    
                    # Move robot
                    if robot.path:
                        target = robot.path[0]
                        dx = target[0] - robot.position[0]
                        dy = target[1] - robot.position[1]
                        dist = (dx*dx + dy*dy)**0.5
                        
                        if dist < 0.1:  # Reached waypoint
                            robot.position = target
                            robot.path.pop(0)
                        else:
                            speed = robot.speed * simulation_state['speed']
                            move_dist = min(speed * dt, dist)
                            robot.position = (
                                robot.position[0] + dx/dist * move_dist,
                                robot.position[1] + dy/dist * move_dist
                            )
                            robot.total_distance += move_dist
                    
                    # Handle tasks
                    if not robot.path and robot.tasks:
                        current_task = robot.tasks[0]
                        if current_task.type == TaskType.PICKUP:
                            item = warehouse.get_item_at(current_task.target_pos)
                            if item:
                                robot.carrying_item = item
                                warehouse.remove_item(current_task.target_pos)
                            robot.tasks.pop(0)
                        elif current_task.type == TaskType.DELIVER:
                            if robot.carrying_item:
                                warehouse.add_item(current_task.target_pos, robot.carrying_item)
                                robot.carrying_item = None
                            robot.tasks.pop(0)
                        elif current_task.type == TaskType.MOVE:
                            robot.tasks.pop(0)
                        elif current_task.type == TaskType.WAIT:
                            if not robot.carrying_item:
                                robot.tasks.pop(0)
                
            # Generate random tasks in auto mode
            if simulation_state['auto_mode'] and random.random() < 0.02:
                generate_random_task()
            
            emit_state()
        
        time.sleep(0.05)  # Limit update rate

if __name__ == '__main__':
    # Start simulation thread
    update_thread = threading.Thread(target=update_simulation)
    update_thread.daemon = True
    update_thread.start()
    
    # Run Flask app
    socketio.run(app, host='127.0.0.1', port=8080, debug=False)
