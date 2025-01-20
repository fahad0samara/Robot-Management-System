import pygame
import asyncio
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from rich.console import Console as Rich
from visualization import WarehouseVisualization
from warehouse import Warehouse, StorageLocation
from robot import Robot, RobotSpecifications, RobotStatus
from task import Task, TaskType
import matplotlib.pyplot as plt
from analytics import WarehouseAnalytics
from predictive_maintenance import MaintenancePredictor
from rl_pathfinder import RLPathfinder

class WarehouseSimulation:
    def __init__(self, num_robots=5, width=20, height=20):
        """Initialize the simulation"""
        pygame.init()  # Initialize pygame
        
        # Initialize warehouse and dimensions
        self.dimensions = (width, height)
        self.warehouse = Warehouse(dimensions=self.dimensions)
        self.warehouse.initialize_layout()
        
        # Initialize robots
        self.robots = []
        specialties = ["pickup", "delivery", "transfer"]
        for i in range(num_robots):
            position = (random.randint(0, width-1), random.randint(0, height-1))
            robot = Robot(position)
            robot.id = i
            robot.name = f"Robot {i}"
            robot.specialize(random.choice(specialties))
            self.robots.append(robot)
            self.warehouse.add_robot(robot)
        
        # Initialize visualization and timing
        self.visualization = WarehouseVisualization(warehouse_size=self.dimensions)
        self.start_time = time.time()
        self.last_metrics_time = 0
        self.analytics = WarehouseAnalytics()
        self.maintenance_predictor = MaintenancePredictor()
        self.pathfinder = RLPathfinder(grid_size=self.dimensions)
        self._initialize_storage_items()
        self.simulation_time = 0.0
        self.time_step = 0.5  
        self.console = Rich()
        self.display = True

    def _initialize_storage_items(self):
        """Add initial products to the warehouse"""
        items = [
            ("Electronics", 50.0),
            ("Books", 30.0),
            ("Food", 100.0),
            ("Clothing", 45.0),
            ("Tools", 75.0)
        ]
        
        locations = list(self.warehouse.storage_locations.values())
        for item_type, weight in items:
            if locations:
                location = locations.pop()
                self.warehouse.update_storage_location(
                    location.id,
                    weight,
                    item_type
                )

    def generate_task(self):
        """Generate a new task with appropriate probability"""
        # Don't generate too many waiting tasks
        idle_robots = [r for r in self.robots if r.status == "idle"]
        active_tasks = len([r for r in self.robots if r.current_task])
        max_waiting = max(len(idle_robots) * 3, 2)  # Allow even more waiting tasks
        
        if len(self.warehouse.waiting_tasks) >= max_waiting:
            return
        
        # Higher probability when robots are idle
        base_prob = 0.6  # Increased base probability
        idle_factor = len(idle_robots) / len(self.robots)
        task_factor = 1 - (active_tasks / len(self.robots))
        prob = base_prob * (idle_factor + 0.4) * (task_factor + 0.4)  # Higher minimum probability
        
        if random.random() < prob:
            # Weight task types towards idle robot specialties
            available_types = ["pickup", "delivery", "transfer"]
            weights = [1, 1, 1]
            
            for robot in idle_robots:
                if robot.specialty in available_types:
                    idx = available_types.index(robot.specialty)
                    weights[idx] += 4  # Increased specialty weight
            
            task_type = random.choices(available_types, weights=weights)[0]
            
            # Generate distinct start and end positions
            while True:
                start = (random.randint(0, self.warehouse.width-1), 
                        random.randint(0, self.warehouse.height-1))
                end = (random.randint(0, self.warehouse.width-1), 
                      random.randint(0, self.warehouse.height-1))
                if start != end:
                    break
            
            # Priority based on waiting tasks and idle robots
            priority = (len(self.warehouse.waiting_tasks) + len(idle_robots)) / 2.0  # Higher priority scaling
            
            # Create and add task
            self.warehouse.add_task(Task(task_type, start, end, priority))

    def assign_tasks(self):
        """Assign available tasks to idle robots"""
        try:
            # Get available tasks and idle robots
            available_tasks = self.warehouse.get_available_tasks()
            idle_robots = [r for r in self.robots if r.status == "idle"]
            
            if not available_tasks or not idle_robots:
                return
                
            # Track assigned task coordinates to prevent duplicates
            assigned_tasks = set()
            
            # Sort tasks by age and priority
            available_tasks.sort(key=lambda t: (time.time() - t.creation_time + t.priority * 15), reverse=True)
            
            # Sort robots by tasks completed (ascending) and battery level (descending)
            idle_robots.sort(key=lambda r: (r.tasks_completed, -r.battery_level))
            
            # Try to assign each task to the best robot
            for task in available_tasks[:]:  # Use slice to allow removal during iteration
                if task in assigned_tasks:
                    continue
                    
                best_robot = None
                best_score = float('-inf')
                
                for robot in idle_robots[:]:  # Use slice to allow removal during iteration
                    # Calculate distance to task start
                    dx = task.start[0] - robot.position[0]
                    dy = task.start[1] - robot.position[1]
                    distance = (dx**2 + dy**2)**0.5
                    
                    # Base score is negative distance (closer is better)
                    score = -distance * 0.1  # Very low distance penalty
                    
                    # Major bonus for robots with no tasks completed
                    if robot.tasks_completed == 0:
                        score += 500  # Very high bonus
                    
                    # Significant bonus for specialized tasks
                    if task.task_type == robot.specialty:
                        score += 300  # Very high specialty bonus
                    
                    # Bonus for high battery level
                    score += robot.battery_level * 3  # Higher battery weight
                    
                    # Major penalty for low battery
                    if robot.battery_level < 30:
                        score -= 1000  # Very high penalty
                    
                    # Minor penalty for completed tasks
                    score -= robot.tasks_completed * 10  # Lower task penalty
                    
                    if score > best_score and robot.battery_level > 20:
                        best_score = score
                        best_robot = robot
                
                # Assign task if we found a suitable robot
                if best_robot:
                    if best_robot.assign_task(task):
                        idle_robots.remove(best_robot)
                        assigned_tasks.add(task)
                        available_tasks.remove(task)
                        
                        # Break if no more idle robots
                        if not idle_robots:
                            break
                            
        except Exception as e:
            print(f"Error in assign_tasks: {e}")
            
    async def run(self, duration=300):
        """Run the simulation for specified duration"""
        try:
            self.start_time = time.time()
            
            while time.time() - self.start_time < duration:
                # Update robots
                for robot in self.robots:
                    await robot.update(self.warehouse)
                
                # Generate new tasks
                self.generate_task()
                
                # Assign tasks to idle robots
                self.assign_tasks()
                
                # Print metrics every second
                current_time = int(time.time() - self.start_time)
                if current_time != self.last_metrics_time:
                    self.print_metrics(current_time)
                    self.last_metrics_time = current_time
                
                # Update visualization
                self.visualization.draw(self.warehouse, self.robots)
                pygame.display.flip()
                
                await asyncio.sleep(0.016)  # ~60 FPS
                
        except asyncio.CancelledError:
            print("\nSimulation cancelled gracefully")
        except Exception as e:
            print(f"Error in simulation: {e}")
            raise  # Re-raise to see full traceback
        finally:
            pygame.quit()

    def print_metrics(self, current_time):
        """Print simulation metrics"""
        print("\n=== Simulation Metrics ===")
        print(f"Time: {current_time}s\n")
        
        print("Robot Status:\n")
        for robot in self.robots:
            print(f"Robot {robot.id}:")
            print(f"  Status: {robot.status}")
            print(f"  Battery: {robot.battery_level:.1f}/100.0")
            print(f"  Distance: {robot.distance_traveled:.2f}")
            print(f"  Tasks Completed: {robot.tasks_completed}")
            if robot.current_task:
                print(f"  Current Task: {robot.current_task.task_type}")
                print(f"    From: {robot.current_task.start}")
                print(f"    To: {robot.current_task.end}")
            print()
            
        print("Warehouse Stats:")
        active_tasks = len([r for r in self.robots if r.current_task])
        waiting_tasks = len(self.warehouse.waiting_tasks)
        completed_tasks = len(self.warehouse.completed_tasks)
        print(f"  Completed Tasks: {completed_tasks}")
        print(f"  Active Tasks: {active_tasks}")
        print(f"  Waiting Tasks: {waiting_tasks}")
        print()
        
        print("Efficiency Metrics:")
        total_tasks = completed_tasks + active_tasks + waiting_tasks
        efficiency = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        avg_battery = sum(r.battery_level for r in self.robots) / len(self.robots)
        total_distance = sum(r.distance_traveled for r in self.robots)
        total_collisions = sum(r.collision_count for r in self.robots)
        collisions_avoided = sum(r.performance_metrics['collisions_avoided'] for r in self.robots)
        
        print(f"  Warehouse Efficiency: {efficiency:.1f}%")
        print(f"  Average Battery: {avg_battery:.1f}")
        print(f"  Total Distance: {total_distance:.1f}")
        print(f"  Total Collisions: {total_collisions}")
        print(f"  Collisions Avoided: {collisions_avoided}")
        print("==============================\n")
        
    def _select_best_task(self, robot, available_tasks):
        """Select best task for robot based on distance, battery, and specialization"""
        if not available_tasks:
            return None
            
        best_task = None
        best_score = float('-inf')
        
        for task in available_tasks:
            # Calculate base distance score
            dx = task.start[0] - robot.position[0]
            dy = task.start[1] - robot.position[1]
            distance_to_start = (dx**2 + dy**2)**0.5
            
            # Calculate task length
            dx = task.end[0] - task.start[0]
            dy = task.end[1] - task.start[1]
            task_length = (dx**2 + dy**2)**0.5
            
            # Calculate total distance
            total_distance = distance_to_start + task_length
            distance_score = 100 - total_distance  # Higher score for shorter distances
            
            # Battery score - prefer tasks when battery is high
            battery_score = robot.battery_level
            
            # Specialization bonus
            specialization_bonus = 1.0
            if robot.specialization == task.task_type.value:
                specialization_bonus = 1.5
                
            # Priority bonus
            priority_bonus = task.priority * 10
            
            # Combine scores
            score = (distance_score * 0.4 + 
                    battery_score * 0.3 + 
                    priority_bonus * 0.2) * specialization_bonus
            
            if score > best_score:
                best_score = score
                best_task = task
                
        return best_task
        
    def _estimate_task_distance(self, start_pos, task):
        """Estimate total distance needed for task"""
        to_start = ((task.start[0] - start_pos[0])**2 + (task.start[1] - start_pos[1])**2)**0.5
        task_length = ((task.end[0] - task.start[0])**2 + (task.end[1] - task.start[1])**2)**0.5
        return to_start + task_length
        
    def _find_nearest_charger(self, position):
        """Find the nearest charging station"""
        nearest_charger = None
        min_distance = float('inf')
        
        for y in range(self.warehouse.dimensions[1]):
            for x in range(self.warehouse.dimensions[0]):
                if self.warehouse.grid[y, x] == 2:  # Charging station
                    distance = ((position[0] - x)**2 + (position[1] - y)**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        nearest_charger = (x, y)
                        
        return nearest_charger

    def _handle_maintenance_need(self, robot: Robot, prediction: Dict):
        """Handle maintenance needs for a robot"""
        if prediction['maintenance_urgency'] > 0.9:
            # Immediate maintenance needed
            robot.status = "maintenance"
            self.console.print(f"[red]URGENT: Robot {robot.id} needs immediate maintenance!")
            self.console.print(f"Recommended actions: {prediction['recommended_actions']}")
        elif prediction['maintenance_urgency'] > 0.7:
            # Schedule maintenance soon
            self.console.print(f"[yellow]WARNING: Robot {robot.id} will need maintenance soon")
            self.console.print(f"Scheduled for: {prediction['next_maintenance_window']}")

    def get_simulation_stats(self):
        """Get current simulation statistics"""
        warehouse_stats = self.analytics.calculate_warehouse_efficiency()
        robot_stats = {
            robot.id: self.analytics.get_robot_efficiency(robot.id)
            for robot in self.robots
        }
        
        return {
            'warehouse_efficiency': warehouse_stats,
            'robot_efficiency': robot_stats,
            'heatmap': self.analytics.get_heatmap()
        }

    def _generate_display(self) -> Rich:
        """Generate rich display for simulation status"""
        # Create status table
        table = Rich.Table(title="Warehouse Simulation Status")
        table.add_column("Robot ID", justify="right", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Battery", justify="right", style="green")
        table.add_column("Position", justify="right", style="yellow")
        
        for robot in self.robots:
            status = robot.generate_status_report()
            table.add_row(
                str(robot.id),
                status['status'].value,
                f"{status['battery_level']:.1f}%",
                f"({status['position'][0]:.1f}, {status['position'][1]:.1f})"
            )

        # Create warehouse utilization info
        utilization = self.warehouse.get_storage_utilization()
        util_text = f"""
        Storage Utilization: {utilization['utilization_percentage']:.1f}%
        Available Locations: {utilization['available_locations']}
        Total Capacity: {utilization['total_capacity']:.1f} kg
        Simulation Time: {self.time:.1f} s
        """

        return Rich.Panel(
            table,
            title="[bold blue]Warehouse Robotics Simulation[/bold blue]",
            subtitle=util_text
        )

    def plot_heatmap(self):
        """Plot heatmap of robot movements"""
        # Initialize heatmap with zeros
        heatmap = np.zeros((self.dimensions[1], self.dimensions[0]))
        
        # Update heatmap based on robot positions
        for robot in self.robots:
            if hasattr(robot, 'position'):
                x, y = int(robot.position[0]), int(robot.position[1])
                if 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1]:
                    heatmap[y, x] += 1
        
        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Robot Density')
        plt.title('Robot Movement Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()

    def save_models(self, base_path: str):
        """Save all trained models"""
        self.maintenance_predictor.save_models(f"{base_path}/maintenance_model.pkl")
        self.pathfinder.save_model(f"{base_path}/pathfinder_model.pt")
        
    def load_models(self, base_path: str):
        """Load all trained models"""
        self.maintenance_predictor.load_models(f"{base_path}/maintenance_model.pkl")
        self.pathfinder.load_model(f"{base_path}/pathfinder_model.pt")

async def main():
    """Main simulation function"""
    sim = WarehouseSimulation(num_robots=5)
    await sim.run(duration=300)  # Run for 5 minutes
    sim.plot_heatmap()

if __name__ == "__main__":
    asyncio.run(main())
