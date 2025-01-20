import pygame
import numpy as np
from warehouse import Warehouse
from robot import Robot, RobotStatus
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (100, 149, 237)  # Cornflower blue
YELLOW = (255, 255, 0)
GRAY = (220, 220, 220)
ORANGE = (255, 165, 0)
DARK_GRAY = (169, 169, 169)

class WarehouseVisualization:
    def __init__(self, warehouse_size=(20, 20), cell_size=30):
        pygame.init()
        self.warehouse_size = warehouse_size
        self.cell_size = cell_size
        
        # Calculate dimensions
        self.panel_width = 300  # Width of the status panel
        self.grid_width = warehouse_size[0] * cell_size
        self.grid_height = warehouse_size[1] * cell_size
        self.screen_width = self.grid_width + self.panel_width + 20  # Add padding
        self.screen_height = max(self.grid_height, 600)  # Minimum height for panels
        
        # Initialize the display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Warehouse Simulation")
        
        # Colors
        self.colors = {
            'background': WHITE,
            'grid': GRAY,
            'robot': BLUE,
            'robot_charging': ORANGE,
            'robot_idle': GREEN,
            'robot_task': BLUE,
            'storage': DARK_GRAY,
            'charging': YELLOW,
            'obstacle': BLACK,
            'text': BLACK,
            'panel': WHITE
        }
        
        # Font initialization
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # Button configuration
        button_x = self.grid_width + 10
        self.buttons = {
            'analytics': pygame.Rect(button_x, 20, 280, 30),
            'heatmap': pygame.Rect(button_x, 60, 280, 30),
            'maintenance': pygame.Rect(button_x, 100, 280, 30)
        }
        
        self.selected_robot = None
        self.status_panel_height = 200
        
    def draw(self, warehouse, robots, analytics=None, show_heatmap=False):
        """Draw the complete warehouse visualization"""
        # Fill background
        self.screen.fill(self.colors['background'])
        
        # Draw heatmap if enabled
        if show_heatmap and analytics:
            analytics.draw_heatmap(self.screen, self.warehouse_size, self.cell_size)
        
        # Draw warehouse grid
        self._draw_grid(warehouse)
        
        # Draw robots
        self._draw_robots(robots)
        
        # Draw status panel
        self._draw_status_panel(robots)
        
        # Draw analytics if enabled
        if analytics:
            self._draw_analytics_panel(analytics)
        
        # Update display
        pygame.display.flip()
        
    def _draw_grid(self, warehouse):
        """Draw the warehouse grid"""
        # Draw grid lines
        for x in range(self.warehouse_size[0] + 1):
            pygame.draw.line(
                self.screen,
                self.colors['grid'],
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_height)
            )
        for y in range(self.warehouse_size[1] + 1):
            pygame.draw.line(
                self.screen,
                self.colors['grid'],
                (0, y * self.cell_size),
                (self.grid_width, y * self.cell_size)
            )
            
        # Draw charging stations
        for station in warehouse.charging_stations:
            x, y = station
            rect = pygame.Rect(
                x * self.cell_size + 2,
                y * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4
            )
            pygame.draw.rect(self.screen, self.colors['charging'], rect)
            
        # Draw storage locations
        for x in range(self.warehouse_size[0]):
            for y in range(self.warehouse_size[1]):
                if warehouse.is_storage_location((x, y)):
                    rect = pygame.Rect(
                        x * self.cell_size + 2,
                        y * self.cell_size + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                    pygame.draw.rect(self.screen, self.colors['storage'], rect)
                    
    def _draw_robots(self, robots):
        """Draw all robots"""
        for robot in robots:
            # Get robot color based on status
            color = {
                'idle': self.colors['robot_idle'],
                'charging': self.colors['robot_charging'],
                'need_charge': self.colors['robot_charging'],
                'moving_to_charge': self.colors['robot_charging']
            }.get(robot.status, self.colors['robot_task'])
            
            # Draw robot body
            x = int(robot.position[0] * self.cell_size)
            y = int(robot.position[1] * self.cell_size)
            robot_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, robot_rect)
            
            # Draw battery indicator
            battery_width = self.cell_size - 4
            battery_height = 4
            battery_x = x + 2
            battery_y = y + self.cell_size - 6
            
            # Background (empty battery)
            pygame.draw.rect(self.screen, RED,
                           (battery_x, battery_y, battery_width, battery_height))
            
            # Foreground (filled battery)
            filled_width = int(battery_width * (robot.battery_level / 100.0))
            if filled_width > 0:
                pygame.draw.rect(self.screen, GREEN,
                               (battery_x, battery_y, filled_width, battery_height))
                
            # Draw robot ID
            text = self.small_font.render(str(robot.id), True, WHITE)
            text_rect = text.get_rect(center=(x + self.cell_size/2, y + self.cell_size/2))
            self.screen.blit(text, text_rect)
            
    def _draw_status_panel(self, robots):
        """Draw the status panel"""
        panel_x = self.grid_width + 10
        panel_y = 10
        line_height = 20
        
        for robot in robots:
            # Robot header
            header = self.small_font.render(f"Robot {robot.id}", True, self.colors['text'])
            self.screen.blit(header, (panel_x, panel_y))
            panel_y += line_height
            
            # Status
            status = self.small_font.render(f"Status: {robot.status}", True, self.colors['text'])
            self.screen.blit(status, (panel_x + 20, panel_y))
            panel_y += line_height
            
            # Battery level
            battery_text = f"Battery: {robot.battery_level:.1f}%"
            battery = self.small_font.render(battery_text, True, self.colors['text'])
            self.screen.blit(battery, (panel_x + 20, panel_y))
            panel_y += line_height
            
            # Current task
            if robot.current_task:
                task_text = f"Task: {robot.current_task.task_type}"
                task = self.small_font.render(task_text, True, self.colors['text'])
                self.screen.blit(task, (panel_x + 20, panel_y))
                panel_y += line_height
                
                # Task details
                start_text = f"From: {robot.current_task.start}"
                end_text = f"To: {robot.current_task.end}"
                start = self.small_font.render(start_text, True, self.colors['text'])
                end = self.small_font.render(end_text, True, self.colors['text'])
                self.screen.blit(start, (panel_x + 40, panel_y))
                panel_y += line_height
                self.screen.blit(end, (panel_x + 40, panel_y))
                panel_y += line_height
            else:
                task = self.small_font.render("Task: None", True, self.colors['text'])
                self.screen.blit(task, (panel_x + 20, panel_y))
                panel_y += line_height
            
            panel_y += 10  # Add space between robots
            
    def _draw_analytics_panel(self, analytics):
        """Draw analytics panel on the right side"""
        panel_rect = pygame.Rect(
            self.grid_width + 10,
            self.status_panel_height + 10,
            self.screen_width - self.grid_width - 20,
            self.screen_height - self.status_panel_height - 20
        )
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        
        # Draw analytics title
        title_font = pygame.font.Font(None, 24)
        title = title_font.render("Analytics", True, self.colors['text'])
        title_rect = title.get_rect(
            centerx=panel_rect.centerx,
            top=panel_rect.top + 10
        )
        self.screen.blit(title, title_rect)
        
        # Draw metrics
        metrics = analytics.metrics
        y_offset = title_rect.bottom + 20
        font = pygame.font.Font(None, 20)
        
        for key, value in metrics.items():
            # Format the key for display
            display_key = key.replace('_', ' ').title()
            
            # Format the value based on type
            if isinstance(value, float):
                display_value = f"{value:.1f}"
            else:
                display_value = str(value)
                
            # Add units where appropriate
            if 'efficiency' in key.lower() or 'utilization' in key.lower():
                display_value += '%'
            elif 'distance' in key.lower():
                display_value += ' units'
                
            # Render metric
            text = font.render(f"{display_key}: {display_value}", True, self.colors['text'])
            text_rect = text.get_rect(
                left=panel_rect.left + 10,
                top=y_offset
            )
            self.screen.blit(text, text_rect)
            y_offset += 25
            
        # Draw warehouse efficiency
        efficiency = analytics.calculate_warehouse_efficiency()
        efficiency_text = font.render(
            f"Warehouse Efficiency: {efficiency:.1f}%",
            True,
            self.colors['text']
        )
        efficiency_rect = efficiency_text.get_rect(
            left=panel_rect.left + 10,
            top=y_offset + 10
        )
        self.screen.blit(efficiency_text, efficiency_rect)
        
    def handle_click(self, pos):
        """Handle mouse clicks"""
        # Adjust position for panel buttons
        panel_pos = (pos[0] - self.grid_width - 10, pos[1])
        
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                return f"toggle_{name}"
        
        return None
        
    def _get_battery_color(self, percentage):
        """Get color based on battery percentage"""
        if percentage > 0.7:
            return (0, 255, 0)  # Green
        elif percentage > 0.3:
            return (255, 165, 0)  # Orange
        else:
            return (255, 0, 0)  # Red

    def draw_analytics(self, warehouse_stats: dict, robot_stats: dict):
        """Draw analytics panel with real-time statistics"""
        self.analytics_surface.fill(WHITE)
        y_offset = 10
        
        # Draw warehouse efficiency stats
        if 'warehouse_efficiency' in warehouse_stats:
            efficiency = warehouse_stats['warehouse_efficiency']
            stats_to_show = [
                f"Tasks/Hour: {efficiency.get('tasks_per_hour', 0):.1f}",
                f"Robot Utilization: {efficiency.get('avg_robot_utilization', 0):.1f}%",
                f"Distance Covered: {efficiency.get('total_distance_covered', 0):.1f}m",
                f"Collisions Avoided: {efficiency.get('collision_avoidance_count', 0)}"
            ]
            
            for stat in stats_to_show:
                text = self.font.render(stat, True, BLACK)
                self.analytics_surface.blit(text, (10, y_offset))
                y_offset += 30
        
        # Draw robot status bars
        y_offset += 20
        for robot_id, stats in robot_stats.items():
            # Battery bar
            pygame.draw.rect(self.analytics_surface, GRAY, 
                           (10, y_offset, 100, 20))
            battery_level = stats.get('battery_level', 0)
            pygame.draw.rect(self.analytics_surface, GREEN if battery_level > 20 else RED,
                           (10, y_offset, battery_level, 20))
            
            # Status text
            status_text = f"Robot {robot_id}: {stats.get('status', 'Unknown')}"
            text = self.font.render(status_text, True, BLACK)
            self.analytics_surface.blit(text, (120, y_offset))
            y_offset += 30
            
        # Draw performance charts if it's time to update
        current_time = pygame.time.get_ticks() / 1000
        if current_time - self.last_chart_update >= self.chart_update_interval:
            self._update_performance_charts(warehouse_stats, robot_stats)
            self.last_chart_update = current_time
            
    def _update_performance_charts(self, warehouse_stats: dict, robot_stats: dict):
        """Update performance charts using matplotlib"""
        # Clear previous charts
        plt.clf()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
        
        # Robot utilization chart
        if robot_stats:
            utilization_data = [stats.get('utilization_rate', 0) for stats in robot_stats.values()]
            robot_ids = list(robot_stats.keys())
            ax1.bar(robot_ids, utilization_data)
            ax1.set_title('Robot Utilization')
            ax1.set_ylabel('Utilization %')
            
        # Task completion trend
        if 'task_completion_trend' in warehouse_stats:
            trend_data = warehouse_stats['task_completion_trend']
            ax2.plot(trend_data)
            ax2.set_title('Task Completion Trend')
            ax2.set_ylabel('Tasks/Hour')
            
        plt.tight_layout()
        
        # Save to a temporary buffer
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Convert to pygame surface
        import pygame.image
        buf.seek(0)
        charts_surface = pygame.image.load(buf)
        charts_surface = pygame.transform.scale(charts_surface, 
                                             (self.analytics_surface.get_width() - 20,
                                              300))
        self.analytics_surface.blit(charts_surface, (10, self.analytics_surface.get_height() - 320))

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.simulation_speed = min(4.0, self.simulation_speed + 0.5)
                elif event.key == pygame.K_DOWN:
                    self.simulation_speed = max(0.5, self.simulation_speed - 0.5)
                elif event.key == pygame.K_p:
                    self.show_paths = not self.show_paths
                elif event.key == pygame.K_q:
                    return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                clicked_cell = self.get_clicked_cell(mouse_pos)
                if clicked_cell is not None:
                    # Select robot at cell
                    for robot in self.robots:
                        if robot.get_position_tuple() == clicked_cell:
                            self.selected_robot = robot
                            break
                else:
                    action = self.handle_click(mouse_pos)
                    if action == 'toggle_analytics':
                        self.show_heatmap = not self.show_heatmap
                    elif action == 'toggle_heatmap':
                        self.show_heatmap = not self.show_heatmap
                    elif action == 'toggle_maintenance':
                        self.show_heatmap = not self.show_heatmap
        return True

    def get_clicked_cell(self, pos):
        """Convert screen coordinates to grid cell coordinates"""
        if (0 <= pos[0] <= self.grid_width and
            0 <= pos[1] <= self.grid_height):
            
            cell_x = pos[0] // self.cell_size
            cell_y = pos[1] // self.cell_size
            
            if (0 <= cell_x < self.warehouse_size[0] and
                0 <= cell_y < self.warehouse_size[1]):
                return (int(cell_x), int(cell_y))
        return None
