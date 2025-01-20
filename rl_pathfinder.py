import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict
import math

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class RLPathfinder:
    def __init__(self, grid_size: Tuple[int, int], learning_rate: float = 0.001):
        self.grid_size = grid_size
        self.state_size = 8  # Robot position (2), Goal position (2), Nearest obstacles (4)
        self.action_size = 4  # Up, Down, Left, Right
        
        # Neural Networks
        self.policy_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.steps = 0
        
    def get_state(self, robot_pos: Tuple[int, int], goal_pos: Tuple[int, int],
                  obstacles: List[Tuple[int, int]]) -> torch.Tensor:
        """Convert the environment state to a tensor"""
        # Get nearest obstacles
        distances = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Four directions
            nearest = float('inf')
            for ox, oy in obstacles:
                if (dx != 0 and ox == robot_pos[0]) or (dy != 0 and oy == robot_pos[1]):
                    dist = abs(ox - robot_pos[0]) + abs(oy - robot_pos[1])
                    nearest = min(nearest, dist)
            distances.append(min(nearest, 10.0) / 10.0)  # Normalize distance
            
        state = [
            robot_pos[0] / self.grid_size[0],
            robot_pos[1] / self.grid_size[1],
            goal_pos[0] / self.grid_size[0],
            goal_pos[1] / self.grid_size[1]
        ] + distances
        
        return torch.FloatTensor(state).unsqueeze(0)
        
    def select_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()
        return random.randrange(self.action_size)
        
    def store_experience(self, experience: Experience):
        """Store experience in replay memory"""
        self.memory.append(experience)
        
    def optimize_model(self):
        """Train the model using experiences from memory"""
        if len(self.memory) < self.batch_size:
            return
            
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.FloatTensor(batch.done)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def get_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                 obstacles: List[Tuple[int, int]], max_steps: int = 100) -> List[Tuple[int, int]]:
        """Find path using trained model"""
        path = [start]
        current_pos = start
        
        for _ in range(max_steps):
            if current_pos == goal:
                break
                
            state = self.get_state(current_pos, goal, obstacles)
            action = self.select_action(state)
            
            # Convert action to movement
            dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0)][action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check if move is valid
            if (0 <= next_pos[0] < self.grid_size[0] and
                0 <= next_pos[1] < self.grid_size[1] and
                next_pos not in obstacles):
                current_pos = next_pos
                path.append(current_pos)
                
        return path
        
    def train_episode(self, start: Tuple[int, int], goal: Tuple[int, int],
                     obstacles: List[Tuple[int, int]], max_steps: int = 100):
        """Train for one episode"""
        current_pos = start
        total_reward = 0
        
        for step in range(max_steps):
            state = self.get_state(current_pos, goal, obstacles)
            action = self.select_action(state)
            
            # Execute action
            dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0)][action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Calculate reward
            if next_pos == goal:
                reward = 100
                done = True
            elif next_pos in obstacles or not (0 <= next_pos[0] < self.grid_size[0] and
                                            0 <= next_pos[1] < self.grid_size[1]):
                reward = -50
                done = True
                next_pos = current_pos  # Stay in place if move is invalid
            else:
                reward = -1 + (-0.1 * math.sqrt((next_pos[0] - goal[0])**2 +
                                              (next_pos[1] - goal[1])**2))
                done = False
                
            next_state = self.get_state(next_pos, goal, obstacles)
            
            # Store experience
            self.store_experience(Experience(state, action, reward, next_state, done))
            
            # Train model
            self.optimize_model()
            
            total_reward += reward
            current_pos = next_pos
            
            if done:
                break
                
        return total_reward, step + 1
        
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
