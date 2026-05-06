"""
Reinforcement Learning Engine for Habit Recommendations
Q-Learning with Neural Network function approximation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import json
import os
from datetime import datetime

# One-hot suggestion mapping
SUGGESTIONS = [
    'drink_water', 'exercise', 'sleep_early', 'meditate',
    'read_book', 'healthy_food', 'walk', 'stretch',
    'deep_work', 'no_sugar'
]

SUGGESTION_TEXTS = {
    'drink_water': '💧 اشرب كوب ماء الآن',
    'exercise': '🏃 مارس رياضة خفيفة 10 دقائق',
    'sleep_early': '😴 حدد موعد نوم ثابت الليلة',
    'meditate': '🧘 جرب 5 دقائق تنفس عميق',
    'read_book': '📖 اقرأ صفحة واحدة قبل النوم',
    'healthy_food': '🥗 اختر وجبة صحية اليوم',
    'walk': '🚶‍♂️ امشِ 10 دقائق في الهواء الطلق',
    'stretch': '🤸 جرب تمارين التمدد 5 دقائق',
    'deep_work': '⏱️ استخدم تقنية Pomodoro 25 دقيقة',
    'no_sugar': '🚫 تجنب السكر اليوم'
}


class QNetwork(nn.Module):
    """Neural Network for Q-Value approximation"""
    def __init__(self, state_size=12, action_size=10, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience Replay Buffer for stable learning"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action_idx, reward, next_state, done):
        """Store experience tuple"""
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size=32):
        """Random sampling for experience replay"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path):
        """Save buffer to disk"""
        with open(path, 'w') as f:
            json.dump(list(self.buffer), f)
    
    def load(self, path):
        """Load buffer from disk"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.buffer = deque(data, maxlen=self.buffer.maxlen)


class RLEngine:
    """
    Reinforcement Learning Engine for Smart Habit Suggestions
    Uses Deep Q-Network (DQN) with Experience Replay
    """
    def __init__(self, model_path='rl_model.pth', buffer_path='replay_buffer.json'):
        self.state_size = 12
        self.action_size = len(SUGGESTIONS)
        self.model_path = model_path
        self.buffer_path = buffer_path
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ RL Engine using device: {self.device}")
        
        # Q-Networks (main and target)
        self.q_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.target_update_freq = 100  # Update target network every N steps
        self.training_step = 0
        
        # Load existing model if available
        self.load_model()
        
    def get_state_vector(self, features):
        """
        Convert feature dictionary to normalized state vector
        
        Features expected:
        - completion_rate: 0-1
        - consistency: 0-1
        - drop_rate: 0-1
        - active_streaks: count (normalized to 0-1)
        - best_streak: count (normalized to 0-1)
        - total_habits: count (normalized to 0-1)
        - hour: 0-23 (normalized)
        - day_of_week: 0-6 (normalized)
        - is_weekend: 0 or 1
        - previous_completion: 0-1
        - trend_7d: -1 to 1
        - trend_30d: -1 to 1
        """
        return np.array([
            float(features.get('completion_rate', 0)),
            float(features.get('consistency', 0)),
            float(features.get('drop_rate', 0)),
            min(float(features.get('active_streaks', 0)) / 20, 1.0),
            min(float(features.get('best_streak', 0)) / 30, 1.0),
            min(float(features.get('total_habits', 0)) / 10, 1.0),
            float(features.get('hour', 12)) / 24,
            float(features.get('day_of_week', 0)) / 7,
            1.0 if features.get('is_weekend', False) else 0.0,
            float(features.get('previous_completion', 0)),
            float(features.get('trend_7d', 0)),
            float(features.get('trend_30d', 0))
        ], dtype=np.float32)
    
    def select_action(self, state_vector, available_suggestions=None, explore=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state_vector: numpy array of state features
            available_suggestions: list of suggestion IDs to choose from
            explore: if False, always use best action (no exploration)
        
        Returns:
            Selected suggestion ID
        """
        if available_suggestions is None:
            available_suggestions = SUGGESTIONS
        
        # Exploration: random choice
        if explore and random.random() < self.epsilon:
            return random.choice(available_suggestions)
        
        # Exploitation: best Q-value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Get indices of available suggestions
            available_indices = [SUGGESTIONS.index(s) for s in available_suggestions if s in SUGGESTIONS]
            
            if not available_indices:
                return random.choice(available_suggestions)
            
            # Select best among available
            best_idx = available_indices[np.argmax([q_values[i] for i in available_indices])]
            return SUGGESTIONS[best_idx]
    
    def get_q_values(self, state_vector):
        """Get Q-values for all actions"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        return {SUGGESTIONS[i]: float(q_values[i]) for i in range(len(SUGGESTIONS))}
    
    def store_experience(self, state, action, reward, next_state, done=False):
        """
        Store experience tuple in replay buffer
        
        Args:
            state: state vector before action
            action: suggestion ID taken
            reward: float reward value
            next_state: state vector after action
            done: whether episode ended (False for our case)
        """
        action_idx = SUGGESTIONS.index(action) if action in SUGGESTIONS else 0
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step using experience replay
        
        Returns:
            loss value or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values (Bellman equation)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
            print(f"🎯 Target network updated at step {self.training_step}")
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train_on_batch(self, num_steps=10):
        """Train for multiple steps"""
        losses = []
        for _ in range(num_steps):
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)
        
        return np.mean(losses) if losses else 0.0
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self):
        """Save model weights and training state"""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, self.model_path)
        self.replay_buffer.save(self.buffer_path)
        print(f"💾 RL Model saved: {self.model_path}")
    
    def load_model(self):
        """Load model weights if available"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint.get('epsilon', 1.0)
                self.training_step = checkpoint.get('training_step', 0)
                self.replay_buffer.load(self.buffer_path)
                print(f"📂 RL Model loaded: {self.model_path}")
                print(f"   Epsilon: {self.epsilon:.3f}, Training steps: {self.training_step}")
                print(f"   Buffer size: {len(self.replay_buffer)}")
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
                print("   Starting with fresh model")
                self.update_target_network()
        else:
            print("🆕 No existing model found. Initializing fresh Q-networks")
            self.update_target_network()
    
    def get_stats(self):
        """Get current training statistics"""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'model_path': self.model_path,
            'device': str(self.device)
        }


def get_suggestion_text(suggestion_id):
    """Get Arabic text for suggestion ID"""
    return SUGGESTION_TEXTS.get(suggestion_id, f'اقتراح: {suggestion_id}')


def detect_missing_categories(habit_names):
    """Detect which habit categories are missing"""
    categories = {
        'physical': ['water', 'exercise', 'sleep', 'walk'],
        'mental': ['thinking', 'reading', 'journaling', 'breathing'],
        'productivity': ['focus_time', 'screen_limit', 'morning_routine', 'planning']
    }
    
    user_cats = set()
    for habit in habit_names:
        for cat, habits in categories.items():
            if any(h in habit.lower() for h in habits):
                user_cats.add(cat)
    
    missing_cats = set(categories.keys()) - user_cats
    
    # Map categories to suggestions
    cat_suggestions = {
        'physical': ['drink_water', 'exercise', 'sleep_early', 'walk'],
        'mental': ['meditate', 'read_book'],
        'productivity': ['deep_work', 'stretch']
    }
    
    suggestions = []
    for cat in missing_cats:
        suggestions.extend(cat_suggestions.get(cat, []))
    
    return suggestions if suggestions else SUGGESTIONS


# Singleton instance
_rl_engine = None

def get_rl_engine():
    """Get or create singleton RL Engine instance"""
    global _rl_engine
    if _rl_engine is None:
        _rl_engine = RLEngine()
    return _rl_engine


if __name__ == '__main__':
    # Test the RL Engine
    print("🧪 Testing RL Engine...")
    engine = RLEngine()
    
    # Test state vector
    features = {
        'completion_rate': 0.75,
        'consistency': 0.6,
        'drop_rate': 0.2,
        'active_streaks': 3,
        'best_streak': 7,
        'total_habits': 5,
        'hour': 9,
        'day_of_week': 2,
        'is_weekend': False,
        'previous_completion': 0.8,
        'trend_7d': 0.1,
        'trend_30d': 0.05
    }
    
    state = engine.get_state_vector(features)
    print(f"State vector: {state}")
    
    # Test action selection
    action = engine.select_action(state)
    print(f"Selected action: {action} -> {get_suggestion_text(action)}")
    
    # Test Q-values
    q_values = engine.get_q_values(state)
    print(f"Top Q-values: {sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    print("✅ RL Engine test complete!")
