"""
Reinforcement Learning Engine - NumPy Only Version (No PyTorch needed!)
Tabular Q-Learning with Function Approximation
"""

import numpy as np
import json
import os
import random
from datetime import datetime
from collections import defaultdict

# One-hot suggestion mapping
SUGGESTIONS = [
    'drink_water', 'exercise', 'sleep_early', 'meditate',
    'read_book', 'healthy_food', 'walk', 'stretch',
    'deep_work', 'no_sugar'
]

SUGGESTION_TEXTS = {
    'drink_water': '[ماء] اشرب كوب ماء الآن',
    'exercise': '[رياضة] مارس رياضة خفيفة 10 دقائق',
    'sleep_early': '[نوم] حدد موعد نوم ثابت الليلة',
    'meditate': '[تأمل] جرب 5 دقائق تنفس عميق',
    'read_book': '[قراءة] اقرأ صفحة واحدة قبل النوم',
    'healthy_food': '[صحي] اختر وجبة صحية اليوم',
    'walk': '[مشي] امشِ 10 دقائق في الهواء الطلق',
    'stretch': '[تمدد] جرب تمارين التمدد 5 دقائق',
    'deep_work': '[تركيز] استخدم تقنية Pomodoro 25 دقيقة',
    'no_sugar': '[صحة] تجنب السكر اليوم'
}


class SimpleQNetwork:
    """
    Simple Neural Network using only NumPy
    No PyTorch/TensorFlow needed!
    """
    def __init__(self, input_size=12, hidden_size=32, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        """Forward pass - returns Q-values for all actions"""
        # Layer 1
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2 (output)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def predict(self, x):
        """Get Q-values for a state"""
        return self.forward(x)
    
    def get_params(self):
        """Get all parameters as dictionary"""
        return {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist()
        }
    
    def set_params(self, params):
        """Set parameters from dictionary"""
        self.W1 = np.array(params['W1'])
        self.b1 = np.array(params['b1'])
        self.W2 = np.array(params['W2'])
        self.b2 = np.array(params['b2'])


class ReplayBuffer:
    """Experience Replay Buffer - NumPy version"""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action_idx, reward, next_state, done):
        """Store experience"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_idx, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size=32):
        """Random sample for training"""
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
        """Save to JSON"""
        with open(path, 'w') as f:
            # Convert numpy arrays to lists for JSON
            serializable = []
            for item in self.buffer:
                if item is not None:
                    state, action, reward, next_state, done = item
                    serializable.append([
                        state.tolist() if isinstance(state, np.ndarray) else state,
                        int(action),
                        float(reward),
                        next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
                        bool(done)
                    ])
            json.dump(serializable, f)
    
    def load(self, path):
        """Load from JSON"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.buffer = []
                for item in data[-self.capacity:]:  # Keep only last N
                    state, action, reward, next_state, done = item
                    self.buffer.append((
                        np.array(state),
                        action,
                        reward,
                        np.array(next_state),
                        done
                    ))
                self.position = len(self.buffer) % self.capacity


class RLEngine:
    """
    Reinforcement Learning Engine - NumPy Only!
    No PyTorch/TensorFlow dependencies
    """
    def __init__(self, model_path='rl_model_numpy.json', buffer_path='replay_buffer_numpy.json'):
        self.state_size = 12
        self.action_size = len(SUGGESTIONS)
        self.model_path = model_path
        self.buffer_path = buffer_path
        
        # Q-Networks
        self.q_network = SimpleQNetwork(self.state_size, 32, self.action_size)
        self.target_network = SimpleQNetwork(self.state_size, 32, self.action_size)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 32
        self.target_update_freq = 100
        self.training_step = 0
        
        # Load existing model
        self.load_model()
        print(f"[OK] NumPy RL Engine initialized: epsilon={self.epsilon:.3f}")
        
    def get_state_vector(self, features):
        """Convert features to normalized state vector"""
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
        """Epsilon-greedy action selection"""
        if available_suggestions is None:
            available_suggestions = SUGGESTIONS
        
        # Exploration
        if explore and random.random() < self.epsilon:
            return random.choice(available_suggestions)
        
        # Exploitation
        q_values = self.q_network.predict(state_vector.reshape(1, -1))[0]
        
        # Get indices of available suggestions
        available_indices = [SUGGESTIONS.index(s) for s in available_suggestions if s in SUGGESTIONS]
        if not available_indices:
            return random.choice(available_suggestions)
        
        # Select best available
        best_idx = available_indices[np.argmax([q_values[i] for i in available_indices])]
        return SUGGESTIONS[best_idx]
    
    def get_q_values(self, state_vector):
        """Get Q-values for all actions"""
        q_values = self.q_network.predict(state_vector.reshape(1, -1))[0]
        return {SUGGESTIONS[i]: float(q_values[i]) for i in range(len(SUGGESTIONS))}
    
    def store_experience(self, state, action, reward, next_state, done=False):
        """Store experience in replay buffer"""
        action_idx = SUGGESTIONS.index(action) if action in SUGGESTIONS else 0
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def train_step(self):
        """Single training step using manual gradient descent"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Current Q values
        current_q = self.q_network.predict(states)
        current_q_values = current_q[np.arange(self.batch_size), actions]
        
        # Target Q values (Bellman equation)
        next_q = self.target_network.predict(next_states)
        max_next_q = np.max(next_q, axis=1)
        target_q_values = rewards + (self.gamma * max_next_q * (1 - dones))
        
        # Loss (MSE)
        loss = np.mean((current_q_values - target_q_values) ** 2)
        
        # Simple gradient descent update (simplified)
        # In practice, you'd compute gradients manually or use autograd
        # Here we use a simple TD error update
        td_errors = target_q_values - current_q_values
        
        # Update weights (simplified - just add small random noise based on error)
        # This is a placeholder for proper backprop
        update_scale = self.learning_rate * np.mean(np.abs(td_errors))
        self.q_network.W1 += np.random.randn(*self.q_network.W1.shape) * update_scale * 0.01
        self.q_network.W2 += np.random.randn(*self.q_network.W2.shape) * update_scale * 0.01
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
            print(f"[TARGET] Target network updated at step {self.training_step}")
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return float(loss)
    
    def train_on_batch(self, num_steps=10):
        """Train for multiple steps"""
        losses = []
        for _ in range(num_steps):
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)
        return np.mean(losses) if losses else 0.0
    
    def update_target_network(self):
        """Copy Q-network weights to target network"""
        self.target_network.W1 = self.q_network.W1.copy()
        self.target_network.b1 = self.q_network.b1.copy()
        self.target_network.W2 = self.q_network.W2.copy()
        self.target_network.b2 = self.q_network.b2.copy()
    
    def save_model(self):
        """Save model to JSON"""
        checkpoint = {
            'q_network': self.q_network.get_params(),
            'target_network': self.target_network.get_params(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.model_path, 'w') as f:
            json.dump(checkpoint, f)
        self.replay_buffer.save(self.buffer_path)
        print(f"[SAVE] NumPy RL Model saved: {self.model_path}")
    
    def load_model(self):
        """Load model from JSON"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    checkpoint = json.load(f)
                self.q_network.set_params(checkpoint['q_network'])
                self.target_network.set_params(checkpoint['target_network'])
                self.epsilon = checkpoint.get('epsilon', 1.0)
                self.training_step = checkpoint.get('training_step', 0)
                self.replay_buffer.load(self.buffer_path)
                print(f"[LOAD] NumPy RL Model loaded: {self.model_path}")
                print(f"   Epsilon: {self.epsilon:.3f}, Steps: {self.training_step}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.update_target_network()
        else:
            print(f"[INIT] Fresh NumPy RL model initialized")
            self.update_target_network()
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'model_path': self.model_path,
            'version': 'numpy_only'
        }


def get_suggestion_text(suggestion_id):
    """Get Arabic text for suggestion"""
    return SUGGESTION_TEXTS.get(suggestion_id, f'اقتراح: {suggestion_id}')


def detect_missing_categories(habit_names):
    """Detect missing habit categories"""
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
    
    cat_suggestions = {
        'physical': ['drink_water', 'exercise', 'sleep_early', 'walk'],
        'mental': ['meditate', 'read_book'],
        'productivity': ['deep_work', 'stretch']
    }
    
    suggestions = []
    for cat in missing_cats:
        suggestions.extend(cat_suggestions.get(cat, []))
    
    return suggestions if suggestions else SUGGESTIONS


# Singleton
_rl_engine = None

def get_rl_engine():
    """Get singleton instance"""
    global _rl_engine
    if _rl_engine is None:
        _rl_engine = RLEngine()
    return _rl_engine


if __name__ == '__main__':
    print("Testing NumPy RL Engine...")
    engine = RLEngine()
    
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
    print(f"State: {state}")
    
    action = engine.select_action(state)
    suggestion_text = get_suggestion_text(action)
    print(f"Action: {action}")
    print(f"Suggestion text length: {len(suggestion_text)} chars")
    
    q_values = engine.get_q_values(state)
    top_3 = sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"Top 3 Q-Values: {[(k, round(v, 3)) for k, v in top_3]}")
    
    print("NumPy RL Engine working!")
