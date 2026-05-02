import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import random

class HabitModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.feature_columns = [
            'completionRate', 'activeStreaks', 'bestStreak', 
            'totalHabits', 'consistency', 'dropRate'
        ]
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic habit data for training"""
        data = []
        
        for i in range(n_samples):
            # Generate realistic habit patterns
            completion_rate = np.random.beta(2, 2)  # Beta distribution for completion rates
            total_habits = np.random.randint(1, 15)
            
            # Active streaks influenced by completion rate
            active_streaks = max(0, int(np.random.normal(completion_rate * 10, 3)))
            best_streak = max(active_streaks, int(np.random.normal(completion_rate * 15, 5)))
            
            # Consistency and drop rate calculations
            consistency = max(0, min(1, completion_rate + np.random.normal(0, 0.2)))
            drop_rate = max(0, min(1, (1 - consistency) + np.random.normal(0, 0.1)))
            
            # Determine user state based on patterns
            user_state = self._determine_user_state(
                completion_rate, consistency, active_streaks, drop_rate
            )
            
            data.append({
                'completionRate': completion_rate,
                'activeStreaks': active_streaks,
                'bestStreak': best_streak,
                'totalHabits': total_habits,
                'consistency': consistency,
                'dropRate': drop_rate,
                'userState': user_state
            })
        
        return pd.DataFrame(data)
    
    def _determine_user_state(self, completion_rate, consistency, active_streaks, drop_rate):
        """Determine user state based on performance metrics"""
        score = (completion_rate * 0.3 + 
                consistency * 0.3 + 
                (active_streaks / 20) * 0.2 + 
                (1 - drop_rate) * 0.2)
        
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'moderate'
        elif score >= 0.2:
            return 'struggling'
        else:
            return 'critical'
    
    def train(self, custom_data=None):
        """Train the model"""
        if custom_data is not None:
            df = custom_data
        else:
            df = self.generate_synthetic_data()
        
        # Prepare features
        X = df[self.feature_columns]
        y = df['userState']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Save model and scaler
        joblib.dump(self.model, 'habit_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        
        return train_score, test_score
    
    def predict_user_state(self, features):
        """Predict user state for given features"""
        # Load model and scaler if not in memory
        if self.model is None:
            self.model = joblib.load('habit_model.pkl')
            self.scaler = joblib.load('feature_scaler.pkl')
        
        # Prepare features
        feature_array = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.model.predict(feature_scaled)[0]
        probabilities = self.model.predict_proba(feature_scaled)[0]
        
        # Get class labels
        classes = self.model.classes_
        confidence = max(probabilities)
        
        return {
            'state': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(classes, probabilities))
        }
    
    def predict_future_performance(self, current_features, historical_data=None):
        """Predict future performance trend"""
        if historical_data is None:
            # Generate trend based on current state
            current_state = self.predict_user_state(current_features)
            
            # Simulate future trajectory
            trend_factor = np.random.normal(0, 0.1)  # Random trend component
            
            # Calculate future score based on current metrics
            current_score = (
                current_features['completionRate'] * 0.3 +
                current_features['consistency'] * 0.3 +
                (current_features['activeStreaks'] / 20) * 0.2 +
                (1 - current_features['dropRate']) * 0.2
            )
            
            # Predict future score with trend
            future_score = max(0, min(1, current_score + trend_factor))
            
            # Determine trend direction
            if future_score > current_score + 0.05:
                trend = 'improving'
            elif future_score < current_score - 0.05:
                trend = 'declining'
            else:
                trend = 'stable'
            
            return {
                'currentScore': current_score,
                'futureScore': future_score,
                'trend': trend,
                'confidence': abs(future_score - current_score) * 100
            }

if __name__ == "__main__":
    trainer = HabitModelTrainer()
    trainer.train()
    
    # Test prediction
    test_features = {
        'completionRate': 0.75,
        'activeStreaks': 8,
        'bestStreak': 15,
        'totalHabits': 5,
        'consistency': 0.8,
        'dropRate': 0.2
    }
    
    result = trainer.predict_user_state(test_features)
    print(f"Predicted state: {result}")
    
    future = trainer.predict_future_performance(test_features)
    print(f"Future prediction: {future}")
