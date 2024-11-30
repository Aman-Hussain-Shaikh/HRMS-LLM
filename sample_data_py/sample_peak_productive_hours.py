import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample time tracking and productivity data
num_samples = 200
start_time = datetime.now() - timedelta(days=30)

# List of activities with productivity weights
activities = {
    "Deep Work": 1.0,       # Highest productivity
    "Coding": 0.9,          # High productivity
    "Design": 0.85,         # High productivity
    "Meetings": 0.5,        # Moderate productivity
    "Project Planning": 0.75,  # Moderate-high productivity
    "Administrative Tasks": 0.4,  # Lower productivity
    "Email Management": 0.3,   # Low productivity
    "Learning": 0.6         # Moderate productivity
}

# Generate time tracking and productivity data
productivity_data = []

for _ in range(num_samples):
    # Random timestamp within the last 30 days
    timestamp = start_time + timedelta(
        days=np.random.randint(0, 30),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 59)
    )
    
    # Choose activity
    activity = np.random.choice(list(activities.keys()))
    
    # Duration with bias towards realistic work durations
    # Shorter tasks: 15-60 mins, Longer tasks: 60-180 mins
    if activity in ["Deep Work", "Coding", "Design", "Project Planning"]:
        duration = np.random.randint(60, 180)  # longer, focused work
    else:
        duration = np.random.randint(15, 60)   # shorter, less intensive tasks
    
    # Calculate productivity based on activity and time of day
    # Create a productivity curve that peaks during typical work hours
    hour_productivity_curve = {
        0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 
        4: 0.2, 5: 0.3, 6: 0.5, 
        7: 0.7, 8: 0.9, 9: 1.0, 
        10: 1.0, 11: 0.95, 12: 0.8, 
        13: 0.75, 14: 0.85, 15: 0.9, 
        16: 0.85, 17: 0.7, 
        18: 0.5, 19: 0.3, 20: 0.2, 
        21: 0.1, 22: 0.1, 23: 0.1
    }
    
    # Calculate base productivity
    base_productivity = activities[activity]
    hour_factor = hour_productivity_curve.get(timestamp.hour, 0.5)
    
    # Overall productivity score
    productivity_score = base_productivity * hour_factor * (duration / 60)
    
    # Add some randomness
    productivity_score *= np.random.uniform(0.8, 1.2)
    
    productivity_data.append({
        'timestamp': timestamp,
        'hour_of_day': timestamp.hour,
        'activity': activity,
        'duration': duration,
        'base_productivity': base_productivity,
        'hour_productivity_factor': hour_factor,
        'productivity_score': round(productivity_score, 2)
    })

# Create DataFrame
productivity_df = pd.DataFrame(productivity_data)

# Analyze peak productive hours
peak_hours = productivity_df.groupby('hour_of_day').agg({
    'productivity_score': ['mean', 'count']
}).reset_index()
peak_hours.columns = ['hour_of_day', 'avg_productivity', 'activity_count']
peak_hours = peak_hours.sort_values('avg_productivity', ascending=False)

# Analyze activity productivity
activity_productivity = productivity_df.groupby('activity').agg({
    'productivity_score': ['mean', 'count', 'sum']
}).reset_index()
activity_productivity.columns = ['activity', 'avg_productivity', 'activity_count', 'total_productivity']
activity_productivity = activity_productivity.sort_values('avg_productivity', ascending=False)

# Save to CSV
productivity_df.to_csv('../sample_data/peak_productivity_data.csv', index=False)
peak_hours.to_csv('../sample_data/peak_hours_analysis.csv', index=False)
activity_productivity.to_csv('../sample_data/activity_productivity.csv', index=False)

print("Peak Productive Hours:")
print(peak_hours)
print("\nActivity Productivity:")
print(activity_productivity)
print("\nSample Productivity Data:")
print(productivity_df.head())