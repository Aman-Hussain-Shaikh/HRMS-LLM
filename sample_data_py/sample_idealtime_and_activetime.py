import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample active user and ideal time insights data
num_samples = 500
start_time = datetime.now() - timedelta(days=30)

# List of employees
employees = [f"EMP{i:03d}" for i in range(1, 21)]

# Device types
devices = ['Laptop', 'Desktop', 'Tablet', 'Mobile']

# Generate active user and ideal time data
active_user_data = []

for _ in range(num_samples):
    emp = np.random.choice(employees)
    timestamp = start_time + timedelta(
        days=np.random.randint(0, 30),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    )
    
    # Active time calculation
    total_session_time = np.random.randint(15, 480)  # 15 mins to 8 hours
    active_time = np.random.randint(int(total_session_time * 0.6), total_session_time)
    
    # Engagement metrics
    devices_used = np.random.choice(devices, size=np.random.randint(1, 3), replace=False)
    application_switches = np.random.randint(1, 20)
    
    # Workflow efficiency
    task_completion_rate = np.random.uniform(0.5, 1.0)
    
    # Idle time calculation
    idle_time = total_session_time - active_time
    
    # Categorize ideal working time
    if 9 <= timestamp.hour < 12 or 13 <= timestamp.hour < 17:
        ideal_time_category = 'Core Working Hours'
    elif 6 <= timestamp.hour < 9 or 17 <= timestamp.hour < 20:
        ideal_time_category = 'Flexible Working Hours'
    else:
        ideal_time_category = 'Non-Working Hours'
    
    active_user_data.append({
        'employee_id': emp,
        'timestamp': timestamp,
        'total_session_time': total_session_time,
        'active_time': active_time,
        'idle_time': idle_time,
        'active_percentage': (active_time / total_session_time) * 100,
        'devices_used': ', '.join(devices_used),
        'application_switches': application_switches,
        'task_completion_rate': task_completion_rate,
        'ideal_time_category': ideal_time_category
    })

# Create DataFrame
active_user_df = pd.DataFrame(active_user_data)

# Analyze insights
insights = active_user_df.groupby('ideal_time_category').agg({
    'active_time': 'mean',
    'task_completion_rate': 'mean',
    'application_switches': 'mean'
}).reset_index()

# Save to CSV
active_user_df.to_csv('../sample_data/active_user_data.csv', index=False)
insights.to_csv('../sample_data/active_user_insights.csv', index=False)

print("Sample Active User Data:")
print(active_user_df.head())
print("\nActive User Insights:")
print(insights)