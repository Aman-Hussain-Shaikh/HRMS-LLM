import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample time tracking data
num_samples = 50
start_time = datetime.now()

activities = ["Working", "Meeting", "Break", "Learning", "Administrative"]

time_tracking_data = pd.DataFrame({
    'timestamp': [start_time + timedelta(hours=i*2) for i in range(num_samples)],
    'activity': np.random.choice(activities, num_samples),
    'duration': np.random.randint(5*60, 120*60, num_samples)
})

# Save to CSV
time_tracking_data.to_csv('../sample_data/time_tracking_data.csv', index=False)
print(time_tracking_data.head())