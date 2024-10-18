import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample keystroke data
num_samples = 500
start_time = datetime.now()

keystroke_data = pd.DataFrame({
    'timestamp': [start_time + timedelta(seconds=i*0.5) for i in range(num_samples)],
    'key': np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), num_samples),
    'duration': np.random.uniform(0.05, 0.2, num_samples)
})

# Save to CSV
keystroke_data.to_csv('../sample_data/keystroke_data.csv', index=False)
print(keystroke_data.head())