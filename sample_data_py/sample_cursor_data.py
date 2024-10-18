import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample cursor data
num_samples = 1000
start_time = datetime.now()

cursor_data = pd.DataFrame({
    'timestamp': [start_time + timedelta(seconds=i) for i in range(num_samples)],
    'x': np.random.randint(0, 1920, num_samples),
    'y': np.random.randint(0, 1080, num_samples),
    'click': np.random.choice([True, False], num_samples, p=[0.1, 0.9])
})

# Save to CSV
cursor_data.to_csv('../sample_data/cursor_data.csv', index=False)
print(cursor_data.head())