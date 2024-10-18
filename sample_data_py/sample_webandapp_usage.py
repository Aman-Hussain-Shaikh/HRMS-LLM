import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample web app usage data
num_samples = 100
start_time = datetime.now()

apps = ["Browser", "Word Processor", "Email Client", "Video Player", "Code Editor"]
websites = ["example.com", "google.com", "github.com", "stackoverflow.com", "youtube.com"]

web_app_data = pd.DataFrame({
    'timestamp': [start_time + timedelta(minutes=i*15) for i in range(num_samples)],
    'app': np.random.choice(apps, num_samples),
    'website': np.random.choice(websites + [None], num_samples),
    'duration': np.random.randint(30, 3600, num_samples)
})

# Save to CSV
web_app_data.to_csv('../sample_data/web_app_data.csv', index=False)
print(web_app_data.head())