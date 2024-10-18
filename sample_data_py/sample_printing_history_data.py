import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample printing history data
num_samples = 20
start_time = datetime.now()

document_types = ["Report", "Letter", "Invoice", "Resume", "Article"]

printing_data = pd.DataFrame({
    'timestamp': [start_time + timedelta(hours=i*2) for i in range(num_samples)],
    'document_type': np.random.choice(document_types, num_samples),
    'pages': np.random.randint(1, 21, num_samples),
    'color': np.random.choice([True, False], num_samples)
})

# Save to CSV
printing_data.to_csv('../sample_data/printing_data.csv', index=False)
print(printing_data.head())