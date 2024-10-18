import pandas as pd
import numpy as np

# Generate sample HRMS sentiment data
num_samples = 1000

positive_samples = [
    "I'm really enjoying the new project management software!",
    "The recent team-building activity was fantastic and improved our collaboration.",
    "The new flexible work hours policy has greatly improved my work-life balance.",
    "I appreciate the constructive feedback from my manager during the performance review.",
    "The company's commitment to professional development is impressive."
]

negative_samples = [
    "The outdated hardware is slowing down my productivity.",
    "I feel overwhelmed with the current workload and lack of resources.",
    "The recent changes in the vacation policy are disappointing.",
    "Communication between departments has been lacking lately.",
    "I'm concerned about the lack of growth opportunities in my current role."
]

texts = []
labels = []

for _ in range(num_samples):
    if np.random.random() < 0.6:  # 60% positive samples
        texts.append(np.random.choice(positive_samples))
        labels.append(1)
    else:
        texts.append(np.random.choice(negative_samples))
        labels.append(0)

hrms_sentiment_data = pd.DataFrame({
    'text': texts,
    'label': labels
})

# Save to CSV
hrms_sentiment_data.to_csv('../sample_data/hrms_sentiment_data.csv', index=False)
print(hrms_sentiment_data.head())