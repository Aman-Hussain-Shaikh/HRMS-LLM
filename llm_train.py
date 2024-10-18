import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import logging
import json
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

def load_real_data(file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    data = {}
    for key, path in file_paths.items():
        try:
            data[key] = pd.read_csv(path)
        except Exception as e:
            logging.error(f"Error loading {key} data: {str(e)}")
    return data


# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)


def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1)
    sentiment = "positive" if prediction == 1 else "negative"
    confidence = probs[0][prediction].item()
    return sentiment, confidence

def load_real_data(file_paths):
    data = {}
    for key, path in file_paths.items():
        data[key] = pd.read_csv(path)
    return data

def preprocess_data(data):
    # Convert timestamps to datetime
    for key in data:
        if 'timestamp' in data[key].columns:
            data[key]['timestamp'] = pd.to_datetime(data[key]['timestamp'])
    
    # Sort data by timestamp
    for key in data:
        data[key] = data[key].sort_values('timestamp')
    
    return data

def analyze_activity(data):
    analysis = {}
    
    # Cursor activity analysis
    if 'cursor' in data:
        cursor_df = data['cursor']
        total_distance = np.sum(np.sqrt(np.diff(cursor_df['x'])**2 + np.diff(cursor_df['y'])**2))
        time_diff = (cursor_df['timestamp'].max() - cursor_df['timestamp'].min()).total_seconds()
        avg_speed = total_distance / time_diff if time_diff > 0 else 0
        click_rate = cursor_df['click'].mean()
        
        analysis['cursor_avg_speed'] = avg_speed
        analysis['cursor_click_rate'] = click_rate
    
    # Keystroke activity analysis
    if 'keystroke' in data:
        keystroke_df = data['keystroke']
        time_diff = (keystroke_df['timestamp'].max() - keystroke_df['timestamp'].min()).total_seconds()
        avg_typing_speed = len(keystroke_df) / time_diff if time_diff > 0 else 0
        avg_key_duration = keystroke_df['duration'].mean()
        
        analysis['typing_speed'] = avg_typing_speed
        analysis['avg_key_duration'] = avg_key_duration
    
    # Web and app usage analysis
    if 'web_app' in data:
        web_app_df = data['web_app']
        total_usage_time = web_app_df['duration'].sum()
        most_used_app = web_app_df.groupby('app')['duration'].sum().idxmax()
        most_visited_website = web_app_df[web_app_df['website'].notna()].groupby('website')['duration'].sum().idxmax()
        
        analysis['total_usage_time'] = total_usage_time
        analysis['most_used_app'] = most_used_app
        analysis['most_visited_website'] = most_visited_website
    
    # Printing history analysis
    if 'printing' in data:
        printing_df = data['printing']
        total_pages_printed = printing_df['pages'].sum()
        color_print_ratio = printing_df['color'].mean()
        
        analysis['total_pages_printed'] = total_pages_printed
        analysis['color_print_ratio'] = color_print_ratio
    
    # Time tracking analysis
    if 'time_tracking' in data:
        time_tracking_df = data['time_tracking']
        total_tracked_time = time_tracking_df['duration'].sum()
        most_time_spent = time_tracking_df.groupby('activity')['duration'].sum().idxmax()
        
        analysis['total_tracked_time'] = total_tracked_time
        analysis['most_time_spent'] = most_time_spent
    
    return analysis

def generate_report(analysis_results):
    report = "Activity Analysis Report:\n\n"
    
    if 'cursor_avg_speed' in analysis_results:
        report += f"1. Cursor Activity:\n"
        report += f"   - Average cursor speed: {analysis_results['cursor_avg_speed']:.2f} pixels/second\n"
        report += f"   - Click rate: {analysis_results['cursor_click_rate']:.2%}\n\n"
    
    if 'typing_speed' in analysis_results:
        report += f"2. Keystroke Activity:\n"
        report += f"   - Average typing speed: {analysis_results['typing_speed']:.2f} keys/second\n"
        report += f"   - Average key press duration: {analysis_results['avg_key_duration']*1000:.2f} ms\n\n"
    
    if 'total_usage_time' in analysis_results:
        total_usage_time = int(analysis_results['total_usage_time'])  # Convert numpy.int64 to int
        report += f"3. Web and App Usage:\n"
        report += f"   - Total usage time: {timedelta(seconds=total_usage_time)}\n"
        report += f"   - Most used app: {analysis_results['most_used_app']}\n"
        report += f"   - Most visited website: {analysis_results['most_visited_website']}\n\n"
    
    if 'total_pages_printed' in analysis_results:
        report += f"4. Printing History:\n"
        report += f"   - Total pages printed: {analysis_results['total_pages_printed']}\n"
        report += f"   - Color print ratio: {analysis_results['color_print_ratio']:.2%}\n\n"
    
    if 'total_tracked_time' in analysis_results:
        total_tracked_time = int(analysis_results['total_tracked_time'])  # Convert numpy.int64 to int
        report += f"5. Time Tracking:\n"
        report += f"   - Total tracked time: {timedelta(seconds=total_tracked_time)}\n"
        report += f"   - Activity with most time spent: {analysis_results['most_time_spent']}\n\n"
    
    report += "Analysis:\n"
    # Add your analysis logic here based on the results
    
    return report

from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_sentiment_model(train_data, val_data, model, tokenizer):
    train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_data['text'].tolist(), truncation=True, padding=True)

    train_dataset = SentimentDataset(train_encodings, train_data['label'].tolist())
    val_dataset = SentimentDataset(val_encodings, val_data['label'].tolist())

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return model, trainer, val_dataset 

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def visualize_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Main execution
if __name__ == "__main__":

    
    # Load real data
    file_paths = {
        'cursor': './sample_data/cursor_data.csv',
        'keystroke': './sample_data/keystroke_data.csv',
        'web_app': './sample_data/web_app_data.csv',
        'printing': './sample_data/printing_data.csv', 
        'time_tracking': './sample_data/time_tracking_data.csv'
    }
    # raw_data = load_real_data(file_paths)
    raw_data = load_real_data(config['file_paths'])
     # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    # Preprocess data
    processed_data = preprocess_data(raw_data)
    
    # Analyze activity
    analysis_results = analyze_activity(processed_data)
    
    # Generate report
    report = generate_report(analysis_results)
    print(report)
    
    # Example sentiment analysis
    text = "I'm really enjoying the new project management software!"
    sentiment, confidence = analyze_sentiment(text)
    print(f"\nSentiment Analysis:")
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")
    
    # Train sentiment model on HRMS data
    hrms_sentiment_data = pd.read_csv('./sample_data/hrms_sentiment_data.csv')
    train_data, val_data = train_test_split(hrms_sentiment_data, test_size=0.2, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(hrms_sentiment_data)):
        logging.info(f"Training fold {fold + 1}")
        train_data = hrms_sentiment_data.iloc[train_idx]
        val_data = hrms_sentiment_data.iloc[val_idx]
        
        trained_model, trainer, val_dataset = train_sentiment_model(train_data, val_data, model, tokenizer)
        
        # Evaluate the model
        eval_results = trainer.evaluate()
        cv_scores.append(eval_results['eval_accuracy'])
        
        logging.info(f"Fold {fold + 1} accuracy: {eval_results['eval_accuracy']:.4f}")

    logging.info(f"Cross-validation mean accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")


    trained_model = train_sentiment_model(train_data, val_data, model, tokenizer)
    
    # Save the trained model
       # Save the final trained model
    trained_model.save_pretrained(config['./hrms_sentiment_model'])
    tokenizer.save_pretrained(config['./hrms_sentiment_model'])

    # Visualize confusion matrix
 
    y_true = val_data['label']
    y_pred = trainer.predict(val_dataset).predictions.argmax(-1)
    visualize_confusion_matrix(y_true, y_pred, labels=['negative', 'positive'])
    
   # Test the trained model
    test_text = "The recent team-building activity was fantastic and improved our collaboration."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = trained_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1)
    sentiment = "positive" if prediction == 1 else "negative"
    confidence = probs[0][prediction].item()
    
    print(f"\nTrained Model Sentiment Analysis:")
    print(f"Text: {test_text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")