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
    for key in data:
        # Check if 'timestamp' column exists
        if 'timestamp' in data[key].columns:
            try:
                # Convert to datetime
                data[key]['timestamp'] = pd.to_datetime(data[key]['timestamp'])
                # Sort by timestamp
                data[key] = data[key].sort_values('timestamp')
            except Exception as e:
                print(f"Could not process timestamp for {key}: {e}")
        else:
            print(f"No timestamp column found in {key} dataset")
    
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
    
    if 'attendance' in data:
        attendance_df = data['attendance']
        total_employees = attendance_df['employee_id'].nunique()
        attendance_summary = attendance_df['attendance_status'].value_counts(normalize=True)
        
        analysis['total_employees'] = total_employees
        analysis['attendance_distribution'] = attendance_summary.to_dict()
        analysis['avg_working_hours'] = attendance_df[attendance_df['is_present']]['total_hours'].mean()
    
    # New: Peak Productivity Analysis
    if 'productivity' in data:
        productivity_df = data['productivity']
        peak_hours = productivity_df.groupby('hour_of_day').agg({
            'productivity_score': ['mean', 'count']
        }).reset_index()
        peak_hours.columns = ['hour', 'avg_productivity', 'activity_count']
        
        # Most productive activities
        activity_productivity = productivity_df.groupby('activity').agg({
            'productivity_score': ['mean', 'count']
        }).reset_index()
        activity_productivity.columns = ['activity', 'avg_productivity', 'activity_count']
        
        analysis['peak_hours'] = peak_hours.sort_values('avg_productivity', ascending=False).to_dict('records')
        analysis['activity_productivity'] = activity_productivity.sort_values('avg_productivity', ascending=False).to_dict('records')
    
    # New: Active User Insights Analysis
    if 'active_user' in data:
        active_user_df = data['active_user']
        
        analysis['total_active_sessions'] = len(active_user_df)
        analysis['avg_session_time'] = active_user_df['total_session_time'].mean()
        analysis['avg_active_time'] = active_user_df['active_time'].mean()
        analysis['avg_idle_time'] = active_user_df['idle_time'].mean()
        analysis['active_time_percentage'] = active_user_df['active_percentage'].mean()
        
        # Breakdown by ideal time category
        ideal_time_breakdown = active_user_df['ideal_time_category'].value_counts(normalize=True)
        analysis['ideal_time_breakdown'] = ideal_time_breakdown.to_dict()
        
        # Device usage
        device_usage = active_user_df['devices_used'].str.split(', ', expand=True).stack().value_counts(normalize=True)
        analysis['device_usage'] = device_usage.to_dict()
    
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
    
    if 'total_employees' in analysis_results:
        report += "6. Attendance Management:\n"
        report += f"   - Total Employees Tracked: {analysis_results['total_employees']}\n"
        report += "   - Attendance Distribution:\n"
        for status, percentage in analysis_results['attendance_distribution'].items():
            report += f"     * {status}: {percentage:.2%}\n"
        report += f"   - Average Working Hours: {analysis_results['avg_working_hours']:.2f}\n\n"
    
    if 'peak_hours' in analysis_results:
        report += "7. Productivity Insights:\n"
        report += "   - Peak Productive Hours:\n"
        for hour_data in analysis_results['peak_hours'][:3]:
            report += f"     * Hour {hour_data['hour']}: Avg Productivity {hour_data['avg_productivity']:.2f}\n"
        
        report += "   - Most Productive Activities:\n"
        for activity_data in analysis_results['activity_productivity'][:3]:
            report += f"     * {activity_data['activity']}: Avg Productivity {activity_data['avg_productivity']:.2f}\n\n"
    
    if 'total_active_sessions' in analysis_results:
        report += "8. Active User Insights:\n"
        report += f"   - Total Active Sessions: {analysis_results['total_active_sessions']}\n"
        report += f"   - Average Session Time: {timedelta(seconds=int(analysis_results['avg_session_time']))}\n"
        report += f"   - Average Active Time: {timedelta(seconds=int(analysis_results['avg_active_time']))}\n"
        report += f"   - Average Idle Time: {timedelta(seconds=int(analysis_results['avg_idle_time']))}\n"
        report += f"   - Active Time Percentage: {analysis_results['active_time_percentage']:.2f}%\n\n"
        
        report += "   - Working Time Category Breakdown:\n"
        for category, percentage in analysis_results['ideal_time_breakdown'].items():
            report += f"     * {category}: {percentage:.2%}\n"
        
        report += "   - Device Usage:\n"
        for device, usage in analysis_results['device_usage'].items():
            report += f"     * {device}: {usage:.2%}\n"
    
    return report
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
        'time_tracking': './sample_data/time_tracking_data.csv',
        'attendance': './sample_data/attendance_data.csv',
        'productivity': './sample_data/peak_productivity_data.csv',
        'active_user': './sample_data/active_user_data.csv'
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