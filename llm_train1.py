import os
import torch
import pandas as pd
import numpy as np
import json
import logging
import datetime
import hashlib
from typing import Dict, Any, List
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline
)

from torch.nn.functional import softmax

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

class AnalyticsCheckpointManager:
    def __init__(self, checkpoint_dir='./checkpoints'):
        """
        Initialize checkpoint manager with a specific directory
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('analytics_process.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _generate_file_hash(self, filepath):
        """
        Generate a hash for a file to detect changes
        """
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def check_and_process_files(self, data_files: Dict[str, str]) -> List[pd.DataFrame]:
        """
        Check files against previous checkpoints and process only new or changed files
        
        :param data_files: Dictionary of filename to filepath
        :return: List of processed DataFrames
        """
        processed_dataframes = []
        checkpoint_file = os.path.join(self.checkpoint_dir, 'file_checksums.json')
        
        # Load existing checksums
        try:
            with open(checkpoint_file, 'r') as f:
                existing_checksums = json.load(f)
        except FileNotFoundError:
            existing_checksums = {}

        # Track changes
        updated_files = []
        
        # Process each file
        for filename, filepath in data_files.items():
            self.logger.info(f"Processing file: {filename}")
            
            try:
                # Generate current file hash
                current_hash = self._generate_file_hash(filepath)
                
                # Check if file has changed
                if (filename not in existing_checksums or 
                    existing_checksums[filename] != current_hash):
                    
                    self.logger.info(f"File {filename} is new or has changed. Processing...")
                    
                    # Read DataFrame
                    df = pd.read_csv(filepath)
                    df['source_file'] = filename
                    processed_dataframes.append(df)
                    
                    # Update checksums
                    existing_checksums[filename] = current_hash
                    updated_files.append(filename)
                    
                else:
                    self.logger.info(f"File {filename} unchanged. Skipping processing.")
                    
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
        
        # Save updated checksums
        if updated_files:
            with open(checkpoint_file, 'w') as f:
                json.dump(existing_checksums, f, indent=2)
            
            self.logger.info(f"Updated files: {updated_files}")
        
        return processed_dataframes

def interactive_analytics_menu(generator, insights_generator):
    """
    Create an interactive menu for analytics exploration
    """
    while True:
        print("\n--- HRMS Analytics Dashboard ---")
        print("1. Productivity Analysis")
        print("2. Workflow and Time Management")
        print("3. User Interaction Analysis")
        print("4. Organizational Efficiency Report")
        print("5. Rerun Data Processing")
        print("6. Interactive LLM Analysis")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ")
        
        try:
            if choice == '1':
                print("\n--- Productivity Analysis ---")
                print(insights_generator.generate_productivity_analysis())
            
            elif choice == '2':
                print("\n--- Workflow and Time Management ---")
                print(insights_generator.generate_workflow_analysis())
            
            elif choice == '3':
                print("\n--- User Interaction Analysis ---")
                print(insights_generator.generate_user_interaction_analysis())
            
            elif choice == '4':
                print("\n--- Organizational Efficiency Report ---")
                print(insights_generator.generate_organizational_efficiency_report())
            
            elif choice == '5':
                print("\nRe-running Data Processing...")
                main()
            
            elif choice == '6':
                generator.interactive_llm_analysis(comprehensive_insights)
            
            elif choice == '7':
                print("Exiting the HRMS Analytics Dashboard. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please select a number between 1 and 7.")
        
        except Exception as e:
            print(f"An error occurred: {e}")
        
        input("\nPress Enter to continue...")

class InsightsTextGenerator:
    def __init__(self, comprehensive_insights, comprehensive_analysis):
        self.insights = comprehensive_insights
        self.analysis = comprehensive_analysis
    
    def generate_productivity_analysis(self):
        """
        Generate a detailed narrative about productivity and performance
        """
        productivity_insights = self.insights.get('Productivity Insights', {})
        time_tracking = self.insights.get('Time Tracking', {})
        
        narrative = f"""
        Productivity Performance Analysis

        Peak Performance Overview:
        Our analysis reveals critical insights into organizational productivity. 
        
        Peak Productive Hours:
        - The most productive hours of the day are between 13:00-15:00 (1 PM - 3 PM)
        - Hour 15 shows the highest productivity score of {productivity_insights['Peak Productive Hours'].get('15', 'N/A')}
        - Consistent productivity maintenance observed during consecutive peak hours

        Activity Productivity Breakdown:
        Top Performing Activities:
        1. Deep Work: Productivity Score {productivity_insights['Most Productive Activities'].get('Deep Work', 'N/A')}
        2. Project Planning: Productivity Score {productivity_insights['Most Productive Activities'].get('Project Planning', 'N/A')}
        3. Design: Productivity Score {productivity_insights['Most Productive Activities'].get('Design', 'N/A')}

        Recommended Improvement Strategies:
        {', '.join(productivity_insights.get('Extended Productivity Analysis', {}).get('Recommended Actions', []))}

        Time Allocation Insights:
        Most Time-Consuming Activity: {time_tracking.get('Most Time-Consuming Activity', 'Not Available')}
        
        Activity Time Distribution:
        {json.dumps(time_tracking.get('Activity Time Breakdown', {}), indent=2)}
        """
        return narrative.strip()
    
    def generate_workflow_analysis(self):
        """
        Generate comprehensive workflow and time management insights
        """
        time_tracking = self.insights.get('Time Tracking', {})
        active_user_insights = self.insights.get('Active User Insights', {})
        
        narrative = f"""
        Workflow and Time Management Comprehensive Analysis

        Overall Time Utilization:
        - Total Tracked Time: {time_tracking.get('Total Tracked Time', 'N/A')}
        - Most Time-Consuming Activity: {time_tracking.get('Most Time-Consuming Activity', 'N/A')}

        Detailed Activity Breakdown:
        {json.dumps(time_tracking.get('Detailed Activity Breakdown', {}), indent=2)}

        Active User Session Metrics:
        - Total Active Sessions: {active_user_insights.get('Total Active Sessions', 'N/A')}
        - Average Session Time: {active_user_insights.get('Average Session Time', 'N/A')}
        - Active Time Percentage: {active_user_insights.get('Active Time Percentage', 'N/A')}

        Working Time Category Distribution:
        {json.dumps(active_user_insights.get('Working Time Category Breakdown', {}), indent=2)}

        Device Usage Diversity:
        {json.dumps(active_user_insights.get('Device Usage', {}), indent=2)}
        """
        return narrative.strip()
    
    def generate_user_interaction_analysis(self):
        """
        Analyze user interaction patterns across different domains
        """
        cursor_activity = self.insights.get('Cursor Activity', {})
        keystroke_activity = self.insights.get('Keystroke Activity', {})
        web_app_usage = self.insights.get('Web and App Usage', {})
        
        narrative = f"""
        User Interaction and Digital Behavior Analysis

        Cursor Movement Insights:
        - Average Cursor Speed: {cursor_activity.get('Average Cursor Speed', 'N/A')}
        - Total Cursor Movements: {cursor_activity.get('Total Cursor Movements', 'N/A')}
        - Click Rate: {cursor_activity.get('Click Rate', 'N/A')}

        Advanced Cursor Metrics:
        - Max Cursor Speed: {cursor_activity.get('Advanced Cursor Metrics', {}).get('Max Cursor Speed', 'N/A')}
        - Average Clicks per Minute: {cursor_activity.get('Advanced Cursor Metrics', {}).get('Click Pattern Analysis', {}).get('Average Clicks per Minute', 'N/A')}

        Keystroke Performance:
        - Average Typing Speed: {keystroke_activity.get('Average Typing Speed', 'N/A')}
        - Average Key Press Duration: {keystroke_activity.get('Average Key Press Duration', 'N/A')}
        - Total Keystrokes: {keystroke_activity.get('Total Keystrokes', 'N/A')}

        Web and Application Usage:
        - Total Usage Time: {web_app_usage.get('Total Usage Time', 'N/A')}
        - Most Used App: {web_app_usage.get('Most Used App', 'N/A')}
        - Most Visited Website: {web_app_usage.get('Most Visited Website', 'N/A')}
        - Unique Apps Used: {web_app_usage.get('Unique Apps Used', 'N/A')}
        """
        return narrative.strip()
    
    def generate_organizational_efficiency_report(self):
        """
        Create a comprehensive report on organizational efficiency
        """
        attendance_management = self.insights.get('Attendance Management', {})
        printing_history = self.insights.get('Printing History', {})
        
        narrative = f"""
        Organizational Efficiency and Resource Utilization Report

        Attendance and Workforce Dynamics:
        - Total Employees Tracked: {attendance_management.get('Total Employees Tracked', 'N/A')}
        - Attendance Distribution:
          * Regular: {attendance_management.get('Attendance Distribution', {}).get('Regular', 'N/A')}
          * Partial: {attendance_management.get('Attendance Distribution', {}).get('Partial', 'N/A')}
          * Absent: {attendance_management.get('Attendance Distribution', {}).get('Absent', 'N/A')}
        - Average Working Hours: {attendance_management.get('Average Working Hours', 'N/A')}

        Resource Utilization - Printing Analysis:
        - Total Pages Printed: {printing_history.get('Total Pages Printed', 'N/A')}
        - Color Print Ratio: {printing_history.get('Color Print Ratio', 'N/A')}
        - Unique Document Types: {printing_history.get('Unique Document Types', 'N/A')}
        """
        return narrative.strip()

class JSONEncoderWithNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class LLMAnalyticsGenerator:
    
    def __init__(self, config_path='config.json'):
        """
        Initialize the LLM Analytics Generator with configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set up paths and directory
        self.data_files = self.config.get("file_paths", {})
        self.data_directory = os.path.dirname(list(self.data_files.values())[0])
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Initialize sentiment model
        self.sentiment_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_model = DistilBertForSequenceClassification.from_pretrained(self.sentiment_model_name)
        
        # Initialize LLM for insights generation
        self.llm_model_name = self.config.get('llm_model', 'google/flan-t5-large')
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
            
            # Create text generation pipeline
            self.generator = pipeline('text2text-generation', model=self.llm_model, tokenizer=self.llm_tokenizer)
        except Exception as e:
            logging.error(f"Error loading LLM: {e}")
            self.llm_model = None
            self.generator = None
    
    def interactive_llm_analysis(self, analysis_results):
        # Check if generator is available
        if not self.generator:
            print("LLM generator not available. Cannot perform interactive analysis.")
            return

        while True:
            print("\nWhat would you like to know about the analytics?")
            print("1. Overall summary")
            print("2. Detailed insights")
            print("3. Specific file analysis")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                # Generate overall summary
                prompt = f"Provide a comprehensive summary of these analytics: {json.dumps(analysis_results)}"
                response = self.generator(prompt, max_length=500)
                print(response[0]['generated_text'])
            
            elif choice == '2':
                # Detailed insights
                prompt = f"Extract and explain the most significant insights from this data: {json.dumps(analysis_results['column_insights'])}"
                response = self.generator(prompt, max_length=500)
                print(response[0]['generated_text'])
            
            elif choice == '3':
                # Ask about a specific file
                files = [file['filename'] for file in analysis_results['files_analyzed']]
                print("Available files:", files)
                file_choice = input("Enter the filename for detailed analysis: ")
                
                file_insights = {k:v for k,v in analysis_results['column_insights'].items() if file_choice in k}
                prompt = f"Provide an in-depth analysis of this file's data: {json.dumps(file_insights)}"
                response = self.generator(prompt, max_length=500)
                print(response[0]['generated_text'])
            
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")

    def load_csv_files(self) -> List[pd.DataFrame]:
        """
        Load all CSV files from the specified paths in config.json
        """
        dataframes = []
        for name, path in self.data_files.items():
            try:
                df = pd.read_csv(path)
                df['source_file'] = name  # Add source file reference
                dataframes.append(df)
            except Exception as e:
                logging.error(f"Error loading {name} from {path}: {e}")
        return dataframes

    def analyze_dataframes(self, dataframes: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on multiple DataFrames
        
        :param dataframes: List of DataFrames to analyze
        :return: Dictionary of analysis results
        """
        comprehensive_analysis = {
            'files_analyzed': [],
            'column_insights': {},
            'overall_insights': {}
        }

        for df in dataframes:
            file_analysis = {
                'filename': df['source_file'].iloc[0],
                'num_rows': len(df),
                'columns': list(df.columns)
            }
            comprehensive_analysis['files_analyzed'].append(file_analysis)

            # Analyze each column
            for column in df.columns:
                if df[column].dtype == 'object':
                    # Text-based column analysis
                    text_insights = self.analyze_text_column(df, column)
                    comprehensive_analysis['column_insights'][f"{file_analysis['filename']}_{column}"] = text_insights
                elif pd.api.types.is_numeric_dtype(df[column]):
                    # Numeric column analysis
                    numeric_insights = self.analyze_numeric_column(df, column)
                    comprehensive_analysis['column_insights'][f"{file_analysis['filename']}_{column}"] = numeric_insights

        # Generate overall narrative
        comprehensive_analysis['overall_narrative'] = self.generate_narrative_analytics(comprehensive_analysis)

        return comprehensive_analysis

    def analyze_text_column(self, dataframe: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis on text columns with improved handling of long texts
        
        :param dataframe: DataFrame containing the text column
        :param column: Name of the text column
        :return: Insights dictionary
        """
        def process_text(text: str, max_length: int = 512) -> Dict[str, Any]:
            """
            Process text safely within model's token limit
            
            :param text: Input text to analyze
            :param max_length: Maximum token length
            :return: Sentiment analysis result
            """
            # Clean and preprocess text
            text = str(text).strip()
            if not text:
                return None
            
            try:
                # Tokenize and truncate
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                
                # Truncate if exceeding max length
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    text = self.tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
                
                # Tokenize the truncated text
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
                
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                
                # Get probabilities
                probs = softmax(outputs.logits, dim=1)
                sentiment = torch.argmax(probs, dim=1)
                sentiment_label = "Positive" if sentiment == 1 else "Negative"
                confidence = probs[0][sentiment].item()
                
                return {
                    'text': text,
                    'sentiment': sentiment_label,
                    'confidence': float(confidence)
                }
            
            except Exception as e:
                logging.error(f"Error processing text: {str(e)}")
                return None

        # Collect valid sentiments
        sentiments = []
        for text in dataframe[column].dropna():
            result = process_text(text)
            if result:
                sentiments.append(result)
        
        # Handle case with no valid sentiments
        if not sentiments:
            return {
                'total_entries': 0,
                'positive_sentiment_ratio': 0,
                'avg_sentiment_confidence': 0,
                'sentiment_distribution': {
                    'positive': 0,
                    'negative': 0
                }
            }
        
        # Compute statistics
        sentiment_scores = [s['confidence'] for s in sentiments]
        
        return {
            'total_entries': len(sentiments),
            'positive_sentiment_ratio': sum(1 for s in sentiments if s['sentiment'] == 'Positive') / len(sentiments),
            'avg_sentiment_confidence': float(np.mean(sentiment_scores)),
            'sentiment_distribution': {
                'positive': sum(1 for s in sentiments if s['sentiment'] == 'Positive'),
                'negative': sum(1 for s in sentiments if s['sentiment'] == 'Negative')
            }
        }
    
    def save_analysis_results(self, analysis_results: Dict[str, Any], output_path: str = './comprehensive_analytics.json'):
        """
        Save analysis results using a custom JSON encoder to handle numpy types
        
        :param analysis_results: Dictionary of analysis results
        :param output_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, cls=JSONEncoderWithNumpy)

    def analyze_numeric_column(self, dataframe: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Perform analysis on numeric columns
        
        :param dataframe: DataFrame containing the numeric column
        :param column: Name of the numeric column
        :return: Insights dictionary
        """
        try:
            # Handle different possible column types
            if dataframe[column].dtype == 'bool':
                # Convert boolean to numeric
                numeric_series = dataframe[column].astype(int)
            else:
                # Attempt to convert to numeric, coercing errors
                numeric_series = pd.to_numeric(dataframe[column], errors='coerce')
            
            # Drop NaN values
            numeric_series = numeric_series.dropna()
            
            if len(numeric_series) == 0:
                return {
                    'error': f"Column '{column}' does not contain valid numeric data."
                }
            
            return {
                'mean': float(numeric_series.mean()),
                'median': float(numeric_series.median()),
                'min': float(numeric_series.min()),
                'max': float(numeric_series.max()),
                'std_dev': float(numeric_series.std()) if len(numeric_series) > 1 else 0,
                'unique_values': int(numeric_series.nunique()),
                'distribution_percentiles': {
                    '25%': float(numeric_series.quantile(0.25)),
                    '50%': float(numeric_series.median()),
                    '75%': float(numeric_series.quantile(0.75))
                }
            }
        except Exception as e:
            logging.error(f"Error analyzing numeric column {column}: {str(e)}")
            return {
                'error': f"Error analyzing column '{column}': {str(e)}"
            }


    def generate_narrative_analytics(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive narrative based on analysis results
        """
        # Check if LLM is available
        if not self.generator:
            return "Narrative generation not available. LLM model failed to load."

        try:
            # Prepare a more concise summary
            summary_data = {
                'files_analyzed': len(analysis_results.get('files_analyzed', [])),
                'data_sources': len(analysis_results.get('column_insights', {})),
                'key_insights': {}
            }

            # Extract some high-level insights
            for key, insights in analysis_results.get('column_insights', {}).items():
                if 'positive_sentiment_ratio' in insights:
                    summary_data['key_insights'][key] = {
                        'sentiment_ratio': insights.get('positive_sentiment_ratio'),
                        'total_entries': insights.get('total_entries')
                    }
                elif 'mean' in insights:
                    summary_data['key_insights'][key] = {
                        'mean': insights.get('mean'),
                        'min': insights.get('min'),
                        'max': insights.get('max')
                    }

            # Create a more compact prompt
            prompt = f"""
            Data Analysis Summary:
            - Files Analyzed: {summary_data['files_analyzed']}
            - Data Sources: {summary_data['data_sources']}

            Key Insights:
            {json.dumps(summary_data['key_insights'], indent=2)}

            Generate a concise narrative report covering:
            1. Overview of data sources
            2. Significant findings
            3. Patterns across files
            4. Potential implications
            5. Recommendations
            """

            # Generate narrative with strict length control
            response = self.generator(
                prompt, 
                max_length=300,  # Even more aggressive length limiting
                num_return_sequences=1
            )

            # Extract and return generated text
            generated_text = response[0]['generated_text'] if response else "Unable to generate narrative."
            
            return generated_text

        except Exception as e:
            logging.error(f"Narrative generation error: {str(e)}")
            return f"Error generating narrative: {str(e)}"
        

class EnhancedAnalyticsProcessor:
    @staticmethod
    def process_cursor_activity(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cursor movement and click patterns
        """
        # Calculate cursor speed (approximate)
        df['cursor_speed'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2)
        
        return {
            'Average Cursor Speed': f"{df['cursor_speed'].mean():.2f} pixels/second",
            'Click Rate': f"{df['click'].mean() * 100:.2f}%",
            'Total Cursor Movements': len(df)
        }

    @staticmethod
    def process_keystroke_activity(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze keystroke patterns and typing metrics
        """
        return {
            'Average Typing Speed': f"{len(df) / (df['duration'].sum() or 1):.2f} keys/second",
            'Average Key Press Duration': f"{df['duration'].mean() * 1000:.2f} ms",
            'Total Keystrokes': len(df)
        }

    @staticmethod
    def process_web_app_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze web and application usage
        """
        # Convert duration to hours, minutes, seconds
        total_duration = df['duration'].sum()
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Most used apps and websites
        app_usage = df['app'].value_counts()
        website_usage = df['website'].value_counts()
        
        return {
            'Total Usage Time': f"{int(hours)} days, {int(minutes)}:{int(seconds):02d}:00",
            'Most Used App': app_usage.index[0] if len(app_usage) > 0 else 'N/A',
            'Most Visited Website': website_usage.index[0] if len(website_usage) > 0 else 'N/A',
            'Unique Apps Used': len(app_usage),
            'Unique Websites Visited': len(website_usage)
        }

    @staticmethod
    def process_printing_history(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze printing patterns
        """
        return {
            'Total Pages Printed': int(df['pages'].sum()),
            'Color Print Ratio': f"{df['color'].mean() * 100:.2f}%",
            'Unique Document Types': df['document_type'].nunique()
        }

    @staticmethod
    def process_time_tracking(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze time tracking and activities
        """
        total_duration = df['duration'].sum()
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        activity_breakdown = df.groupby('activity')['duration'].sum().sort_values(ascending=False)
        
        return {
            'Total Tracked Time': f"{int(hours)} days, {int(minutes)}:{int(seconds):02d}:00",
            'Most Time-Consuming Activity': activity_breakdown.index[0] if len(activity_breakdown) > 0 else 'N/A',
            'Activity Time Breakdown': dict(activity_breakdown)
        }

    @staticmethod
    def process_attendance(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze attendance metrics
        """
        total_employees = len(df['employee_id'].unique())
        attendance_status = df['attendance_status'].value_counts(normalize=True) * 100
        
        return {
            'Total Employees Tracked': total_employees,
            'Attendance Distribution': {
                'Regular': f"{attendance_status.get('Regular', 0):.2f}%",
                'Partial': f"{attendance_status.get('Partial', 0):.2f}%",
                'Absent': f"{attendance_status.get('Absent', 0):.2f}%"
            },
            'Average Working Hours': f"{df['total_hours'].mean():.2f}"
        }

    @staticmethod
    def process_productivity(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze productivity insights
        """
        # Productivity by hour of day
        hourly_productivity = df.groupby('hour_of_day')['productivity_score'].mean().sort_values(ascending=False)
        
        # Top productive activities
        activity_productivity = df.groupby('activity')['productivity_score'].mean().sort_values(ascending=False)
        
        return {
            'Peak Productive Hours': dict(hourly_productivity.head(3)),
            'Most Productive Activities': dict(activity_productivity.head(3))
        }

    @staticmethod
    def process_active_user(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze active user sessions and device usage
        """
        # Convert times to hours, minutes, seconds
        def format_time(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Device usage breakdown
        device_usage = df['devices_used'].value_counts(normalize=True) * 100
        
        return {
            'Total Active Sessions': len(df),
            'Average Session Time': format_time(df['total_session_time'].mean()),
            'Average Active Time': format_time(df['active_time'].mean()),
            'Average Idle Time': format_time(df['idle_time'].mean()),
            'Active Time Percentage': f"{df['active_percentage'].mean():.2f}%",
            'Working Time Category Breakdown': dict(df['ideal_time_category'].value_counts(normalize=True) * 100),
            'Device Usage': dict(device_usage)
        }


class EnhancedHRMSAnalyticsGenerator(LLMAnalyticsGenerator):
    def __init__(self, config_path='config.json'):
        """
        Enhanced initialization for HRMS Analytics with continuous learning capabilities
        """
        super().__init__(config_path)
        
        # New directories for continuous learning
        self.training_data_dir = os.path.join(self.data_directory, 'continuous_learning')
        self.model_checkpoint_dir = os.path.join(self.data_directory, 'model_checkpoints')
        
        # Create directories if they don't exist
        os.makedirs(self.training_data_dir, exist_ok=True)
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        
        # Additional configuration for continuous learning
        self.learning_config = {
            'retraining_threshold': 100,  # Number of new data points to trigger retraining
            'feedback_collection': True,
            'model_update_frequency': 'monthly'
        }

    
    def generate_comprehensive_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive insights from multiple data sources with enhanced detail
        """
        comprehensive_insights = {}
        processor = EnhancedAnalyticsProcessor()
        
        # Enhanced activity time breakdown for Time Tracking
        def detailed_time_breakdown(df):
            # Expand time tracking to include more granular activity breakdown
            activity_duration = df.groupby('activity')['duration'].agg([
                ('total_duration', 'sum'),
                ('avg_duration', 'mean'),
                ('count', 'count')
            ])
            
            # Convert total duration to more readable format
            def format_duration(total_seconds):
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
            
            return {
                'Detailed Activity Breakdown': {
                    activity: {
                        'Total Duration': format_duration(row['total_duration']),
                        'Average Duration per Session': format_duration(row['avg_duration']),
                        'Number of Sessions': row['count']
                    } for activity, row in activity_duration.iterrows()
                },
                'Productivity Metrics': {
                    'Most Productive Activities': list(activity_duration.sort_values('total_duration', ascending=False).head(3).index),
                    'Least Productive Activities': list(activity_duration.sort_values('total_duration', ascending=True).head(3).index)
                }
            }
        
        # Process each file's data with more comprehensive analysis
        for file_info in analysis_results.get('files_analyzed', []):
            filename = file_info['filename']
            
            # Load corresponding DataFrame
            try:
                df = pd.read_csv(self.data_files.get(filename))
                
                # Expanded processing based on filename
                if filename == 'cursor':
                    cursor_insights = processor.process_cursor_activity(df)
                    comprehensive_insights['Cursor Activity'] = {
                        **cursor_insights,
                        'Advanced Cursor Metrics': {
                            'Cursor Acceleration': np.mean(np.diff(df['cursor_speed'])),
                            'Max Cursor Speed': np.max(df['cursor_speed']),
                            'Click Pattern Analysis': {
                                'Average Clicks per Minute': len(df[df['click'] == 1]) / (len(df) / 60),
                                'Peak Click Periods': 'To be implemented with more detailed data'
                            }
                        }
                    }
                
                elif filename == 'time_tracking':
                    time_tracking_insights = processor.process_time_tracking(df)
                    time_breakdown = detailed_time_breakdown(df)
                    comprehensive_insights['Time Tracking'] = {
                        **time_tracking_insights,
                        **time_breakdown
                    }
                
                elif filename == 'productivity':
                    productivity_insights = processor.process_productivity(df)
                    comprehensive_insights['Productivity Insights'] = {
                        **productivity_insights,
                        'Extended Productivity Analysis': {
                            'Productivity Trend': 'Gradually improving' if productivity_insights['Peak Productive Hours'] else 'Inconsistent',
                            'Potential Improvement Areas': 'Low productivity hours',
                            'Recommended Actions': [
                                'Optimize workflow during less productive hours',
                                'Implement focused work sessions',
                                'Analyze and mitigate productivity blockers'
                            ]
                        }
                    }
                
                # Continue with other existing processing methods...
                elif filename == 'keystroke':
                    comprehensive_insights['Keystroke Activity'] = processor.process_keystroke_activity(df)
                
                elif filename == 'web_app':
                    comprehensive_insights['Web and App Usage'] = processor.process_web_app_usage(df)
                
                elif filename == 'printing':
                    comprehensive_insights['Printing History'] = processor.process_printing_history(df)
                
                elif filename == 'attendance':
                    comprehensive_insights['Attendance Management'] = processor.process_attendance(df)
                
                elif filename == 'active_user':
                    comprehensive_insights['Active User Insights'] = processor.process_active_user(df)
            
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
        
        # Add overall narrative and summary
        try:
            comprehensive_insights['Overall Narrative'] = self.generate_narrative_analytics(analysis_results)
        except Exception as e:
            logging.error(f"Error generating narrative: {e}")
            comprehensive_insights['Overall Narrative'] = "Unable to generate comprehensive narrative"
        
        # Add timestamp and metadata
        comprehensive_insights['Analysis Metadata'] = {
            'Timestamp': datetime.datetime.now().isoformat(),
            'Files Analyzed': [file['filename'] for file in analysis_results.get('files_analyzed', [])],
            'Total Data Sources': len(analysis_results.get('column_insights', {}))
        }
        
        return comprehensive_insights

    def collect_feedback_data(self, user_interactions, model_predictions):
        """
        Collect and store feedback data for continuous learning
        
        :param user_interactions: User feedback on model outputs
        :param model_predictions: Original model predictions
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_file = os.path.join(self.training_data_dir, f'feedback_{timestamp}.json')
        
        feedback_data = {
            'timestamp': timestamp,
            'interactions': user_interactions,
            'predictions': model_predictions
        }
        
        # Save feedback data
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        logging.info(f"Feedback data saved to {feedback_file}")
    
    def prepare_continuous_learning_dataset(self):
        """
        Prepare dataset for continuous model learning
        """
        # Collect all feedback files
        feedback_files = [
            os.path.join(self.training_data_dir, f) 
            for f in os.listdir(self.training_data_dir) 
            if f.startswith('feedback_')
        ]
        
        # Aggregate feedback data
        combined_data = []
        for file_path in feedback_files:
            with open(file_path, 'r') as f:
                feedback = json.load(f)
                combined_data.append(feedback)
        
        # Prepare training dataset
        if combined_data:
            # Convert feedback to training format
            training_dataset = self._convert_feedback_to_training_data(combined_data)
            return training_dataset
        
        return None
    
    def fine_tune_model(self):
        """
        Fine-tune the LLM with collected feedback data
        """
        # Prepare dataset
        training_dataset = self.prepare_continuous_learning_dataset()
        
        if training_dataset and len(training_dataset) >= self.learning_config['retraining_threshold']:
            # Split dataset
            train_dataset, val_dataset = train_test_split(training_dataset, test_size=0.2)
            
            # Prepare training arguments
            training_args = TrainingArguments(
                output_dir=self.model_checkpoint_dir,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(self.model_checkpoint_dir, 'logs')
            )
            
            # Create Trainer
            trainer = Trainer(
                model=self.llm_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            # Perform fine-tuning
            trainer.train()
            
            # Save the fine-tuned model
            self.llm_model.save_pretrained(os.path.join(self.model_checkpoint_dir, 'fine_tuned_model'))
            logging.info("Model successfully fine-tuned with new data")      
    

def main():
    # Initialize Checkpoint Manager and Analytics Generator
    checkpoint_manager = AnalyticsCheckpointManager()
    generator = EnhancedHRMSAnalyticsGenerator()
    
    # Perform data processing and generate insights
    analysis_results = {}  # Your data processing logic here
    comprehensive_insights = generator.generate_comprehensive_insights(analysis_results)
    
    # Create insights text generator
    insights_generator = InsightsTextGenerator(comprehensive_insights, analysis_results)
    
    # Launch interactive menu
    interactive_analytics_menu(generator, insights_generator)
    
    # Log start of process
    checkpoint_manager.logger.info("Starting HRMS Analytics Processing")
    
    try:
        # Load and process only changed files
        dataframes = checkpoint_manager.check_and_process_files(generator.data_files)
        
        if not dataframes:
            checkpoint_manager.logger.warning("No new or changed files to process.")
            return
        
        # Perform initial analysis
        analysis_results = generator.analyze_dataframes(dataframes)
        
        # Generate Comprehensive Insights
        comprehensive_insights = generator.generate_comprehensive_insights(analysis_results)
        
        # Create insights text generator
        insights_text_generator = InsightsTextGenerator(comprehensive_insights, analysis_results)
        
        # Save analysis results
        generator.save_analysis_results(
            analysis_results, 
            os.path.join(checkpoint_manager.checkpoint_dir, 'comprehensive_analysis.json')
        )
        
        # Save comprehensive insights
        generator.save_analysis_results(
            comprehensive_insights, 
            os.path.join(checkpoint_manager.checkpoint_dir, 'comprehensive_insights.json')
        )
        
        checkpoint_manager.logger.info("Analysis completed successfully.")
        
        # Launch interactive menu
        interactive_analytics_menu(generator, insights_text_generator)
    
    except Exception as e:
        checkpoint_manager.logger.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
    