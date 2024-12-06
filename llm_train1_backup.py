import os
import torch
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline
)
from torch.nn.functional import softmax

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
        def process_text_in_chunks(text: str, chunk_size: int = 512) -> List[Dict[str, Any]]:
            """
            Process long text by breaking it into chunks
            
            :param text: Input text to analyze
            :param chunk_size: Maximum token length per chunk
            :return: List of sentiment analysis results for each chunk
            """
            # Clean and preprocess text
            text = str(text).strip()
            if not text:
                return []
            
            # Tokenize the full text
            full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Process text in overlapping chunks
            chunk_results = []
            for start in range(0, len(full_tokens), chunk_size - 100):  # 100 token overlap
                # Extract chunk with some context from previous chunk
                chunk_tokens = full_tokens[max(0, start-100):start+chunk_size]
                
                # Decode chunk tokens
                chunk_text = self.tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
                
                try:
                    # Tokenize the chunk
                    inputs = self.tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    
                    with torch.no_grad():
                        outputs = self.sentiment_model(**inputs)
                    
                    # Get probabilities
                    probs = softmax(outputs.logits, dim=1)
                    sentiment = torch.argmax(probs, dim=1)
                    sentiment_label = "Positive" if sentiment == 1 else "Negative"
                    confidence = probs[0][sentiment].item()
                    
                    chunk_results.append({
                        'text': chunk_text,
                        'sentiment': sentiment_label,
                        'confidence': float(confidence)
                    })
                
                except Exception as e:
                    logging.error(f"Error processing text chunk: {str(e)}")
            
            return chunk_results

        # Collect results from all texts
        all_sentiments = []
        
        # Process each text in the column
        for text in dataframe[column].dropna():
            try:
                chunk_sentiments = process_text_in_chunks(text)
                all_sentiments.extend(chunk_sentiments)
            except Exception as e:
                logging.error(f"Error processing text: {str(e)}")
        
        # Handle case with no valid sentiments
        if not all_sentiments:
            return {
                'total_entries': 0,
                'positive_sentiment_ratio': 0,
                'avg_sentiment_confidence': 0,
                'sentiment_distribution': {
                    'positive': 0,
                    'negative': 0
                }
            }
        
        # Compute overall statistics
        sentiment_scores = [s['confidence'] for s in all_sentiments]
        
        return {
            'total_entries': len(all_sentiments),
            'positive_sentiment_ratio': sum(1 for s in all_sentiments if s['sentiment'] == 'Positive') / len(all_sentiments),
            'avg_sentiment_confidence': np.mean(sentiment_scores),
            'sentiment_distribution': {
                'positive': sum(1 for s in all_sentiments if s['sentiment'] == 'Positive'),
                'negative': sum(1 for s in all_sentiments if s['sentiment'] == 'Negative')
            }
        }

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
        
    

def main():
    # Initialize LLM Analytics Generator
    generator = LLMAnalyticsGenerator()
    
    # Load CSV files
    dataframes = generator.load_csv_files()
    
    if not dataframes:
        logging.error("No dataframes loaded. Please check the file paths.")
        return
    
    # Perform analysis
    analysis_results = generator.analyze_dataframes(dataframes)
    
    # Save results
    output_path = './comprehensive_analytics.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Interactive LLM analysis
    generator.interactive_llm_analysis(analysis_results)

if __name__ == "__main__":
    main()