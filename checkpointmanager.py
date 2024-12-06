import os
import json
import hashlib
import logging
import pickle
import datetime
from typing import Dict, Any
# from transformers import (
#     DistilBertTokenizer, 
#     DistilBertForSequenceClassification, 
#     AutoTokenizer, 
#     AutoModelForSeq2SeqLM, 
#     pipeline
# )
# from torch.nn.functional import softmax

# from sklearn.model_selection import train_test_split
# from transformers import Trainer, TrainingArguments

class CheckpointManager:
    def __init__(self, checkpoint_dir='./checkpoints'):
        """
        Initialize CheckpointManager with a directory for storing checkpoints
        
        :param checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def find_latest_checkpoint(self, data_files: Dict[str, str]) -> Dict:
        """
        Find the most recent checkpoint that matches current data sources
        
        :param data_files: Dictionary of data file paths
        :return: Checkpoint dictionary or None
        """
        try:
            # Implement checkpoint retrieval logic here
            # This is a placeholder implementation
            checkpoint_files = sorted(
                [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.json')],
                reverse=True
            )
            
            if checkpoint_files:
                latest_checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_files[0])
                with open(latest_checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                
                # Add validation logic to ensure checkpoint matches current data sources
                # For example, check if checkpoint's data sources match input data_files
                return checkpoint
            
            return None
        
        except Exception as e:
            logging.error(f"Error finding checkpoint: {e}")
            return None

    def _generate_file_hash(self, filepath):
        """
        Generate a unique hash for a file to track changes
        
        :param filepath: Path to the file
        :return: SHA-256 hash of file contents
        """
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash
    
    def create_checkpoint(self, data_sources: Dict[str, str], analysis_results: Dict[str, Any]):
        """
        Create a checkpoint of the current analysis state
        
        :param data_sources: Dictionary of data source files with their paths
        :param analysis_results: Comprehensive analysis results to save
        """
        # Generate checkpoint metadata
        checkpoint_metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data_source_hashes': {},
            'analysis_results': analysis_results
        }
        
        # Generate and store file hashes
        for name, filepath in data_sources.items():
            try:
                checkpoint_metadata['data_source_hashes'][name] = self._generate_file_hash(filepath)
            except Exception as e:
                logging.warning(f"Could not generate hash for {name}: {e}")
        
        # Create unique checkpoint filename
        checkpoint_filename = f"checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_metadata, f)
        
        logging.info(f"Checkpoint created: {checkpoint_path}")
        return checkpoint_path
    
    def find_latest_checkpoint(self, data_sources: Dict[str, str]):
        """
        Find the most recent valid checkpoint that matches current data sources
        
        :param data_sources: Current data source files
        :return: Checkpoint data or None if no valid checkpoint found
        """
        # Get all checkpoint files, sorted by creation time (most recent first)
        checkpoint_files = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_')],
            reverse=True
        )
        
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
            
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Validate checkpoint against current data sources
                valid_checkpoint = True
                for name, filepath in data_sources.items():
                    if name not in checkpoint_data['data_source_hashes']:
                        valid_checkpoint = False
                        break
                    
                    current_hash = self._generate_file_hash(filepath)
                    if current_hash != checkpoint_data['data_source_hashes'][name]:
                        valid_checkpoint = False
                        break
                
                if valid_checkpoint:
                    logging.info(f"Valid checkpoint found: {checkpoint_file}")
                    return checkpoint_data
            
            except Exception as e:
                logging.warning(f"Error processing checkpoint {checkpoint_file}: {e}")
        
        return None