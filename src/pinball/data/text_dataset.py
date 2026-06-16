# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging
import random
from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """
    Dataset for text data for next token prediction.
    
    This dataset handles:
    - Loading text files
    - Tokenizing text
    - Creating overlapping chunks of a fixed length
    - Setting up inputs and labels for next token prediction
    """
    
    def __init__(
        self,
        file_paths: Union[str, List[str]],
        tokenizer,
        max_length: int = 512,
        stride: int = 256,
        text_column: Optional[str] = None,
        line_by_line: bool = False,
        pad_to_max_length: bool = True,
        return_tensors: str = "pt",
    ):
        """
        Initialize the dataset.
        
        Args:
            file_paths: Path(s) to input files (text or jsonl)
            tokenizer: Tokenizer to convert text to tokens
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks
            text_column: Column name for text in jsonl files
            line_by_line: Whether to treat each line as a separate example
            pad_to_max_length: Whether to pad sequences to max_length
            return_tensors: Type of tensors to return ("pt" for PyTorch)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.text_column = text_column
        self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.return_tensors = return_tensors
        
        # Load data
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        self.examples = []
        for file_path in file_paths:
            self._load_file(file_path)
        
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def _load_file(self, file_path: str) -> None:
        """
        Load and process a file.
        
        Args:
            file_path: Path to the file
        """
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return
        
        logger.info(f"Loading data from {file_path}")
        
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".jsonl" or ext == ".json":
            # Load JSONL file
            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Processing {file_path}"):
                    if not line.strip():
                        continue
                    
                    try:
                        item = json.loads(line)
                        if self.text_column:
                            if self.text_column in item:
                                text = item[self.text_column]
                            else:
                                logger.warning(f"Column {self.text_column} not found in JSONL item")
                                continue
                        else:
                            # Try to find a text field
                            for key in ["text", "content", "body", "message"]:
                                if key in item:
                                    text = item[key]
                                    break
                            else:
                                logger.warning(f"No text column found in JSONL item: {item.keys()}")
                                continue
                        
                        # Process text
                        self._process_text(text)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON line: {line}")
                        continue
        else:
            # Assume text file
            with open(file_path, "r", encoding="utf-8") as f:
                if self.line_by_line:
                    # Process each line separately
                    for line in tqdm(f, desc=f"Processing {file_path}"):
                        if not line.strip():
                            continue
                        self._process_text(line)
                else:
                    # Process entire file as one text
                    text = f.read()
                    self._process_text(text)
    
    def _process_text(self, text: str) -> None:
        """
        Process a text and add examples.
        
        Args:
            text: Text to process
        """
        # Tokenize the text
        tokenized = self.tokenizer(text)
        
        input_ids = tokenized["input_ids"]
        
        # Skip empty sequences
        if len(input_ids) <= 1:  # 1 for just EOS token
            return
        
        # Create overlapping chunks
        for i in range(0, len(input_ids), self.stride):
            chunk_ids = input_ids[i:i + self.max_length]
            
            # Skip chunks that are too short (less than 10% of max_length)
            if len(chunk_ids) < self.max_length * 0.1:
                continue
            
            self.examples.append({
                "input_ids": chunk_ids,
            })
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            length: Number of examples
        """
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Example index
            
        Returns:
            example: Dictionary with input_ids, labels, and attention_mask
        """
        item = self.examples[idx]
        input_ids = item["input_ids"]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if self.pad_to_max_length and len(input_ids) < self.max_length:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # Convert to tensors if requested
        if self.return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            # Labels are the same as input_ids for next token prediction
            labels = input_ids.clone()
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids,  # Labels are the same as input_ids for next token prediction
            }


def create_dataloaders(
    train_files: List[str],
    tokenizer,
    val_files: Optional[List[str]] = None,
    max_length: int = 512,
    stride: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    text_column: Optional[str] = None,
    line_by_line: bool = False,
    val_split: float = 0.1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_files: List of training files
        tokenizer: Tokenizer to use
        val_files: List of validation files (if None, use val_split)
        max_length: Maximum sequence length
        stride: Stride for overlapping chunks
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        text_column: Column name for text in jsonl files
        line_by_line: Whether to treat each line as a separate example
        val_split: Validation split if val_files is None
        
    Returns:
        train_dataloader: DataLoader for training
        val_dataloader: DataLoader for validation (None if val_split=0)
    """
    # Create training dataset
    train_dataset = TextDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        text_column=text_column,
        line_by_line=line_by_line,
    )
    
    # Create validation dataset
    val_dataset = None
    if val_files:
        val_dataset = TextDataset(
            file_paths=val_files,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            text_column=text_column,
            line_by_line=line_by_line,
        )
    elif val_split > 0:
        # Split training dataset
        dataset_size = len(train_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),  # Fixed seed for reproducibility
            )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_dataloader, val_dataloader