# LLM Pretraining Project

A project demonstrating how to pretrain and upscale Large Language Models (LLMs) using the Hugging Face transformers library.

## Project Overview

This project explores different techniques for pretraining and modifying LLMs, with a focus on the OPT (Open Pretrained Transformer) architecture. The project includes data preparation, model architecture modifications, and training procedures.

## Key Components

### 1. Data Preparation (`data_preparation.ipynb`)
- Downloads pretraining data from Hugging Face's "upstage/Pretraining_Dataset"
- Scrapes Python scripts from GitHub for code examples
- Implements data cleaning steps:
  - Filters short samples
  - Removes text repetitions
  - Deduplicates documents
  - Filters non-English texts using FastText
- Saves processed dataset in Parquet format

### 2. Data Packaging (`data_packaging.ipynb`)
- Loads cleaned dataset from Parquet file
- Implements data packaging steps:
  - Shards dataset into manageable pieces (10 shards)
  - Tokenizes text using OPT-125m tokenizer
  - Adds special tokens (BOS, EOS)
  - Packs tokens into fixed sequence length (32 tokens)
  - Tracks token counts for monitoring
- Creates final packaged dataset with:
  - ~4.6M total tokens
  - 144,505 training examples
  - 32 tokens per sequence
- Saves packaged dataset in Parquet format

### 2. Model Architecture (`modelling.ipynb`)
Explores four different approaches to model initialization:
- Random weight initialization
- Using existing pretrained models
- Downscaling (12 layers → 10 layers)
- Upscaling (12 layers → 16 layers)

Uses the facebook/opt-125m model as the base architecture with configurations:
- Hidden size: 768
- FFN dimension: 3072
- Attention heads: 12
- Vocabulary size: 50272

### 3. Training (`training.ipynb`)
Implements training pipeline using Hugging Face's Trainer with:
- Custom dataset loading
- Training arguments configuration
- Loss monitoring
- Gradient checkpointing
- Mixed precision (bfloat16)

## Requirements
- Python 3.11
- PyTorch
- Transformers
- Datasets
- FastText
- Parquet support