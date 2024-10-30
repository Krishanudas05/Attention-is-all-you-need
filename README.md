# Attention Is All You Need - Transformer Implementation

This repository contains a PyTorch implementation of the Transformer model, as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. This model has become a foundational architecture for tasks in Natural Language Processing (NLP), including machine translation, question answering, and text summarization.

## Overview

The Transformer model relies entirely on self-attention to compute representations of input and output sequences, dispensing with recurrence entirely. This implementation is based on the "Annotated Transformer" and includes the following key components:

- **Scaled Dot-Product Attention**: Calculates attention weights for the query, key, and value vectors.
- **Multi-Head Attention**: Uses multiple attention heads to jointly attend to different parts of the sequence.
- **Position-wise Feed-Forward Networks**: Applies a feed-forward neural network to each position separately.
- **Positional Encoding**: Adds positional information to the tokens, allowing the model to capture sequence order.
- **Encoder and Decoder Stacks**: Composed of multi-head attention and feed-forward layers.

This implementation can be fine-tuned on specific NLP tasks such as question answering with additional configuration.

## Requirements

Install the following packages before running the code:

```bash
pip install torch transformers numpy matplotlib pandas scikit-learn ```

## Files

- `transformer.py`: Main Transformer model implementation including all modules (attention, multi-head attention, positional encoding, encoder, and decoder).
- `train.py`: Training loop with fine-tuning for specific tasks (e.g., question answering).
- `evaluate.py`: Evaluation functions to calculate metrics like Exact Match (EM) and F1 Score for question answering.
- `README.md`: This file, which provides an overview and instructions for setup and usage.

## Quick Start

### Step 1: Load the Model and Tokenizer

```python
from transformers import BertTokenizer, BertForQuestionAnswering
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
