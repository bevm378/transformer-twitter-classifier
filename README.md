# Transformer Twitter Sentiment Classifier (PyTorch)

A Transformer-based neural network built for Twitter sentiment classification using PyTorch and GPU acceleration.

## Highlights
- Implemented token embeddings + positional encoding + multi-head self-attention blocks
- End-to-end training/evaluation pipeline with batching, optimizer + learning-rate scheduling, and metric tracking
- Supports attention masking/padding for variable-length sequences (if applicable)
- Example inference on unseen tweets with predicted class probabilities (if applicable)

## Results
Final test accuracy: ~84.68%
(See `train.ipynb` for full logs and evaluation)

## Files
- `model.py` — Transformer architecture (attention blocks + classifier head)
- `train.ipynb` — end-to-end training and evaluation notebook
- `utils.py` — preprocessing/tokenization helpers + plotting (if used)

## Skills Demonstrated
Python, PyTorch, Transformers, self-attention, sequence modeling, GPU training, model evaluation
