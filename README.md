# Transformer Twitter Sentiment Classifier (PyTorch)

A Transformer-based neural network built for Twitter sentiment classification using PyTorch and GPU acceleration.

## Highlights
- Implemented scaled dot-product attention and efficient multi-head attention from scratch (no nn.MultiheadAttention)
- Stacked Transformer encoder blocks (LayerNorm + residual connections + MLP)
- Learned token + positional embeddings, then mean-pooled over time for classification
- Trained with AdamW + weight decay and a learning-rate drop mid-training
- Includes inference examples: random test tweet + custom user-written tweets with predicted class probabilities

## Model 
- Encoder-only Transformer
- Mean pooling over sequence dimension → linear classifier head (2 classes)

## Results
Final test accuracy: ~84.68%
Final test error: ~15.32%
(See `train.ipynb` for full logs and evaluation)


## Files
- `model.py` — Transformer architecture (attention blocks + embeddings + classifier head)
- `train.ipynb` — training, evaluation, and inference examples
- `utils_twitter.py` — data utilities (minibatching, tokenization helpers, error metric, display helpers)

## Data 
This repo expects a course-provided tokenized Twitter dataset (not included here).

`train.ipynb` loads the preprocessed tensors (e.g., `train_data`, `train_label`, `test_data`, `test_label`) using the helper functions in `utils_twitter.py`. You can still view the model architecture in `model.py` and run the notebook after swapping in your own tokenized data.

Expected paths (example):
- `data_nlp/twitter/train_data.pt`
- `data_nlp/twitter/train_label.pt`
- `data_nlp/twitter/test_data.pt`
- `data_nlp/twitter/test_label.pt`

## How to Run
1. Open train.ipynb (Colab or local Jupyter)
2. Install dependencies if needed:
   ```bash
   pip install torch
   ```
3. Run cells top-to-bottom to train and evaluate
4. Scroll to the bottom for inference examples and probability outputs

## Skills Demonstrated
Python, PyTorch, Transformers, self-attention, sequence modeling, GPU training, model evaluation
