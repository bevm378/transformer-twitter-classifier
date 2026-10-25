# Transformer Twitter Sentiment Classifier (PyTorch)

An encoder-only Transformer built from scratch in PyTorch to classify tweets as positive or negative.

## Highlights
- Implemented scaled dot-product attention and multi-head attention from scratch (no `nn.MultiheadAttention`)
- Stacked Transformer encoder blocks with LayerNorm, residual connections, and an MLP
- Learned token and positional embeddings, mean-pooled over the sequence for classification
- Trained with AdamW, weight decay, and a learning-rate drop partway through training
- Includes inference examples on a random test tweet and on custom tweets, with predicted class probabilities

## Results
| Metric | Value |
| --- | --- |
| Test accuracy | ~84.68% |
| Test error | ~15.32% |

See `train.ipynb` for full training logs and evaluation.

## Model
- 4 encoder layers, embedding size 512, 8 attention heads (about 28 million parameters)
- Mean pooling over the sequence, then a linear classifier head (2 classes)

## Training Setup
- 1.43 million training tweets and 150,000 test tweets
- Batch size 256, 5 epochs
- AdamW (weight decay 0.1), learning rate 1e-4, dropped to 1e-5 for the last epochs
- Trained on a GPU in Google Colab

## Files
- `model.py`: Transformer architecture (attention, encoder blocks, embeddings, classifier head)
- `train.ipynb`: training, evaluation, and inference examples
- `utils_twitter.py`: data helpers (tokenization, minibatching, error metric, probability plot), provided as part of the course
- `requirements.txt`: Python dependencies

## Data
The tokenized Twitter dataset was provided for the course and is not included in this repo. The notebook expects these files:

```
data_nlp/twitter/train_data.pt
data_nlp/twitter/train_label.pt
data_nlp/twitter/test_data.pt
data_nlp/twitter/test_label.pt
data_nlp/twitter/word2idx.pt
data_nlp/twitter/idx2word.pt
```

You can still read through the architecture in `model.py` and the saved outputs in `train.ipynb` without the data.

## How to Run
1. Place the data files in `data_nlp/twitter/` as shown above.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `train.ipynb` in Google Colab or Jupyter (a GPU is recommended) and run the cells top to bottom.
4. Scroll to the end for the inference examples and probability outputs.

## Skills Demonstrated
Python, PyTorch, Transformers, self-attention, sequence modeling, GPU training, model evaluation
