# Transformer from Scratch: English-to-Vietnamese Translation

This repository contains a Transformer implementation based on the paper **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** by Vaswani et al. The model is coded from scratch (without relying on high-level libraries like `transformers`) and is trained on an English-to-Vietnamese dataset that including 600k sentences collected and processed on TED..

![image](https://github.com/user-attachments/assets/38d5815b-3af0-4735-bbba-bd5c9b121409)


## Table of Contents
- [Overview](#overview)
- [Files and Directories](#files-and-directories)
- [Dependencies](#dependencies)
- [Data](#data)
  - [Training Set](#training-set)
  - [Validation and Test Sets](#validation-and-test-sets)
- [Usage](#usage)
  - [Training](#training)
  - [Validation](#validation)
  - [Testing](#testing)
  - [Inference](#inference)
- [References](#references)

## Overview

1. **Model Architecture:**  
   - Implements a multi-head attention mechanism, fully connected layers, and positional embeddings. Positional embeddings are learned via an nn.Embedding layer, deviating slightly from the fixed sinusoidal embeddings in the original "Attention Is All You Need" paper. 
   - Consists of an `Encoder` (for reading English text) and a `Decoder` (for generating the Vietnamese translation).

2. **Training Pipeline:**  
   - Uses cross-entropy loss and teacher forcing with `<sos>` and `<eos>` tokens for the target.
   - A learning rate scheduler (warmup + inverse sqrt decay) is optionally included.
   - Produces a final checkpoint (`best_model.pth`) upon reaching the best validation loss.

3. **Data Splits:**  
   - **`train.en` / `train.vi`** for training  
   - **`tst2012.en` / `tst2012.vi`** for validation  
   - **`tst2013.en` / `tst2013.vi`** for testing  

## Files and Directories

- **`Transformer.py`**  
  Defines the core `Transformer` class (with encoder, decoder, and embedding logic).

- **`Encoder.py`** / **`Decoder.py`**  
  Separate modules containing the `Encoder`, `DecoderBlock`, `SelfAttention`, and other components that make up the Transformer.

- **`SelfAttention.py`**  
  Implements the multi-head attention mechanism used by both the encoder and decoder blocks.

- **`train.en` / `train.vi`**  
  Training data: parallel English-Vietnamese sentences.

- **`tst2012.en` / `tst2012.vi`**  
  Parallel sentences used for validation (development set).

- **`tst2013.en` / `tst2013.vi`**  
  Parallel sentences used for final testing.

- **`best_model.pth`**  
  [Download my pre-train model](https://drive.google.com/file/d/1fbtQGSG_O83tmO9k3yQ7Io34MDNIep17/view?usp=drive_link)
- **`main.ipynb`** 
  A Jupyter notebook demonstrating how to:
  1. Preprocess the data  
  2. Initialize and train the Transformer    
  3. Use the saved model for inference  

- **`Attention is all you need.pdf`**  
  The original paper for reference.

## Dependencies

-   **Python:** 3.7 or higher
-   **PyTorch:** 2.5.1 or higher
-   **transformers:** (Hugging Face Transformers library, for `AutoTokenizer`)

Install via:
```bash
pip install torch==2.5.1
pip install numpy
pip install transformers
```

## Data

### Training Set

- **`train.en` / `train.vi`**  
  Contains the raw English and Vietnamese sentences used for training. Each line in `train.en` corresponds to the same line (translation) in `train.vi`.

### Validation and Test Sets

- **`tst2012.en` / `tst2012.vi`**  
- **`tst2013.en` / `tst2013.vi`**  

## Usage

### Training

1. **Preprocess the Data:**  
   - Tokenize, create vocabulary, and prepare `<sos>`, `<eos>`, `<pad>` tokens if using a custom pipeline.  

2. **Run the Training Script or Notebook:**  
   - `main.ipynb` typically contains code to:
     - Build vocab from `train.en` and `train.vi`.
     - Initialize the Transformer model (`Transformer.py`).
     - Train the model, tracking the loss on the training set. 
   - Save a checkpoint after each epoch if avg loss improving as `best_model.pth`.

### Inference

- With the trained model loaded (`best_model.pth`), you can translate new sentences by:
  1. Tokenizing the source text (English).  
  2. Passing token IDs through the encoder.  
  3. Using a greedy decoding in the decoder until `<eos>` is generated.

## References

-   **Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).
-   **GitHub Repository:** [pbcquoc/transformer](https://github.com/pbcquoc/transformer)

**Happy translating!** Feel free to submit issues or pull requests if you find any bugs or want to contribute improvements to the code or email me at haidangnguyen1815@gmail.com.
