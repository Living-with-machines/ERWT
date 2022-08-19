# bNERT

# Temporary Readme with overview of data, code and models

## Models

### bnert_time

`/datadrive_2/bnert_time`: DistilBERT model created by earlier experiments. Trained on sentences longer than 10 topics and and OCR quality higher than 0.75. For this models we simple prepended the sentences with the year and a `[SEP]` special token. Example of use:

```python
tokenizer = AutoTokenizer.from_pretrained("/datadrive_2/bnert_time")
mask_filler = pipeline(
    "fill-mask", model="/datadrive_2/bnert_time", top_k=10, tokenizer=tokenizer
)

text = f"1810 [SEP] [MASK] Majesty."
preds = mask_filler(text)
```

Original train-test split is lost. 

### bnert-time-st-y (almost completed ETA 3pm)

`/datadrive_2/bnert-time-st-y`: DistilBERT model trained on 0.5 billion tokens (see below for path to dataset). We divided the text into chunks of 100 words (i.e. no sentence splitting). 

To use preprocess text as follows:
```python
text = f"[1810] [SEP] [MASK] Majesty."
preds = mask_filler(text)
```

Data used for training is stored here: `/datadrive_2/frozen_corpus`. It still requires preprocessing, which is captured in `PrepareDataset.ipynb`.

## Data

## Code

Code is currently stored in `/datadrive/bNERT` (should be pushed to GitHub after cleaning). Main Notebook is `PrepareDataset.ipynb` which includes a data loading, preprocessing and training code. This is still under construction and ~~should~~ will be cleaned later on.
