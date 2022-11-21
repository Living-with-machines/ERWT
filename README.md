# ERWT//PEA Language Models

*ERWT* (de; v(m); meervoud: erwten): naam van verschillende peulvruchten

*PEA* noun: pea; plural noun: peas: a spherical green seed that is eaten as a vegetable or as a pulse when dried.

Models availalbe on the [Living with Machines HuggingFace page](https://huggingface.co/Livingwithmachines).

**Consult Model Cards for more information.**

## Models


### [erwt-year](https://huggingface.co/Livingwithmachines/erwt-year)

`Livingwithmachines/erwt-year`: DistilBERT model trained on 0.5 billion tokens (see below for path to dataset). We divided the text into chunks of 100 words (i.e. no sentence splitting). 

To use, please preprocess text as follows:
```python
from transformers import pipeline

mask_filler = pipeline("fill-mask",
                       model='Livingwithmachines/erwt-year')

preds = mask_filler("1810 [DATE] [MASK] Majesty.")
```

### [erwt-year-masked-25](https://huggingface.co/Livingwithmachines/erwt-year-masked-25)


`Livingwithmachines/erwt-year-masked-25`: DistilBERT model trained on same data as the above. Dates are added as standard tokens, and separated from the text with a `[DATE]` special token. We masked the time token with probabiliy of 0.25.

```python
from transformers import pipeline

mask_filler = pipeline("fill-mask",
                       model='Livingwithmachines/erwt-year-masked-25')

preds = mask_filler("1810 [DATE] [MASK] Majesty.")
```


### [erwt-year-masked-75](https://huggingface.co/Livingwithmachines/erwt-year-masked-75)


`Livingwithmachines/erwt-year-masked-75`: DistilBERT model trained on same data as the above. Dates are added as standard tokens, and separated from the text with a `[DATE]` special token. We masked the time token with probabiliy of 0.75.
```python
from transformers import pipeline

mask_filler = pipeline("fill-mask",
                       model='Livingwithmachines/erwt-year-masked-75')

preds = mask_filler("1810 [DATE] [MASK] Majesty.")
```


### hmd-text-only **TBC**


`Livingwithmachines/hmd-text-only`: DistilBERT model trained on HMD without additional prepended metadata. 

```python
text = f"[MASK] Majesty."
preds = mask_filler(text)
```

### [pea-pol-st](https://huggingface.co/Livingwithmachines/pea-pol-st)

**Model Card under construction** 🚧 

`Livingwithmachines/pea-pol-st`: DistilBERT trained with metadata on political leaning. The political labels are
```python
['[lib]', '[con]', '[none]', '[rad]', '[neutr]']
```
Metadata and text and separated by a special `[POL]` token.



```python
from transformers import pipeline

mask_filler = pipeline("fill-mask",
                       model='Livingwithmachines/pea-pol-st')

preds = mask_filler("[lib] [POL] [MASK] Majesty.")
```


### [pea-pol](https://huggingface.co/Livingwithmachines/pea-pol)

**Model Card under construction** 🚧 

`Livingwithmachines/pea-pol`: DistilBERT trained with metadata on political leaning. The political labels are from the tokenizer's standard vocabulary.
```python
['liberal', 'conservative', 'none', 'radical', 'neutral']
```
Metadata and text and separated by a special `[POL]` token.


```python
from transformers import pipeline

mask_filler = pipeline("fill-mask",
                       model='Livingwithmachines/pea-pol')

preds = mask_filler("liberal [POL] [MASK] Majesty.")
```


### [pea-year-pol-loc](https://huggingface.co/Livingwithmachines/pea-year-pol-loc)

**Model Card under construction** 🚧 

`Livingwithmachines/pea-year-pol-loc`: DistilBERT trained with metadata on year, political leaning and location. 
The political labels are from the tokenizer's standard vocabulary.
```python
['liberal', 'conservative', 'none', 'radical', 'neutral']
```
The same applies to year and location.

```python
['liverpool', 'london']
```

Metadata and text and separated by a special `[DATE]`, `[POL]` and `[LOC]` token.s


```python

from transformers import pipeline

mask_filler = pipeline("fill-mask",
                       model='Livingwithmachines/pea-year-pol-loc')

preds = mask_filler("1861 [DATE] liberal [POL] london [LOC] [MASK] Majesty.")
```

## Data

### Frozen corpus

Main corpus, to be kept constant during all experiments. Saved in arrow format. To load the dataset use:

```python
train_val_split = load_from_disk('/datadrive_2/frozen_corpus')
```

`/datadrive_2/frozen_corpus`: The `frozen_corpus` was constructed by chunking the complete HMD into segments of hundred words, after which we donwsampled to approx half a billion tokens. The content should look as follows. 

```python
DatasetDict({
    train: Dataset({
        features: ['year', 'nlp', 'pol', 'loc', 'sentences', 'ocr', 'length'],
        num_rows: 5234550
    })
    test: Dataset({
        features: ['year', 'nlp', 'pol', 'loc', 'sentences', 'ocr', 'length'],
        num_rows: 581857
    })
})
```

The `sentences` columns contains the chunk of hundred words (should be renamed). `loc` indexes the place of publication as special token `[liverpool]` and `[london]`. `pol` contains the political leaning of the paper (`[conservative]`, `[liberal]` etc.)
`ocr` is `length` columns are derived from the original article to which the segment belonged and refer to the average OCR quality and respectively the number of tokens in the article. 

### HMD Enriched

`/datadrive_2/HMD_context`: original HMD corpus, but with political leaning and NLP identifier add to each example.

## Code

Code is currently stored in `/datadrive/bNERT` (should be pushed to GitHub after cleaning). Main Notebook is `PrepareDataset.ipynb` which includes a data loading, preprocessing and training code. This is still under construction and ~~should~~ will be cleaned later on.

# Depreciated


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

### erwt-year-st

`Livingwithmachines/erwt-year-st`: DistilBERT model trained on 0.5 billion tokens (see below for path to dataset). We divided the text into chunks of 100 words (i.e. no sentence splitting). 

To use, please preprocess text as follows:
```python
text = f"[1810] [SEP] [MASK] Majesty."
preds = mask_filler(text)
```
