# Word Embeddings: From Bag-of-Words to Word2Vec

A hands-on Jupyter notebook (`wordembeddings.ipynb`) that walks through the core techniques for turning text into numbers, using a short, playful story ("Jack and the NLP Magic Beans") as the corpus. Each technique in the story is implemented in code.

## What's Inside

| Section | Technique | Library |
|---|---|---|
| 1 | **Bag of Words**: text cleaning, sentence tokenization, word frequencies, top-N vocabulary, binary document-term matrix with heatmap | `nltk`, `re`, `heapq`, `seaborn` |
| 2 | **One-Hot Encoding**: encoding the top words as sparse binary vectors | `scikit-learn` (`OneHotEncoder`) |
| 3 | **TF-IDF**: weighting words by importance across sentences | `scikit-learn` (`TfidfVectorizer`) |
| 4 | **Word2Vec**: training dense 4-dimensional embeddings and exploring them | `gensim` |

### Word2Vec Experiments

- **Analogy**: `jack - coder + computer`
- **Similarity**: nearest neighbours of `bean`, cosine similarity/distance, and phrase-level similarity (`n_similarity`)
- **Odd-one-out**: `doesnt_match` on a set of words
- **Vocabulary inspection**: word index, vocabulary size, raw counts, and raw word vectors

> **Note:** The corpus is only five sentences, so the Word2Vec results are illustrative rather than semantically meaningful. For real embeddings, train on a much larger corpus or use pretrained vectors (e.g., GloVe, fastText).

## Getting Started

### 1. Clone and install

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download NLTK data

The notebook uses sentence/word tokenizers and English stopwords. Run once:

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### 3. Launch the notebook

```bash
jupyter notebook wordembeddings.ipynb
```

## Project Structure

```
.
├── wordembeddings.ipynb   # Main notebook
├── requirements.txt       # Python dependencies
└── README.md
```

## Customizing

- Replace the `text` variable in the first code cells with your own corpus.
- Change `heapq.nlargest(10, ...)` to adjust the Bag-of-Words vocabulary size.
- Tune `Word2Vec(vector_size=..., window=..., min_count=...)` for different embedding sizes and context windows.

## Requirements

Python 3.9+ and the packages listed in `requirements.txt`.
