# brave-news-source-suggestion

Service for producing the source embedding representations and similarity matrix needed for source suggestion feature in Brave News.

## Installation

```
pip install -r requirements.txt
```

## Scripts
**source-feed-accumulator.py**: parses Brave News feed periodically, collecting articles for each source in `articles_history.csv`. For each article, we store the `publisher_id` attribute.

**sources-similarity-matrix.py**: takes as input the article history and produces a 512-dimensional embedding for each source, using the Universal Sentence Encoder model (https://arxiv.org/abs/1803.11175). Once an embedding is computed for each source, a source similarity matrix is produced.

## Running locally
To collect and accumulate article history:
```
export NO_UPLOAD=1
export NO_DOWNLOAD=1
python source-feed-accumulator.py
```

To computed source embeddings and produce the source similarity matrix:
```
export NO_UPLOAD=1
export NO_DOWNLOAD=1
python sources-similarity-matrix.py
```
