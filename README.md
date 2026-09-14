# brave-news-source-suggestion

Service for producing the source embedding representations and similarity matrix needed for source suggestion feature in Brave News.

## Installation

```
pip install -r requirements.txt
```

## Scripts
**source-feed-accumulator.py**: parses Brave News feed periodically, collecting articles for each source in `articles_history.csv`. For each article, we store the `publisher_id` attribute.

**sources-similarity-matrix.py**: takes as input the article history and produces a 384-dimensional embedding for each source, using the `sentence-transformer` package. More in particular:
- `all-MiniLM-L6-v2` for english language sources.
- `paraphrase-multilingual-MiniLM-L12-v2` for non-english language sources.
Once all source embeddings are generated, a pairwise source similarity matrix is produced.



## Description
There are two jobs involved in the generation of source suggestions. You can list them in EKS `source-suggestions-prod`.


- **feed-accumulator** is a job that runs hourly that will fetch the feed.json for each locale then accumulate them all into a csv file then write back to S3. This output is available here https://brave-today-cdn.brave.com/source-suggestions/articles_history.en_US.csv and here. The articles_history file is only used by the backend job source-sim-matrix, the client does not use it.


- **source-sim-matrix** is the other job, runs twice a week which will pull the articles_history csv and the publishers json from S3 then perform clustering on the article text and produce the source-suggestions json for each locale:
  - https://brave-today-cdn.brave.com/source-suggestions/source_similarity_t10.en_US.json
  - https://brave-today-cdn.brave.com/source-suggestions/source_similarity_t10_hr.en_US.json.

Non English locales use a multilingual clustering model.  The browser will use this file to then determine which publishers to show in the suggested publisher cards in the feed, about every 7-8 cards you will see the suggestions.


## Running locally
To collect and accumulate article history:

Run this to download the files needed to run the script locally:
```sh
make create-local-env
```

```sh
NO_UPLOAD=1 NO_DOWNLOAD=1 python source-feed-accumulator.py
```

To computed source embeddings and produce the source similarity matrix:
```sh
NO_UPLOAD=1 NO_DOWNLOAD=1 python source-similarity-matrix.py
```
