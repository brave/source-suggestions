# brave-news-feed-suggestion

Pipeline for producing the source embedding representations needed in My Personalised Feed feature for Brave News (see https://docs.google.com/document/d/1t-XxP5ykelNyrkCVKE4_wtjIZzqPB7-6ZrYTq-KSg-Q/edit)

## Scripts

`feeds-accumulator.py` parses periodically Brave News's feed, creating buckets for each of the served sources and storing inside all articles originating from each source. These buckets are catalogued by `PUBLISHER_ID` under `output/feed_buckets/`. No duplicate articles should be found within any of the feed buckets. 

`feeds-similarity-matrix.py` takes in the source buckets and produces an 512-dimensional embedding for each source, built as the mean of the 512-dimensional embeddings of all articles belonging to the source, as generated by the Universal Sentence Encoder model (https://arxiv.org/abs/1803.11175). Once embeddings have been generated for all sources, a similarity matrix is built using cosine distance as similarity metric.

`feeds-sanitize.py` parses each source bucket in `output/feed_buckets/` and remove any duplicate articles, if any exist.

## Outputs

`feeds.csv` is a simple two-column table with all sources' names and publisher ids.
`publisher_name | publisher_id`

`feed_embeddings.csv` stores all the 512-dimensional embeddings for each source under its `publisher_id`.
`index | publisher_id | 0 | 1 ... | ... 511`

`sim_matrix.csv` stores the SxS similarity matrix, where S is the number of sources collected by the `feed-accumulator.py` script.

