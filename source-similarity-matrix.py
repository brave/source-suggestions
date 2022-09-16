import json
import math
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from structlog import get_logger
from tqdm import tqdm

import config
from utils import download_file, upload_file

logger = get_logger()


# Take centroid of 512-d embeddings
def get_source_representation_from_titles(titles):
    source_repr = np.zeros((1, 512))
    if len(titles) < config.MINIMUM_ARTICLE_HISTORY_SIZE:
        return source_repr

    for title in titles:
        source_repr += embed([title])[0]
    norm_repr = tf.nn.l2_normalize(source_repr / len(titles), axis=1)
    return norm_repr.numpy()


def compute_source_similarity(source1, source2, t='dot'):
    cosine_similarities = np.dot(source1, np.transpose(source2))
    clip_cosine_similarity = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    score = 1.0 - tf.acos(clip_cosine_similarity) / math.pi
    return score


def compute_source_representation_from_articles(articles_df, publisher_id):
    publisher_bucket_df = articles_df[articles_df.publisher_id == publisher_id]

    source_titles = [title for title in publisher_bucket_df.title.to_numpy() if title is not None]
    return get_source_representation_from_titles(source_titles)


logger.info("Started computing similarity matrix...")

if not config.NO_DOWNLOAD:
    download_file(config.OUTPUT_DIR + config.ARTICLE_HISTORY_FILE, config.PUB_S3_BUCKET,
                  f"source-suggestions/{config.ARTICLE_HISTORY_FILE}")

pathlib.Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

sources_file = f'{config.SOURCES_JSON_FILE}.json'

if not config.NO_DOWNLOAD:
    download_file(sources_file, config.PUB_S3_BUCKET, sources_file)

with open(sources_file) as sources:
    sources_data = json.loads(sources.read())

sources_df = pd.json_normalize(sources_data)
sources_df["source_representation"] = np.nan

articles_df = pd.read_csv(config.OUTPUT_DIR + config.ARTICLE_HISTORY_FILE, header=None)
articles_df.columns = ['title', 'description', 'timestamp', 'publisher_id']

logger.info("Loading Universal Sentence Encoder...")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
logger.info(f"USE ({module_url}) loaded")


def embed(input):
    return model(input)


logger.info("Building sources embeddings...")
publisher_ids = sources_df.publisher_id.to_numpy()
logger.info(f"Publisher ids size: {publisher_ids.size}")
reprs = np.zeros((publisher_ids.size, 512))
for i, publisher_id in tqdm(enumerate(publisher_ids)):
    reprs[i, :] = compute_source_representation_from_articles(articles_df, publisher_id)

sources_representation = pd.DataFrame({'publisher_id': publisher_ids})
sources_representation = pd.concat([sources_representation, pd.DataFrame(reprs)], axis=1)
sources_representation.to_csv(f'output/{config.SOURCE_EMBEDDINGS}.csv', header=None)

logger.info("Finished building source embeddings.")

sim_matrix = np.zeros((publisher_ids.size, publisher_ids.size))
for i in range(publisher_ids.size):
    for j in range(i + 1, publisher_ids.size):
        repr_i = reprs[i]
        repr_j = reprs[j]
        sim = compute_source_similarity(repr_i, repr_j)
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim


def get_source_id_for_title(title, sources_df):
    return sources_df[sources_df.publisher_name == title].publisher_id.to_numpy()[0]


logger.info("Finished computing similarity matrix. Outputting results...")

publisher_titles = sources_df.publisher_name.to_numpy()
logger.info(f"Publisher titles size: {publisher_titles.size}")
top10_dictionary = {}
top10_dictionary_human_readable = {}
for i, feed in enumerate(publisher_titles):
    sources_ranking = []
    source_id = get_source_id_for_title(feed, sources_df)

    for j in range(sim_matrix.shape[0]):
        if i == j or math.isnan(sim_matrix[i, j]) or sim_matrix[i,j] == 0:
            continue
        sources_ranking.append((publisher_titles[j], sim_matrix[i, j]))

    sources_ranking.sort(key=lambda x: -x[1])

    top10_dictionary[source_id] = [{'source': get_source_id_for_title(source[0], sources_df), 'score': source[1]} for
                                   source in sources_ranking[:10]]
    top10_dictionary_human_readable[feed] = [{'source': source[0], 'score': source[1]} for source in
                                             sources_ranking[:10]]

with open(f'output/{config.SOURCE_SIMILARITY_T10}.json', 'w', encoding='utf-8') as f:
    json.dump(top10_dictionary, f, ensure_ascii=True, indent=4)

with open(f'output/{config.SOURCE_SIMILARITY_T10_HR}.json', 'w', encoding='utf-8') as f:
    json.dump(top10_dictionary_human_readable, f, ensure_ascii=True, indent=4)

logger.info("Script has finished running.")

if not config.NO_UPLOAD:
    upload_file(config.OUTPUT_DIR + f'/{config.SOURCE_SIMILARITY_T10}.json', config.PUB_S3_BUCKET,
                f"source-suggestions/{config.SOURCE_SIMILARITY_T10}.json")

    upload_file(config.OUTPUT_DIR + f'/{config.SOURCE_SIMILARITY_T10_HR}.json', config.PUB_S3_BUCKET,
                f"source-suggestions/{config.SOURCE_SIMILARITY_T10_HR}.json")

    upload_file(config.OUTPUT_DIR + f'/{config.SOURCE_EMBEDDINGS}.csv', config.PUB_S3_BUCKET,
                f"source-suggestions/{config.SOURCE_EMBEDDINGS}.csv")
