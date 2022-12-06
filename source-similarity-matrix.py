import json
import math
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from structlog import get_logger
from tqdm import tqdm

import config
from utils import download_file, upload_file

logger = get_logger()


def clean_source_similarity_file(sources_data, sources_sim_data):
    sources_id = [sources.get("publisher_id") for sources in sources_data]

    for s_id in sources_id:
        if s_id not in sources_sim_data:
            sources_sim_data.pop(s_id, None)
            continue

        if s_id in sources_sim_data:
            for index, suggestion in enumerate(sources_sim_data[s_id]):
                if suggestion["source"] not in sources_id:
                    sources_sim_data[s_id].pop(index)

    return sources_sim_data


def embed(input):
    return model(input)


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


def get_source_id_for_title(title, sources_df):
    return sources_df[sources_df.publisher_name == title].publisher_id.to_numpy()[0]


# Compute similarity matrix for all existing LANG_REGION pairs
for lang_region, model_url in config.LANG_REGION_MODEL_MAP:
    logger.info(f"Started computing similarity matrix for {lang_region} using {model_url}")

    pathlib.Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not config.NO_DOWNLOAD:
        download_file(config.OUTPUT_DIR + "/" + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region),
                      config.PUB_S3_BUCKET,
                      f"source-suggestions/{config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region)}")

    sources_file = f'{config.SOURCES_JSON_FILE.format(LANG_REGION=lang_region)}.json'

    if not config.NO_DOWNLOAD:
        download_file(sources_file, config.PUB_S3_BUCKET, sources_file)

    with open(sources_file) as sources:
        sources_data = json.loads(sources.read())

    sources_df = pd.json_normalize(sources_data)
    sources_df["source_representation"] = np.nan

    articles_df = pd.read_csv(config.OUTPUT_DIR + '/' + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region),
                              header=None)
    articles_df.columns = ['title', 'description', 'timestamp', 'publisher_id']

    logger.info("Loading Universal Sentence Encoder...")
    module_url = model_url  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)
    logger.info(f"USE ({module_url}) loaded")

    logger.info("Building sources embeddings...")
    publisher_ids = sources_df.publisher_id.to_numpy()
    logger.info(f"Publisher ids size: {publisher_ids.size}")

    # For each publisher, compute source representations from all stored
    # articles for that publisher.
    reprs = np.zeros((publisher_ids.size, 512))
    for i, publisher_id in tqdm(enumerate(publisher_ids)):
        reprs[i, :] = compute_source_representation_from_articles(articles_df, publisher_id)

    logger.info(f"Computing sources representations for {lang_region}")
    sources_representation = pd.DataFrame({'publisher_id': publisher_ids})
    sources_representation = pd.concat([sources_representation, pd.DataFrame(reprs)], axis=1)
    sources_representation.to_csv(
        f'output/{config.SOURCE_EMBEDDINGS.format(LANG_REGION=lang_region)}.csv', header=None)
    logger.info("Finished building source embeddings.")

    # For each source pair, compute pair similarity
    sim_matrix = np.zeros((publisher_ids.size, publisher_ids.size))
    for i in range(publisher_ids.size):
        for j in range(i + 1, publisher_ids.size):
            repr_i = reprs[i]
            repr_j = reprs[j]
            sim = compute_source_similarity(repr_i, repr_j)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    logger.info("Finished computing similarity matrix. Outputting results...")

    # Produce T10 (top10) output files. T10_HR stands for Human Readable and
    # exists for debugging purposes.
    publisher_titles = sources_df.publisher_name.to_numpy()
    logger.info(f"Publisher titles size: {publisher_titles.size}")
    top10_dictionary = {}
    top10_dictionary_human_readable = {}
    for i, feed in enumerate(publisher_titles):
        sources_ranking = []
        source_id = get_source_id_for_title(feed, sources_df)

        for j in range(sim_matrix.shape[0]):
            if i == j or math.isnan(sim_matrix[i, j]) or sim_matrix[i, j] == 0.5:
                continue
            sources_ranking.append((publisher_titles[j], sim_matrix[i, j]))

        # Sort sources by descending similarity score
        sources_ranking.sort(key=lambda x: -x[1])

        # Only include suggestion if within 10% of the best match's score
        top_similarity_score = 0.0
        if sources_ranking:
            top_similarity_score = sources_ranking[0][1]
        similarity_cutoff = config.SIMILARITY_CUTOFF_RATIO * top_similarity_score
        top10_dictionary[source_id] = [{'source': get_source_id_for_title(source[0], sources_df), 'score': source[1]}
                                       for
                                       source in sources_ranking[:10] if source[1] > similarity_cutoff]
        top10_dictionary_human_readable[feed] = [{'source': source[0], 'score': source[1]} for source in
                                                 sources_ranking[:10] if source[1] > similarity_cutoff]

    logger.info("Removing un-matched sources")
    top10_dictionary = clean_source_similarity_file(sources_data, top10_dictionary)

    logger.info("Outputting sources similarities files")
    with open(f'output/{config.SOURCE_SIMILARITY_T10.format(LANG_REGION=lang_region)}.json', 'w') as f:
        json.dump(top10_dictionary, f)
    with open(f'output/{config.SOURCE_SIMILARITY_T10_HR.format(LANG_REGION=lang_region)}.json', 'w') as f:
        json.dump(top10_dictionary_human_readable, f)

    logger.info("Script has finished running.")

    if not config.NO_UPLOAD:
        upload_file(config.OUTPUT_DIR + "/" + f'/{config.SOURCE_SIMILARITY_T10.format(LANG_REGION=lang_region)}.json',
                    config.PUB_S3_BUCKET,
                    f"source-suggestions/{config.SOURCE_SIMILARITY_T10.format(LANG_REGION=lang_region)}.json")

        upload_file(
            config.OUTPUT_DIR + "/" + f'/{config.SOURCE_SIMILARITY_T10_HR.format(LANG_REGION=lang_region)}.json',
            config.PUB_S3_BUCKET,
            f"source-suggestions/{config.SOURCE_SIMILARITY_T10_HR.format(LANG_REGION=lang_region)}.json")

        upload_file(config.OUTPUT_DIR + "/" + f'/{config.SOURCE_EMBEDDINGS.format(LANG_REGION=lang_region)}.csv',
                    config.PUB_S3_BUCKET,
                    f"source-suggestions/{config.SOURCE_EMBEDDINGS.format(LANG_REGION=lang_region)}.csv")
