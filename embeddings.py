import numpy as np
from sentence_transformers import util
from structlog import get_logger
import time  # Add import for timing

import config

EMBEDDING_DIMENSIONALITY = 384

logger = get_logger()


def compute_source_similarity(source_1, source_2, function='cosine'):
    if function == 'dot':
        return util.dot_score(source_1, np.transpose(source_2))
    elif function == 'cosine':
        return util.pytorch_cos_sim(source_1, source_2)[0][0]


def get_source_representation_from_titles(titles, model):
    num_titles = len(titles)
    logger.info("get_source_representation_from_titles called", num_titles=num_titles)

    if num_titles < config.MINIMUM_ARTICLE_HISTORY_SIZE:
       logger.warn(
           "Not enough titles for source representation",
           num_titles=num_titles,
           min_required=config.MINIMUM_ARTICLE_HISTORY_SIZE
       )
       return np.zeros((1, EMBEDDING_DIMENSIONALITY))

    start_time = time.time()
    embeddings = model.encode(titles)
    end_time = time.time()
    logger.info(
        "Model encoding finished",
        num_titles=num_titles,
        duration_sec=round(end_time - start_time, 3)
    )

    return embeddings.mean(axis=0)


def compute_source_representation_from_articles(articles_df, publisher_id, model):
    logger.info(
        "compute_source_representation_from_articles called",
        publisher_id=publisher_id,
        dataframe_shape=articles_df.shape
    )

    start_time = time.time()
    publisher_bucket_df = articles_df[articles_df.publisher_id == publisher_id]
    end_time = time.time()
    logger.info(
        "DataFrame filtering finished",
        publisher_id=publisher_id,
        duration_sec=round(end_time - start_time, 3),
        filtered_shape=publisher_bucket_df.shape
    )

    titles = [
        title for title in publisher_bucket_df.title.to_numpy() if title is not None]
    # Pass the model to the helper function for encoding
    return get_source_representation_from_titles(titles, model)
