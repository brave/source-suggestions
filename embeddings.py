import numpy as np
from sentence_transformers import util
from structlog import get_logger

import config

EMBEDDING_DIMENSIONALITY = 384

logger = get_logger()


def compute_source_similarity(source_1, source_2, function='cosine'):
    if function == 'dot':
        return util.dot_score(source_1, np.transpose(source_2))
    elif function == 'cosine':
        return util.pytorch_cos_sim(source_1, source_2)[0][0]


def get_source_representation_from_titles(titles, model):
    if len(titles) < config.MINIMUM_ARTICLE_HISTORY_SIZE:
       return np.zeros((1, EMBEDDING_DIMENSIONALITY))

    return model.encode(titles).mean(axis=0)


def compute_source_representation_from_articles(articles_df, publisher_id, model):
    publisher_bucket_df = articles_df[articles_df.publisher_id == publisher_id]

    titles = [
        title for title in publisher_bucket_df.title.to_numpy() if title is not None]
    return get_source_representation_from_titles(titles, model)
