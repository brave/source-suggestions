import json
import math
import pathlib

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from structlog import get_logger
from tqdm import tqdm

import config
from embeddings import (
    EMBEDDING_DIMENSIONALITY,
    compute_source_representation_from_articles,
    compute_source_similarity,
)
from utils import (
    clean_source_similarity_file,
    download_file,
    get_source_id_for_title,
    upload_file,
)

logger = get_logger()


# Compute similarity matrix for all existing LANG_REGION pairs
for lang_region, model_name in config.LANG_REGION_MODEL_MAP:  # noqa: C901
    logger.info(
        f"Started computing similarity matrix for {lang_region} using {model_name}"
    )

    pathlib.Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not config.NO_DOWNLOAD:
        download_file(
            config.OUTPUT_DIR
            + "/"
            + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region),
            config.PUB_S3_BUCKET,
            f"source-suggestions/{config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region)}",
        )

    sources_file = f"{config.SOURCES_JSON_FILE.format(LANG_REGION=lang_region)}.json"

    if not config.NO_DOWNLOAD:
        download_file(sources_file, config.PUB_S3_BUCKET, sources_file)

    with open(sources_file) as sources:
        sources_data = json.loads(sources.read())

    sources_df = pd.json_normalize(sources_data)
    sources_df["source_representation"] = np.nan

    articles_df = pd.read_csv(
        config.OUTPUT_DIR
        + "/"
        + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region),
        header=None,
    )
    articles_df.columns = ["title", "description", "timestamp", "publisher_id"]

    logger.info("Loading Embedding Model...")
    model = SentenceTransformer(model_name)
    logger.info(f"Model ({model_name}) loaded")

    logger.info("Building sources embeddings...")
    publisher_ids = sources_df.publisher_id.to_numpy()
    logger.info(f"Publisher ids size: {publisher_ids.size}")

    # For each publisher, compute source representations from all stored
    # articles for that publisher.
    reprs = np.zeros((publisher_ids.size, EMBEDDING_DIMENSIONALITY))
    for i, publisher_id in tqdm(enumerate(publisher_ids)):
        reprs[i, :] = compute_source_representation_from_articles(
            articles_df, publisher_id, model
        )
        if not reprs[i, :].any():
            logger.warning(
                f"Source {sources_df[sources_df.publisher_id == publisher_id].publisher_name.item()} "
                f"has no articles. Skipping..."
            )

    logger.info(f"Computing sources representations for {lang_region}")
    sources_representation = pd.DataFrame({"publisher_id": publisher_ids})
    sources_representation = pd.concat(
        [sources_representation, pd.DataFrame(reprs)], axis=1
    )
    sources_representation.to_csv(
        f"output/{config.SOURCE_EMBEDDINGS.format(LANG_REGION=lang_region)}.csv",
        header=None,
    )
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
        top10_dictionary[source_id] = [
            {
                "source": get_source_id_for_title(source[0], sources_df),
                "score": source[1],
            }
            for source in sources_ranking[:10]
            if source[1] > similarity_cutoff
        ]
        top10_dictionary_human_readable[feed] = [
            {"source": source[0], "score": source[1]}
            for source in sources_ranking[:10]
            if source[1] > similarity_cutoff
        ]

    logger.info("Removing un-matched sources")
    top10_dictionary = clean_source_similarity_file(sources_data, top10_dictionary)

    logger.info("Outputting sources similarities files")
    with open(
        f"output/{config.SOURCE_SIMILARITY_T10.format(LANG_REGION=lang_region)}.json",
        "w",
    ) as f:
        json.dump(top10_dictionary, f)
    with open(
        f"output/{config.SOURCE_SIMILARITY_T10_HR.format(LANG_REGION=lang_region)}.json",
        "w",
    ) as f:
        json.dump(top10_dictionary_human_readable, f)

    logger.info("Script has finished running.")

    if not config.NO_UPLOAD:
        upload_file(
            config.OUTPUT_DIR
            + "/"
            + f"/{config.SOURCE_SIMILARITY_T10.format(LANG_REGION=lang_region)}.json",
            config.PUB_S3_BUCKET,
            f"source-suggestions/{config.SOURCE_SIMILARITY_T10.format(LANG_REGION=lang_region)}.json",
        )

        upload_file(
            config.OUTPUT_DIR
            + "/"
            + f"/{config.SOURCE_SIMILARITY_T10_HR.format(LANG_REGION=lang_region)}.json",
            config.PUB_S3_BUCKET,
            f"source-suggestions/{config.SOURCE_SIMILARITY_T10_HR.format(LANG_REGION=lang_region)}.json",
        )

        upload_file(
            config.OUTPUT_DIR
            + "/"
            + f"/{config.SOURCE_EMBEDDINGS.format(LANG_REGION=lang_region)}.csv",
            config.PUB_S3_BUCKET,
            f"source-suggestions/{config.SOURCE_EMBEDDINGS.format(LANG_REGION=lang_region)}.csv",
        )
