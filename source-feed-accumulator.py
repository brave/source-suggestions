import json
import pathlib

import pandas as pd
from structlog import get_logger
from tqdm import tqdm

import config
from utils import download_file, upload_file

logger = get_logger()


def sanitize_articles_history(lang_region):
    articles_history_df = pd.read_csv(
        config.OUTPUT_DIR + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region)
    )
    articles_history_df = articles_history_df.drop_duplicates().dropna()
    articles_history_df.to_csv(
        config.OUTPUT_DIR + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region),
        index=False,
    )


def accumulate_articles(articles, lang_region):
    for i, article in tqdm(enumerate(articles)):
        title = article["title"].replace("\r", "").replace("\n", "").replace('"', "")
        description = (
            article["description"].replace("\r", "").replace("\n", "").replace('"', "")
        )
        publish_time = article["publish_time"]
        publisher_id = article["publisher_id"]

        with open(
            config.OUTPUT_DIR
            + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region),
            "a",
        ) as f:
            f.write(
                '"'
                + '","'.join([title, description, publish_time, publisher_id])
                + '"\n'
            )


for lang_region, model in config.LANG_REGION_MODEL_MAP:
    logger.info(f"Starting feeds accumulator for {lang_region}")

    feed_file = f"{config.FEED_JSON_FILE.format(LANG_REGION=lang_region)}.json"

    pathlib.Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not config.NO_DOWNLOAD:
        download_file(feed_file, config.PUB_S3_BUCKET, f"brave-today/{feed_file}")
        download_file(
            config.OUTPUT_DIR
            + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region),
            config.PUB_S3_BUCKET,
            f"source-suggestions/{config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region)}",
        )

    with open(feed_file) as feeds:
        feeds_data = json.loads(feeds.read())

    accumulate_articles(feeds_data, lang_region)
    logger.info("Finished feeds accumulator")

    logger.info("Start sanitizing articles_history...")
    sanitize_articles_history(lang_region)

    if not config.NO_UPLOAD:
        upload_file(
            config.OUTPUT_DIR
            + config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region),
            config.PUB_S3_BUCKET,
            f"source-suggestions/{config.ARTICLE_HISTORY_FILE.format(LANG_REGION=lang_region)}",
        )

    logger.info("Finished sanitizing articles_history.")
