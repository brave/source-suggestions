import os

# Disable uploads to S3. Useful when running locally or in CI.
NO_UPLOAD = os.getenv('NO_UPLOAD', None)
NO_DOWNLOAD = os.getenv('NO_DOWNLOAD', None)

PCDN_URL_BASE = os.getenv('PCDN_URL_BASE', 'https://pcdn.brave.software')
PUB_S3_BUCKET = os.getenv('PUB_S3_BUCKET', 'brave-today-cdn-development')
# Canonical ID of the public S3 bucket
BRAVE_TODAY_CANONICAL_ID = os.getenv('BRAVE_TODAY_CANONICAL_ID', None)
BRAVE_TODAY_CLOUDFRONT_CANONICAL_ID = os.getenv('BRAVE_TODAY_CLOUDFRONT_CANONICAL_ID', None)

LANG_REGION_MODEL_MAP = os.getenv('LANG_REGION_MODEL_MAP', [
    ('en_US', "https://tfhub.dev/google/universal-sentence-encoder/4"),
    ('en_CA', "https://tfhub.dev/google/universal-sentence-encoder/4"),
    ('es_ES', 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'),
    ('es_MX', 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'),
    ('pt_BR', 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'),
    ('ja_JP', 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'),
])

SOURCES_JSON_FILE = os.getenv('SOURCES_JSON_FILE', 'sources.{LANG_REGION}')
FEED_JSON_FILE = os.getenv('FEED_JSON_FILE', 'feed.{LANG_REGION}')

OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

ARTICLE_HISTORY_FILE = os.getenv('ARTICLE_HISTORY_FILE', "articles_history.{LANG_REGION}.csv")
# Don't compute the embedding for a source that has less than 30 collected articles
MINIMUM_ARTICLE_HISTORY_SIZE = os.getenv('MINIMUM_ARTICLE_HISTORY_SIZE', 30)
SIMILARITY_CUTOFF_RATIO = os.getenv('SIMILARITY_CUTOFF_RATIO', 0.9)
SOURCE_SIMILARITY_T10 = os.getenv('SOURCE_SIMILARITY_T10', "source_similarity_t10.{LANG_REGION}")
SOURCE_SIMILARITY_T10_HR = os.getenv('SOURCE_SIMILARITY_T10_HR', "source_similarity_t10_hr.{LANG_REGION}")

SOURCE_EMBEDDINGS = os.getenv('SOURCE_EMBEDDINGS', "SOURCE_EMBEDDINGS.{LANG_REGION}")
