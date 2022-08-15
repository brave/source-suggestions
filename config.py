import os

# Canonical ID of the public S3 bucket
BRAVE_TODAY_CANONICAL_ID = os.getenv('BRAVE_TODAY_CANONICAL_ID', None)
BRAVE_TODAY_CLOUDFRONT_CANONICAL_ID = os.getenv('BRAVE_TODAY_CLOUDFRONT_CANONICAL_ID', None)

# Set to INFO to see some output during long-running steps.
LOG_LEVEL = os.getenv('LOG_LEVEL', 'WARNING')

# Disable uploads to S3. Useful when running locally or in CI.
NO_UPLOAD = os.getenv('NO_UPLOAD', None)
NO_DOWNLOAD = os.getenv('NO_DOWNLOAD', None)

PCDN_URL_BASE = os.getenv('PCDN_URL_BASE', 'https://pcdn.brave.software')
# Canonical ID of the private S3 bucket
PRIVATE_CDN_CANONICAL_ID = os.getenv('PRIVATE_CDN_CANONICAL_ID', None)
PRIVATE_CDN_CLOUDFRONT_CANONICAL_ID = os.getenv('PRIVATE_CDN_CLOUDFRONT_CANONICAL_ID', None)
PRIV_S3_BUCKET = os.getenv('PRIV_S3_BUCKET', 'brave-private-cdn-development')
PUB_S3_BUCKET = os.getenv('PUB_S3_BUCKET', 'brave-today-cdn-development')


SOURCES_JSON_FILE = os.getenv('SOURCES_JSON_FILE', 'sources.en_US.json')
FEED_JSON_FILE = os.getenv('FEED_JSON_FILE', 'feed.en_US')

OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
ARTICLE_HISTORY_FILE = os.getenv('ARTICLE_HISTORY_FILE', "articles_history.en_US.csv")
