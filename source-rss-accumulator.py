import sys
import io
import pandas as pd
import feedparser
import json
import requests
import time
import hashlib
from tqdm import tqdm

source_url = 'https://raw.githubusercontent.com/brave/news-aggregator/master/sources.en_US.csv'
output_dir = 'output/'

r = requests.get(source_url).text
with open(output_dir + 'sources.csv', 'w') as f:
    f.write(r)

sources_df = pd.read_csv(output_dir + 'sources.csv')
sources_df['source_id'] = sources_df.apply(lambda x: hashlib.sha256(x.Feed.encode('utf-8')).hexdigest(), axis=1)
sources_df.to_csv(output_dir + 'sources.csv', index=False)

for i, row in tqdm(sources_df.iterrows()):
    try:
        resp = requests.get(row.Feed, timeout=30.0)
    except requests.ReadTimeout:
        print("Timeout when reading RSS {} at {}".format(row.Title, row.Feed))
        try:
            resp = requests.get(row.Feed, timeout=30.0)
        except:
            print("Failed even after retry.")
            with open("source_buckets/{}.csv".format(publisher_title), "a") as f:
                pass
            continue
    except requests.exceptions.ConnectionError as e:
        print("Connection error for {} at {}".format(row.Title, row.Feed))
        try:
            resp = requests.get(row.Feed.replace('https','http'), timeout=30.0)
        except:
            print("Failed even after retry.")
            with open("source_buckets/{}.csv".format(publisher_title), "a") as f:
                pass
            continue
    except:
        print("General error with {} at {}".format(row.Title, row.Feed))
        with open("source_buckets/{}.csv".format(publisher_title), "a") as f:
                pass
        continue

    # Put it to memory stream object universal feedparser
    content = io.BytesIO(resp.content)

    # Parse content
    source_feed = feedparser.parse(content)
    publisher_title = row.Title.lower().replace('-',' ').replace(' ','_')
    for entry in source_feed['entries']:
        article_title = entry['title']
        article_link = entry['link']
        article_hash = hashlib.md5(article_title.encode()).hexdigest()

        with open("source_buckets/{}.csv".format(publisher_title), "a") as f:
            f.write('"'+'","'.join([article_title.replace('"',''), article_link, article_hash])+'"\n')


