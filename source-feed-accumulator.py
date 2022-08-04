import sys
import pandas as pd
import json
import requests
from tqdm import tqdm

def sanitize_articles_history():
    articles_history_df = pd.read_csv(output_dir + 'articles_history.csv')
    articles_history_df = articles_history_df.drop_duplicates().dropna()
    articles_history_df.to_csv(output_dir + 'articles_history.csv', index=False)

def accumulate_articles(articles):

    for i, article in tqdm(enumerate(articles)):
        title = article['title'].replace('\r', '').replace('\n', '').replace('"', '')
        description = article['description'].replace('\r', '').replace('\n', '').replace('"', '')
        publish_time = article['publish_time']
        publisher_id = article['publisher_id']

        with open(output_dir + "articles_history.csv", "a") as f:
            f.write('"'+'","'.join([title, description, publish_time, publisher_id])+'"\n')


feed_url = 'https://brave-today-cdn.bravesoftware.com/brave-today/feed.en_US.json'
output_dir = 'output/'

feed_r = requests.get(feed_url)
articles_json = feed_r.json()

print("Starting new parse...")
# TODO: PULL articles_history.csv from S3
accumulate_articles(articles_json)
print("Finished parsing articles from feed.")
print("Sanitizing articles_history...")
sanitize_articles_history()
# TODO: PUSH articles_history.csv to S3
print("Finished sanitizing articles history.")