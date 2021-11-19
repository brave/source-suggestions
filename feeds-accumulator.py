import sys
import os
import numpy as np
import scipy
import pandas as pd
import json
import requests
import time
import hashlib

def parse_articles_from_feed(articles):
    duplicate = 0

    for i, article in enumerate(articles):
        title = article['title'].replace('\r', '').replace('\n', '').replace('"', '')
        publisher_name = article['publisher_name']
        description = article['description'].replace('\r', '').replace('\n', '').replace('"', '')
        publish_time = article['publish_time']
        category = article['category']
        url = article['url']
        publisher_id = article['publisher_id']

        hash_title = hashlib.md5(title.encode()).hexdigest()

        if publisher_id not in checklist:
            with open("feed_buckets/{}.csv".format(publisher_id), "w") as myfile:
                myfile.write('"'+'","'.join([title, publisher_name, description, publish_time, category, url, publisher_id, hash_title])+'"\n')
                checklist[publisher_id] = [hash_title]
            with open("output/feeds.csv", "a") as myfile:
                myfile.write("{},{}\n".format(publisher_name, publisher_id))
        else:
            if hash_title not in checklist[publisher_id]:
                with open("feed_buckets/{}.csv".format(publisher_id), "a") as myfile:
                    myfile.write('"'+'","'.join([title, publisher_name, description, publish_time, category, url, publisher_id, hash_title])+'"\n')
                checklist[publisher_id].append(hash_title)
            else:
                duplicate += 1

    print("There have been {} duplicate articles across adjacent parses".format(duplicate))

checklist = {}

if os.path.isfile("output/feeds.csv"):
    print("Found feeds.csv file, initializing...")
    publisher_ids = list(pd.read_csv('output/feeds.csv', header=None).iloc[:, 1].to_numpy())
    for publisher_id in publisher_ids:
        hash_titles = list(pd.read_csv("feed_buckets/{}.csv".format(publisher_id), header=None).iloc[:, -1].to_numpy())
        checklist[publisher_id] = hash_titles

while True:
    print("Starting new parse...")
    r = requests.get('https://brave-today-cdn.brave.com/brave-today/feed.json')
    articles = r.json()
    parse_articles_from_feed(articles)
    print("Finished parsing articles from feed. Next parse in 3h hours.")
    time.sleep(3*60*60)
