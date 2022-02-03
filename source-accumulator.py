import sys
import os
import numpy as np
import scipy
import pandas as pd
import json
import requests
import time
import hashlib

def parse_articles_from_source(articles):
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
            with open("source_buckets/{}.csv".format(publisher_id), "w") as myfile:
                myfile.write('"'+'","'.join([title, publisher_name, description, publish_time, category, url, publisher_id, hash_title])+'"\n')
                checklist[publisher_id] = [hash_title]
            with open("output/sources.csv", "a") as myfile:
                myfile.write("{},{}\n".format(publisher_name, publisher_id))
        else:
            if hash_title not in checklist[publisher_id]:
                with open("source_buckets/{}.csv".format(publisher_id), "a") as myfile:
                    myfile.write('"'+'","'.join([title, publisher_name, description, publish_time, category, url, publisher_id, hash_title])+'"\n')
                checklist[publisher_id].append(hash_title)
            else:
                duplicate += 1

    print("There have been {} duplicate articles across adjacent parses".format(duplicate))

checklist = {}

# on script start, check if sources.csv and source_buckets have already been created
# this recovers the script state in case of interruption/termination
if os.path.isfile("output/sources.csv"):
    print("Found sources.csv file, initializing...")
    publisher_ids = list(pd.read_csv('output/sources.csv', header=None).iloc[:, 1].to_numpy())
    for publisher_id in publisher_ids:
        hash_titles = list(pd.read_csv("source_buckets/{}.csv".format(publisher_id), header=None).iloc[:, -1].to_numpy())
        checklist[publisher_id] = hash_titles

print("Starting new parse...")
r = requests.get('https://brave-today-cdn.brave.com/brave-today/feed.json')
articles = r.json()
# TODO: PULL feed.csv,feed_buckets from S3
parse_articles_from_source(articles)
# TODO: PUSH source.csv,source_buckets changes to S3
print("Finished parsing articles from feed. Next parse in 3h hours.")
