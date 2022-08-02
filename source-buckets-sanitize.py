import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("Started sanitizing...")
sources_df = pd.read_csv('output/sources.csv')
publisher_names = sources_df.Title.to_numpy()
for publisher in tqdm(publisher_names):
    publisher_id = publisher.lower().replace('-',' ').replace(' ','_')
    my_file = Path("source_buckets/{}.csv".format(publisher_id))
    if not my_file.is_file():
        continue

    try:
        feed_bucket_df = pd.read_csv("source_buckets/{}.csv".format(publisher_id), header=None)
    except:
        print("Error on {} read".format(publisher_id))
        continue
    feed_bucket_df = feed_bucket_df.drop_duplicates().dropna()
    feed_bucket_df.to_csv("source_buckets/{}.csv".format(publisher_id), header=None, index=False)
print("Source buckets sanitized.")
