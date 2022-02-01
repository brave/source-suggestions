import sys
import os
import numpy as np
import scipy
import pandas as pd

print("Started sanitizing...")
feeds_df = pd.read_csv('output/feeds.csv', header=None)
publisher_ids = feeds_df.iloc[:, 1].to_numpy()
for publisher_id in publisher_ids:
    feed_bucket_df = pd.read_csv("feed_buckets/{}.csv".format(publisher_id), header=None)
    feed_bucket_df = feed_bucket_df.drop_duplicates()
    feed_bucket_df.to_csv("feed_buckets/{}.csv".format(publisher_id), header=None)
print("Feed buckets sanitized.")
