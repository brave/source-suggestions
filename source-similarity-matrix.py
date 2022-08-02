import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import math
from pathlib import Path
import json

# Take centroid of 512-d embeddings
def source_representation(source):
    source_repr = np.zeros((1,512))
    for item in source:
        source_repr += embed([item])[0]
    norm_repr = tf.nn.l2_normalize(source_repr/len(source), axis=1)
    return norm_repr

def compute_source_similarity(source1, source2, t='dot'):
    cosine_similarities = np.dot(source1, np.transpose(source2))
    clip_cosine_similarity = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    score = 1.0 - tf.acos(clip_cosine_similarity) / math.pi
    return score


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
    return model(input)

# TODO(): PULL source.csv and source_buckets/

print("Started computing similarity matrix...")
sources_df = pd.read_csv('output/sources.csv')
sources_df["source_representation"] = np.nan
publisher_titles = sources_df.Title.to_numpy()
reprs = np.zeros((publisher_titles.size, 512))
for i, publisher_title in tqdm(enumerate(publisher_titles)):
    publisher_id = publisher_title.lower().replace('-',' ').replace(' ','_')
    my_file = Path("source_buckets/{}.csv".format(publisher_id))
    if not my_file.is_file():
        continue
    try:
        source_bucket_df = pd.read_csv("source_buckets/{}.csv".format(publisher_id), header=None)
    except:
        print("Error on {} read".format(publisher_id))
        continue
    source_name = publisher_title
    source_titles = [title for title in source_bucket_df.iloc[:,0].to_numpy() if title != None]
    source_repr = source_representation(source_titles).numpy()
    reprs[i,:] = source_repr
sources_representation = pd.DataFrame({'publisher':publisher_titles})
sources_representation =  pd.concat([sources_representation, pd.DataFrame(reprs)], axis=1)
sources_representation.to_csv('output/source_embeddings.csv', header=None)

sim_matrix = np.zeros((publisher_titles.size, publisher_titles.size))
for i in range(publisher_titles.size):
    for j in range(i+1, publisher_titles.size):
        repr_i = reprs[i]
        repr_j = reprs[j]
        sim = compute_source_similarity(repr_i, repr_j)
        sim_matrix[i,j] = sim
        sim_matrix[j,i] = sim

np.savetxt("output/sim_matrix.csv", sim_matrix, delimiter=",")

sim_matrix = np.genfromtxt('output/sim_matrix.csv', delimiter=',')

def get_source_id_for_title(title, sources_df):
    return sources_df[sources_df.Title == title].source_id.to_numpy()[0]

top10_dictionary = {}
for i, feed in enumerate(publisher_titles):
    sources_ranking = []
    source_id = get_source_id_for_title(feed, sources_df)

    for j in range(sim_matrix.shape[0]):
        if i == j:
            continue
        sources_ranking.append((publisher_titles[j], sim_matrix[i, j]))

    sources_ranking.sort(key=lambda x: -x[1])

    top10_dictionary[source_id] = [{'source': get_source_id_for_title(source[0], sources_df), 'score':source[1]} for source in sources_ranking[:10]]

with open('output/source_similarity_t10.json', 'w', encoding='utf-8') as f:
    json.dump(top10_dictionary, f, ensure_ascii=True, indent=4)

# PUSH source_similarity_t10, source.json and sim_matrix.csv
