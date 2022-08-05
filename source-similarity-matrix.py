from cmath import nan
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import math
import hashlib
import requests
import json

# Take centroid of 512-d embeddings
def get_source_representation_from_text(source):
    source_repr = np.zeros((1,512))
    for item in source:
        source_repr += embed([item])[0]
    norm_repr = tf.nn.l2_normalize(source_repr/len(source), axis=1)
    return norm_repr.numpy()

def compute_source_similarity(source1, source2, t='dot'):
    cosine_similarities = np.dot(source1, np.transpose(source2))
    clip_cosine_similarity = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    score = 1.0 - tf.acos(clip_cosine_similarity) / math.pi
    return score

def compute_source_representation_from_articles(articles_df, publisher_id):
    publisher_bucket_df = articles_df[articles_df.publisher_id == publisher_id]

    source_titles = [title for title in publisher_bucket_df.title.to_numpy() if title != None]
    return get_source_representation_from_text(source_titles)

print("Started computing similarity matrix...")
# TODO(): PULL article_history.csv

output_dir = 'output/'
source_url = 'https://brave-today-cdn.bravesoftware.com/sources.en_US.json'

source_r = requests.get(source_url)
sources_json = source_r.json()
sources_df = pd.json_normalize(sources_json)
sources_df["source_representation"] = np.nan

articles_df = pd.read_csv(output_dir + 'articles_history.csv', header=None)
articles_df.columns = ['title', 'description', 'timestamp', 'publisher_id']

print("Loading Universal Sentence Encoder...")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print("USE (%s) loaded" % module_url)
def embed(input):
    return model(input)

print("Building sources embeddings...")
publisher_ids = sources_df.publisher_id.to_numpy()
print("Publisher ids size: ", publisher_ids.size)
reprs = np.zeros((publisher_ids.size, 512))
for i, publisher_id in tqdm(enumerate(publisher_ids)):
    reprs[i,:] = compute_source_representation_from_articles(articles_df, publisher_id)

sources_representation = pd.DataFrame({'publisher_id': publisher_ids})
sources_representation =  pd.concat([sources_representation, pd.DataFrame(reprs)], axis=1)
sources_representation.to_csv('output/source_embeddings.csv', header=None)

print("Finished building source embeddings.")

sim_matrix = np.zeros((publisher_ids.size, publisher_ids.size))
for i in range(publisher_ids.size):
    for j in range(i+1, publisher_ids.size):
        repr_i = reprs[i]
        repr_j = reprs[j]
        sim = compute_source_similarity(repr_i, repr_j)
        sim_matrix[i,j] = sim
        sim_matrix[j,i] = sim

def get_source_id_for_title(title, sources_df):
    return sources_df[sources_df.publisher_name == title].publisher_id.to_numpy()[0]

print("Finished computing similarity matrix. Outputting results...")

publisher_titles = sources_df.publisher_name.to_numpy()
print("Publisher titles size: ", publisher_titles.size)
top10_dictionary = {}
top10_dictionary_human_readable = {}
for i, feed in enumerate(publisher_titles):
    sources_ranking = []
    source_id = get_source_id_for_title(feed, sources_df)

    for j in range(sim_matrix.shape[0]):
        if i == j or math.isnan(sim_matrix[i,j]):
            continue
        sources_ranking.append((publisher_titles[j], sim_matrix[i, j]))

    sources_ranking.sort(key=lambda x: -x[1])

    top10_dictionary[source_id] = [{'source': get_source_id_for_title(source[0], sources_df), 'score':source[1]} for source in sources_ranking[:10]]
    top10_dictionary_human_readable[feed] = [{'source': source[0], 'score':source[1]} for source in sources_ranking[:10]]

with open('output/source_similarity_t10.json', 'w', encoding='utf-8') as f:
    json.dump(top10_dictionary, f, ensure_ascii=True, indent=4)

with open('output/source_similarity_t10_hr.json', 'w', encoding='utf-8') as f:
    json.dump(top10_dictionary_human_readable, f, ensure_ascii=True, indent=4)

print("Script has finished running.")

# PUSH source_similarity_t10.json, source_embeddings.csv
