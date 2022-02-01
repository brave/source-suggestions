import sys
import os
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import math

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

# PULL source.csv and source_buckets/

print("Started computing similarity matrix...")
sources_df = pd.read_csv('output/sources.csv', header=None)
sources_df["source_representation"] = np.nan
publisher_ids = sources_df.iloc[:, 1].to_numpy()
reprs = np.zeros((publisher_ids.size, 512))
for i, publisher_id in tqdm(enumerate(publisher_ids)):
    source_bucket_df = pd.read_csv("source_buckets/{}.csv".format(publisher_id), header=None)
    source_name = sources_df[sources_df.iloc[:,1] == publisher_id].iloc[0]
    source_titles = source_bucket_df.iloc[:,0].to_numpy()
    source_repr = source_representation(source_titles).numpy()
    reprs[i,:] = source_repr
sources_representation = pd.DataFrame({'publisher':publisher_ids})
sources_representation =  pd.concat([sources_representation, pd.DataFrame(reprs)], axis=1)
sources_representation.to_csv('output/source_embeddings.csv', header=None)

sim_matrix = np.zeros((publisher_ids.size, publisher_ids.size))
for i in range(publisher_ids.size):
    for j in range(i+1, publisher_ids.size):
        repr_i = reprs[i]
        repr_j = reprs[j]
        sim = compute_source_similarity(repr_i, repr_j)
        sim_matrix[i,j] = sim
        sim_matrix[j,i] = sim

np.savetxt("output/sim_matrix.csv", sim_matrix, delimiter=",")
