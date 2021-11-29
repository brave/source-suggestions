import sys
from numpy import genfromtxt
import pandas as pd
import numpy as np
import json

feeds = pd.read_csv("output/feeds.csv", header=None)
feeds_ids = list(feeds.iloc[:,1].to_numpy())
feeds_titles = list(feeds.iloc[:,0].to_numpy())

sim_matrix = genfromtxt('output/sim_matrix.csv', delimiter=',')

feed_dictionary = []
for i, row in feeds.iterrows():
    feed_dictionary.append({
        'name':row[0],
        'id': row[1]
    })

json_object = {'data':feed_dictionary}

with open('output/sources.json', 'w', encoding='utf-8') as f:
    json.dump(json_object, f, ensure_ascii=True, indent=4)

top10_dictionary = {}
for i, feed in enumerate(feeds_titles):
    sources_ranking = []
    for j in range(sim_matrix.shape[0]):
        if i == j:
            continue
        sources_ranking.append((feeds_titles[j], sim_matrix[i, j]))

    sources_ranking.sort(key=lambda x: -x[1])

    top10_dictionary[feed] = [{'source':source[0], 'score':source[1]} for source in sources_ranking[:10]]

with open('output/sim_matrix_top10.json', 'w', encoding='utf-8') as f:
    json.dump(top10_dictionary, f, ensure_ascii=True, indent=4)
