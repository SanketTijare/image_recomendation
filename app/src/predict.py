#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
# @Time    : 8/4/2021 6:16 PM
# @Author  : sanket.tijare@icertis.com
# @File    : predict.py
"""
from train import SetSeed, Encoder, extractor
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

SS = SetSeed()
SS.setter()
model = Encoder()

DB_PATH = '../db/'


def sim_sort(x):
    return x['similarity']


def predict(image_path, sku_type):
    bottleneck_features = extractor(image_path, model, False)
    interested_db = os.path.join(DB_PATH, sku_type + ".pkl")

    data = pd.read_pickle(interested_db)
    data_list_dict = data.to_dict('records')
    output = []
    for feat in data_list_dict:
        similarity = cosine_similarity(bottleneck_features.reshape(1, -1), feat['features'])
        if similarity >=0.80:
            output.append({
                "filename": feat['filename'],
                "similarity": similarity[0][0]
            })
    if output:
        output.sort(key=sim_sort, reverse=True)
        return output
    else:
        return None


if __name__ == "__main__":
    img_path = r"C:\MyDrive\work\self_project\image_recomendation\data\boots\7688415.411.jpg"
    sku = 'boots'
    op = predict(img_path, sku)
    print(op)