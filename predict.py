import os
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import OrderedDict

from tensorflow_serving.apis import input_pb2

import tensorflow_ranking as tfr

from official.nlp import bert
import official.nlp.bert.tokenization


max_seq_length = 64
bert_path = "./uncased_L-12_H-768_A-12/"
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(bert_path, "vocab.txt"),
     do_lower_case=True)

content_df = pd.read_csv("./data/content.csv")
all_content = content_df["movie_title"].values.tolist()
content_dict = dict(zip(all_content, content_df["title_description"].values.tolist()))


def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    try:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    except:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    try:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    except:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    try:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    except:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def context_example(device_id, platform, dma, region):
    feature = { 
            'user_id': _bytes_feature(bytes(str(user_id), 'utf-8')),
            'agegroup': _bytes_feature(bytes(str(agegroup), 'utf-8')),
            'occupation': _bytes_feature(bytes(str(occupation), 'utf-8')),
            'zipcode': _bytes_feature(bytes(str(zipcode), 'utf-8')),
            'sex': _bytes_feature(bytes(str(sex), 'utf-8'))
            }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def movie_example():
    EXAMPLES = []
    for v in all_content:
        title_description = content_dict[v].translate(str.maketrans('', '', string.punctuation)).replace("\n", "").replace("…", "").lower()

        stokens = tokenizer.tokenize(title_description)
        if len(stokens) + 2 > max_seq_length:
            stokens = stokens[:126] # length cutoff
        stokens = ["[CLS]"] + stokens + ["[SEP]"]
        input_ids = get_ids(stokens, tokenizer, max_seq_length)
        input_masks = get_masks(stokens, max_seq_length)
        input_segments = get_segments(stokens, max_seq_length)

        feature = {
                'movie_title': _bytes_feature(bytes(str(v), 'utf-8')),
                'input_ids': _int64_feature(input_ids),
                'input_masks': _int64_feature(input_masks),
                'input_segments': _int64_feature(input_segments),
                }
        EXAMPLES.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    return EXAMPLES

def process_df(df):
    """
    Takes pandas dataframe and returns lists of
    feature columns
    """

    movie_ids = [str(x) for x in df["movie_id"].values.tolist()]
    movie_titles = [str(x) for x in df["movie_title"].values.tolist()]
    title_descriptions = [str(x) for x in df["title_description"].values.tolist()]
    ratings = df["rating"].values.tolist()
    return movie_ids, movie_titles, title_descriptions, ratings

def create_records(df):
    """
    Takes a pandas dataframe and number of records to create and creates TFRecords.
    """
    all_users = list(set(df.device_id.values.tolist()))
    EXAMPLES = movie_example()
    all_records = []

    for user in all_users:
        user_df = df.loc[df["user_id"] == user]
        agegroup = user_df["agegroup"].values.tolist()[0]
        occupation = user_df["occupation"].values.tolist()[0]
        zipcode = user_df["zipcode"].values.tolist()[0]
        sex = user_df["sex"].values.tolist()[0]
        CONTEXT = context_example(user, agegroup, occupation, zipcode, sex)

        ELWC = input_pb2.ExampleListWithContext()
        ELWC.context.CopyFrom(CONTEXT)
        for example in EXAMPLES:
            example_features = ELWC.examples.add()
            example_features.CopyFrom(example)
            
        all_records.append(ELWC)
    return all_records

def process_predictions(preds, top_k):
    top_k_idx = np.array(preds).argsort()[-top_k:][::-1]
    top_k_content = [str(all_content[idx]) for idx in top_k_idx]
    return top_k_content


if __name__ == "__main__":
    import json
    import numpy
    import requests
    import base64

    top_k = 10
    data_path = "./data/sample_user.csv"
    df = pd.read_csv(data_path)
    sample_users = list(set(df["user_id"].values.tolist()))
    recs = create_records(df) 

    recs = [example.SerializeToString() for example in recs]
    recs = [base64.b64encode(example) for example in recs]
    recs = [example.decode('utf-8') for example in recs]
    print(len(sample_users))
    print(len(recs))

    recommendations = []
    for idx, rec in enumerate(recs):
        data = json.dumps({"signature_name": "serving_default",
            "instances": [{"b64": rec}]})
        headers = {"content-type": "application/json"}
        json_response = requests.post('http://localhost:8501/v1/models/tfrbert:predict',
                                      data=data, headers=headers)
        predictions = numpy.array(json.loads(json_response.text)["predictions"])[0]
        top_preds = process_predictions(predictions, top_k)
        recommendations.append(",".join(top_preds))

    recs_df = pd.DataFrame({"user_id": sample_users, "recommendations": recommendations})
    recs_df.to_csv("sample_recs.csv", index=False)
