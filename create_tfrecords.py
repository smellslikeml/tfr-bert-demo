import os
import pandas as pd
import tensorflow as tf
import numpy as np
import string

from tensorflow_serving.apis import input_pb2

from official.nlp import bert
import official.nlp.bert.tokenization

max_seq_length = 64
bert_path = "./uncased_L-12_H-768_A-12/"
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(bert_path, "vocab.txt"),
     do_lower_case=True)

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

def context_example(user_id, agegroup, occupation, zipcode, sex):
    feature = {
            'user_id': _bytes_feature(bytes(str(user_id), 'utf-8')),
            'agegroup': _bytes_feature(bytes(str(agegroup), 'utf-8')),
            'occupation': _bytes_feature(bytes(str(occupation), 'utf-8')),
            'zipcode': _bytes_feature(bytes(str(zipcode), 'utf-8')),
            'sex': _bytes_feature(bytes(str(sex), 'utf-8'))
            }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def movie_example(movie_id, movie_title, title_description, rating):
    title_description = title_description.translate(str.maketrans('', '', string.punctuation)).replace("\n", "").replace("…", "").lower()
    title_description_bytes = [bytes(x, 'utf-8') for x in title_description.split(" ")]

    stokens = tokenizer.tokenize(title_description)
    if len(stokens) + 2 > max_seq_length:
        stokens = stokens[:max_seq_length-2] # length cutoff
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    input_ids = get_ids(stokens, tokenizer, max_seq_length)
    input_masks = get_masks(stokens, max_seq_length)
    input_segments = get_segments(stokens, max_seq_length)

    feature = {
            'movie_id': _bytes_feature(bytes(movie_id, 'utf-8')),
            'movie_title': _bytes_feature(bytes(movie_title, 'utf-8')),
            'title_description': _bytes_feature(title_description_bytes),
            'input_ids': _int64_feature(input_ids),
            'input_masks': _int64_feature(input_masks),
            'input_segments': _int64_feature(input_segments),
            'relevance': _int64_feature(rating),
            }
    return tf.train.Example(features=tf.train.Features(feature=feature))

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

def create_records(df, output_dir, num_of_records=5, prefix="movielens_"):
    """
    Takes a pandas dataframe and number of records to create and creates TFRecords.
    Saves records in output_dir
    """
    all_users = list(set(df.user_id.values.tolist()))

    record_prefix = os.path.join(output_dir, prefix)
    files_per_record = int(len(all_users) / num_of_records)  #approximate number of examples per record
    chunk_number = 0

    for i in range(0, len(all_users), files_per_record):
        print("Writing chunk ", str(chunk_number))
        user_chunk = all_users[i:i+files_per_record]

        if num_of_records == 1:
            record_file = record_prefix + ".tfrecords"
        else:
            record_file = record_prefix + str(chunk_number).zfill(3) + ".tfrecords"

        with tf.io.TFRecordWriter(record_file) as writer:
            for user in user_chunk:
                user_df = df.loc[df["user_id"] == user]
                agegroup = user_df["agegroup"].values.tolist()[0]
                occupation = user_df["occupation"].values.tolist()[0]
                zipcode = user_df["zipcode"].values.tolist()[0]
                sex = user_df["sex"].values.tolist()[0]
                CONTEXT = context_example(user, agegroup, occupation, zipcode, sex)

                EXAMPLES = []
                movie_ids, movie_titles, title_descriptions, ratings = process_df(user_df)
                for i in range(len(movie_ids)):
                    EXAMPLES.append(movie_example(movie_ids[i], movie_titles[i], title_descriptions[i], ratings[i]))

                ELWC = input_pb2.ExampleListWithContext()
                ELWC.context.CopyFrom(CONTEXT)
                for example in EXAMPLES:
                    example_features = ELWC.examples.add()
                    example_features.CopyFrom(example)

                writer.write(ELWC.SerializeToString())
            chunk_number += 1

def create_vocab(df, vocab_path, columns=['agegroup', 'occupation', 'sex']):
    # Filling in nans
    vocab = []
    for col in columns:
        v = list(set(df[col].values.tolist()))
        v = [str(x) for x in v]
        vocab += v
            
    vocab = list(set(vocab))
    with open(vocab_path, "w") as f:
        for v in vocab:
            if v != "":
                f.write(v + "\n")

def read_and_print_tf_record(target_filename, num_of_examples_to_read):
    filenames = [target_filename]
    tf_record_dataset = tf.data.TFRecordDataset(filenames)
    all_examples = []
    
    for raw_record in tf_record_dataset.take(num_of_examples_to_read):
        example_list_with_context = input_pb2.ExampleListWithContext()
        example_list_with_context.ParseFromString(raw_record.numpy())
        all_examples.append(example_list_with_context)

    return all_examples

if __name__ == "__main__":
    train_df = pd.read_csv("./data/movieLens_train.csv")
    test_df = pd.read_csv("./data/movieLens_test.csv")
    all_df = pd.concat([train_df, test_df])
    output_dir = "./tfrecords/"

    write_records = False

    if write_records:
        create_vocab(all_df, output_dir + "vocab.txt")
            
        num_of_records = 1
        create_records(train_df, output_dir, num_of_records, prefix="train")
        create_records(test_df, output_dir, num_of_records, prefix="test")
    else:
        examples = read_and_print_tf_record(output_dir+"train.tfrecords", 1)
        print(examples)
