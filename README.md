# TF-Ranking + BERT demo using MovieLens data

This is the sample code for the [blog](https://smellslikeml.com) based off the [extension example](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/extension/examples/tfrbert_example.py) in the official [TF-Ranking repo](https://github.com/tensorflow/ranking).

## Setup
Install the python requirements found in the requirements.txt file
```bash
$ pip install -r requirements.txt
```

Download the [official BERT checkpoints](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12.tar.gz) for use in tensorflow 2+ and unzip into this directory.
```bash
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12.tar.gz
$ tar -xvzf uncased_L-12_H-768_A-12.tar.gz 
``` 

For this demo, we enriched the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/) with movie title descriptions. You can download this set from [here](https://drive.google.com/file/d/1QiNMOhwEqd8-aoQAWYwKr1TJfP_xCUN0/view?usp=sharing). Extract the zip file inside the ```data/``` directory.
```bash
$ unzip movieLens_tfrank.zip -d ./data/
```

Run the tfrecord script to create tfrecords from the downloaded movielens data.
```bash
$ python create_tfrecords.py
```

## Train and evaluate

Run the train script to train and evaluate the model. The script will create logs that can be visualized by Tensorboard. This script includes default values for all flag variables but feel free to experiment.
```bash
# run with default parameters
$ python train.py
```

or 

```bash
$ python train.py --train_input_pattern=tfrecords/train.tfrecord \
   --eval_input_pattern=tfrecords/test.tfrecord \
   --vocab_input_pattern=tfrecords/vocab.tfrecord \
   --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
   --bert_init_ckpt=uncased_L-12_H-768_A-12/bert_model.ckpt \
   --bert_max_seq_length=64 \
   --model_dir="models/" \
   --loss=softmax_loss \
   --train_batch_size=1 \
   --eval_batch_size=1 \
   --learning_rate=1e-5 \
   --num_train_steps=50 \
   --num_eval_steps=10 \
   --checkpoint_secs=120 \
   --num_checkpoints=20
```

## Generate Recommendations

Run the predict script to generate recommendations for a set of users. We've included a sample csv of user data for which to run inference.
```bash
$ python predict.py
```
