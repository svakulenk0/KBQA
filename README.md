# KBQA

## Requirements

* Python 3.6
* Tensorflow
* Keras
* elasticsearch

## Datasets

* [LCQUAD](http://lc-quad.sda.tech) 5,000 pairs of questions and SPARQL queries

## Setup

It is not trivial to set up the environment. You need to:

1. Download DBPedia 2016-04 English HDT file and its index from http://www.rdfhdt.org/datasets/
2. Index entities and predicates into ElasticSearch
3. Download LC-QuAD dataset from http://lc-quad.sda.tech

<!-- 1. Create virtual environment and install all dependencies

'''
conda create -n tf36 python=3.6 pip
conda activate tf36
pip install -r requirements.txt
'''

2. Download and make [fastText](https://github.com/facebookresearch/fastText), load the English model trained on Wikipedia and generate fastText embeddings:

'''
cd data
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
unzip wiki.en.zip
rm wiki.en.zip
'''

./fasttext print-word-vectors ../KBQA/data/fasttext/wiki.en.bin < ../KBQA/data/test_question_words.txt > ../KBQA/data/test_question_words_fasttext.txt

3. Run MongoDB

sudo service mongod start -->


## Run

see notebooks

## Results

TODO
