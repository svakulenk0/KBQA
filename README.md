# KBQA

## Requirements

* Python 3.6
* tensorflow==1.11.0
* keras==2.2.4
* elasticsearch==5.5.3
* pymongo
* pyHDT

## Datasets

* [LCQUAD](http://lc-quad.sda.tech) 5,000 pairs of questions and SPARQL queries

## Setup

It is not trivial to set up the environment. You need to:

1. Create virtual environment and install all dependencies (to install CUDA, TF, Keras and friends follow https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25)

```
conda create -n kbqa python=3.6 pip
conda activate kbqa
pip install -r requirements.txt
```

2. Install HDT API:

```
git clone https://github.com/webdata/pyHDT.git
cd pyHDT/
./install.sh
```

3. Download DBPedia 2016-04 English HDT file and its index from http://www.rdfhdt.org/datasets/
4. Index entities and predicates into ElasticSearch
5. Download LC-QuAD dataset from http://lc-quad.sda.tech
6. Import LC-QuAD dataset into MongoDB

```
sudo service mongod start
```


<!-- 
2. Download and make [fastText](https://github.com/facebookresearch/fastText), load the English model trained on Wikipedia and generate fastText embeddings:

'''
cd data
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
unzip wiki.en.zip
rm wiki.en.zip
'''

./fasttext print-word-vectors ../KBQA/data/fasttext/wiki.en.bin < ../KBQA/data/test_question_words.txt > ../KBQA/data/test_question_words_fasttext.txt

 -->


## Run

see notebooks

## Benchmark

python final_benchmark_results.py


## Results

TODO
