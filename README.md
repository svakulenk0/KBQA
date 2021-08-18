# QAmp

[Svitlana Vakulenko, Javier D. Fernandez, Axel Polleres, Maarten de Rijke and Michael Cochez. Message Passing for Complex Question Answering over Knowledge Graphs. CIKM. 2019](https://arxiv.org/abs/1908.06917)


## Requirements

* Python 3.6
* tensorflow==1.11.0
* keras==2.2.4

* pyHDT (for accesssing the DBpedia Knowledge Graph)
* elasticsearch==5.5.3 (for indexing entities and predicate labels of the Knowledge Graph)

* pymongo (for storing the LC-QuAD dataset)
* flask (for the API)


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
4. Follow instructions in https://github.com/svakulenk0/hdt_tutorial to extract the list of entities (dbpedia201604_terms.txt) and predicates
5. Index entities and predicates into ElasticSearch
6. Download LC-QuAD dataset from http://lc-quad.sda.tech
7. Import LC-QuAD dataset into MongoDB

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

## Citation

```bibtex
@inproceedings{DBLP:conf/cikm/VakulenkoGPRC19,
  author    = {Svitlana Vakulenko and
               Javier David Fernandez Garcia and
               Axel Polleres and
               Maarten de Rijke and
               Michael Cochez},
  title     = {Message Passing for Complex Question Answering over Knowledge Graphs},
  booktitle = {Proceedings of the 28th {ACM} International Conference on Information
               and Knowledge Management, {CIKM} 2019, Beijing, China, November 3-7,
               2019},
  pages     = {1431--1440},
  year      = {2019},
  url       = {https://doi.org/10.1145/3357384.3358026},
  doi       = {10.1145/3357384.3358026},
  timestamp = {Mon, 04 Nov 2019 11:09:32 +0100}
}
```
