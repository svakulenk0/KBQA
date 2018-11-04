# KBQA

## Requirements

* Python 2.7
* keras
* Tensorflow


## Setup

1. Create virtual environment and install all dependencies

'''
conda create -n tensorflow2 python=2.7 pip
conda activate tensorflow2
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

./fasttext print-word-vectors ../KBQA/data/fasttext/wiki.en.bin < ../KBQA/data/test_question_words.txt > ../KBQA/data/test_question_words_fasttext.txt


3. Download KGlove embeddings

wget http://data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/11_pageRankSplit


## Run

Training the model:

python kbqa_model4.py train

Test the model with:

python kbqa_model4.py test

## Results

Baseline on LC-Quad SELECT subset Hits@5: 15/737

## References

* [Relational Graph Convolutional Networks (RGCN)](https://github.com/tkipf/relational-gcn)
* [Simple seq2seq model as a baseline model for DSTC7 Task 2](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/baseline/baseline.py)


## Troubleshooting

* Make sure you have the correct Keras version and the backend is set to Tensorflow (use vim ~/.keras/keras.json to specify the backend) python -c 'import keras; print(keras.__version__)'

* Exception: fastText: Cannot load /data/fasttext/wiki.en.bin due to C++ extension failed to allocate the memory
