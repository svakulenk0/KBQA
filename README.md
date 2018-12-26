# KBQA

## Requirements

* Python 3.6
* Tensorflow
* Keras


## Setup

1. Create virtual environment and install all dependencies

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


## Run

see notebooks

## Results

TODO


## References

* [Relational Graph Convolutional Networks (RGCN)](https://github.com/tkipf/relational-gcn)

## Troubleshooting

* Make sure you have the correct Keras version and the backend is set to Tensorflow (use vim ~/.keras/keras.json to specify the backend) python -c 'import keras; print(keras.__version__)'

* Exception: fastText: Cannot load /data/fasttext/wiki.en.bin due to C++ extension failed to allocate the memory
