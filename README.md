# KBQA

## Requirements

* Python 3
* keras==2.2.2
* Tensorflow
* [fastText](https://github.com/facebookresearch/fastText/tree/master/python)


## Setup

Make sure you have the correct Keras version and the backend is set to Tensorflow (use vim ~/.keras/keras.json to specify the backend)

python -c 'import keras; print(keras.__version__)'


## Run

Training the model:

python rgcn_kbqa2.py train

## Results

Baseline on LC-Quad SELECT subset Hits@5: 15/737

## References

* [Relational Graph Convolutional Networks (RGCN)](https://github.com/tkipf/relational-gcn)
* [Simple seq2seq model as a baseline model for DSTC7 Task 2](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/baseline/baseline.py)


## Troubleshooting

* Cleaning Theano cache: [rm -rf ~/.theano](https://stackoverflow.com/questions/43312593/theano-importerror-cannot-import-name-inplace-increment)

* Exception: fastText: Cannot load /data/fasttext/wiki.en.bin due to C++ extension failed to allocate the memory
