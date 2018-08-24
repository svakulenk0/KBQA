# KBQA-RGCN
Knowledge Base Question Answering (KBQA) using Relational Graph Convolutional Networks (RGCN)


## Requirements

(baseline: [Python 3.6.4](https://www.python.org/downloads/) [keras==2.1.6](https://keras.io/), which requires another backend lib. We used [Tensorflow 1.8.0](https://www.tensorflow.org/))

* theano==0.9.0
* keras==1.2.1
* [rgcn](https://github.com/tkipf/relational-gcn)


## References

* [Keras-based implementation of Relational Graph Convolutional Networks](https://github.com/tkipf/relational-gcn)
* [Simple seq2seq model as a baseline model for DSTC7 Task 2](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/baseline/baseline.py)


## Troubleshooting

* Cleaning Theano cache: [rm -rf ~/.theano](https://stackoverflow.com/questions/43312593/theano-importerror-cannot-import-name-inplace-increment)
